"""地理领域"由已知点推未知点"的两个对照（科研推进/参照推测方向对照补测_20261111/方案与事先判定.md）。不改默认流程。
  python scripts/run_spatial.py --k 0 --seed 1 --select   # K=0：RFsp 变体在支持集留出上选
  python scripts/run_spatial.py --k 1 --seed 2
  python scripts/run_spatial.py --report
rfsp   Hengl 等 PeerJ 2018：到每个参照的距离当特征（每个参照一列）＋可选协变量，概率随机森林；只用本区参照。
block  道路街区多数票：道路线切街区，取街区内参照的多数学校；没有就抄最近已知。
输出 outputs/main_pre_v1/k<K>_s<seed>__spatial/<rfsp|block>[_policy]/probs/<区>.npz、eval.json；chosen.json。
"""
import sys, json, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import shapely
from shapely import STRtree
from sklearn.ensemble import RandomForestClassifier
from zoning import evaluate, baselines as B
from zoning.evidence import lines
from zoning.baselines import constrain
from run_matching import setup, BATCHES

ROOT = REPO / 'outputs/main_pre_v1'; DOC = REPO.parent / '科研推进' / '参照推测方向对照补测_20261111'
SEEDS = [20261221, 20261222, 20261223]; RF_VARIANTS = ['dist', 'dist+geometry', 'dist+policy']
ROAD_SETS = {'major': ('major', 'rail', 'river'), 'major+secondary': ('major', 'secondary', 'rail', 'river'), 'major+secondary+tertiary': ('major', 'secondary', 'tertiary', 'rail', 'river')}


# ---------------- RFsp ----------------
def rf_features(dist, rules_out, variant, ref_ids):
    Dm = np.linalg.norm(dist.coords[:, None] - dist.coords[ref_ids][None], axis=-1) / 1000
    if variant == 'dist': return Dm
    cov, _ = B.node_table(dist, rules_out, variant.split('+')[1]); return np.concatenate([Dm, cov], 1)


def rf_fit_predict(dist, rules_out, variant, train_ids, y, query_ids, seeds):
    X = rf_features(dist, rules_out, variant, train_ids); P = np.zeros((len(query_ids), dist.school_n))
    for s in seeds:
        m = RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=8, random_state=s).fit(X[train_ids], y[train_ids])
        P[:, m.classes_] += m.predict_proba(X[query_ids]) / len(seeds)
    return P


def rfsp(dist, rules_out, variant):
    sup = B.support(dist, rules_out); y = np.zeros(dist.n, int)
    for q, s in sup.items(): y[q] = s
    ids = B.one_per_group(dist, list(sup)); n = dist.n
    if len(ids) == 0: return B.nearest_school(dist, rules_out)
    P = rf_fit_predict(dist, rules_out, variant, ids, y, np.arange(n), SEEDS)
    cg = set(dist.groups[ids].tolist()); cset = set(ids.tolist()); ev = {e['q'] for e in evaluate.eval_rows(dist, rules_out.known)}
    clash = [q for q in sorted(ev) if dist.groups[q] in cg and q not in cset]                      # 评价小区与参照同楼盘：去掉那个参照重训
    for g in sorted({dist.groups[q] for q in clash}):
        qs = np.array([q for q in clash if dist.groups[q] == g], int); P[qs] = rf_fit_predict(dist, rules_out, variant, ids[dist.groups[ids] != g], y, qs, SEEDS)
    return P


def rfsp_select(dist, rules_out):
    sup = B.support(dist, rules_out); y = np.zeros(dist.n, int)
    for q, s in sup.items(): y[q] = s
    ids = B.one_per_group(dist, list(sup)); tr, va = B.split_validation(dist, sup, ids); trials = {}
    for v in RF_VARIANTS:
        trials[v] = float((rf_fit_predict(dist, rules_out, v, tr, y, va, SEEDS[:1]).argmax(1) == y[va]).mean()) if len(va) and len(tr) else 0.
    best = max(RF_VARIANTS, key=lambda v: (round(trials[v], 9), -RF_VARIANTS.index(v)))
    return dict(variant=best, val_acc=trials[best], trials=trials, n_val=int(len(va)))


# ---------------- 道路街区多数票 ----------------
_blocks = {}


def block_ids(dist, road_set):
    key = (dist.name, road_set)
    if key in _blocks: return _blocks[key]
    ls = lines(dist.lines_key or dist.city); lo = dist.coords.min(0) - 1500; hi = dist.coords.max(0) + 1500; box = shapely.box(*lo, *hi)
    sel = np.array([i for i, c in enumerate(ls.cls) if c in ROAD_SETS[road_set]], int); g = ls.geoms[sel]; g = g[shapely.intersects(g, box)]
    merged = shapely.unary_union(list(shapely.intersection(g, box)) + [box.boundary]); polys = np.array(list(shapely.polygonize([merged]).geoms), dtype=object)
    tree = STRtree(polys); pts = shapely.points(dist.coords); pi, gi = tree.query(pts, predicate='within'); blk = np.full(dist.n, -1); blk[pi] = gi
    _blocks[key] = blk; return blk


def block_predict(dist, rules_out, blk, refs, query_ids):
    """refs：可用参照小区。返回 [len(query_ids), S]。"""
    S = dist.school_n; P = np.zeros((len(query_ids), S)); refs = np.asarray(refs, int)
    near = B.nearest_school(dist, rules_out)
    for i, q in enumerate(query_ids):
        ok = refs[dist.groups[refs] != dist.groups[q]] if len(refs) else refs
        if not len(ok): P[i] = near[q]; continue
        dd = np.linalg.norm(dist.coords[ok] - dist.coords[q], axis=1); inb = ok[blk[ok] == blk[q]] if blk[q] >= 0 else ok[:0]
        if len(inb):
            v = np.zeros(S)
            for r in inb:
                for s in rules_out.known[int(r)]: v[s] += 1 / len(rules_out.known[int(r)])
            top = np.flatnonzero(v >= v.max() - 1e-9)
            if len(top) > 1:                                                                      # 并列：取并列学校里最近参照的学校
                dmin = [min(dd[j] for j, r in enumerate(ok) if blk[r] == blk[q] and s in rules_out.known[int(r)]) for s in top]; top = top[[int(np.argmin(dmin))]]
            P[i, top[0]] = 1
        else:
            r = int(ok[dd.argmin()])
            for s in rules_out.known[r]: P[i, s] += 1 / len(rules_out.known[r])
    return P


def block_majority(dist, rules_out):
    refs = np.array(sorted(q for q in rules_out.known if q not in dist.not_reference), int)
    sup = B.support(dist, rules_out); ids = B.one_per_group(dist, list(sup)); tr, va = B.split_validation(dist, sup, ids)
    hide = set(dist.groups[va].tolist()); keep = np.array([q for q in refs if dist.groups[q] not in hide], int); trials = {}
    for rs in ROAD_SETS:
        blk = block_ids(dist, rs); trials[rs] = float(np.mean(block_predict(dist, rules_out, blk, keep, va).argmax(1) == np.array([sup[q] for q in va]))) if len(va) else 0.
    names = list(ROAD_SETS); best = max(names, key=lambda r: (round(trials[r], 9), -names.index(r))); blk = block_ids(dist, best)
    P = block_predict(dist, rules_out, blk, refs, np.arange(dist.n))
    inblock = float(np.mean([blk[e['q']] >= 0 and any(blk[r] == blk[e['q']] and dist.groups[r] != dist.groups[e['q']] for r in refs) for e in evaluate.eval_rows(dist, rules_out.known)])) if len(refs) else 0.
    return P, dict(road_set=best, val_acc=trials[best], trials=trials, eval_with_block_refs=inblock)


def run(k, seed, select):
    t0 = time.time(); c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; out = ROOT / f'{name}__spatial'; out.mkdir(parents=True, exist_ok=True)
    n_eval = sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in D); ref_n = json.loads((ROOT / name / 'eval.json').read_text(encoding='utf-8'))['summary']['两城']['n']
    assert n_eval == ref_n, f'评价集与主表不一致 {n_eval} vs {ref_n}'
    if select:
        ch = {d: rfsp_select(D[d], R[d]) for d in D}; (out / 'chosen_rfsp.json').write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding='utf-8')
        for d, x in ch.items(): print('选 RFsp', d, x['variant'], f"{x['val_acc']:.3f}", x['trials'], flush=True)
    ch = json.loads((ROOT / 'k0_s1__spatial' / 'chosen_rfsp.json').read_text(encoding='utf-8')); rows = {}; binfo = {}
    for d in D:
        er = evaluate.eval_rows(D[d], R[d].known); Pb, binfo[d] = block_majority(D[d], R[d])
        for m_, P in (('rfsp', rfsp(D[d], R[d], ch[d]['variant'])), ('block', Pb)):
            for suf, PP in (('', P), ('_policy', constrain(P, R[d]))):
                od = out / (m_ + suf) / 'probs'; od.mkdir(parents=True, exist_ok=True); np.savez_compressed(od / f'{d}.npz', probs=PP[None].astype(np.float32))
                rows.setdefault(m_ + suf, []).extend(evaluate.correct(PP, D[d], er))
        print(name, d, binfo[d]['road_set'], f"街区内有参照的评价题 {binfo[d]['eval_with_block_refs']:.2f}", f'{time.time() - t0:.0f}s', flush=True)
    (out / 'chosen_block.json').write_text(json.dumps(binfo, ensure_ascii=False, indent=1), encoding='utf-8')
    for m_, rr in rows.items():
        s_ = evaluate.summary(rr); (out / m_ / 'eval.json').write_text(json.dumps(dict(summary=s_, k=k, seed=seed), ensure_ascii=False, indent=1), encoding='utf-8')
        print(m_, {kk: round(s_[kk]['accuracy'], 2) for kk in ['两城', '长沙', '杭州']}, flush=True)


CANDS = {'抄最近已知': ['__baselines/copy_nearest'], '标签传播': ['__baselines/label_propagation'], 'Matching Networks 原版': ['__matching/mn', '__matching/mn_policy'],
         'ProtoNet（只本区）': ['__baselines/protonet_raw', '__baselines/protonet_policy'], 'ModernNCA 原版（只本区）': ['__baselines/modernnca_raw', '__baselines/modernnca_policy'],
         'GraphAny': ['__graphany/ga', '__graphany/ga_policy'], 'RFsp': ['__spatial/rfsp', '__spatial/rfsp_policy'], '道路街区多数票': ['__spatial/block', '__spatial/block_policy'],
         'Matching Networks 成对适配版': ['__matching/mn_pair', '__matching/mn_pair_policy'], 'ModernNCA 同表版': ['__tabular/mnca_rows', '__tabular/mnca_rows_policy']}
OTHERS = {'完整模型': [''], '直线最近学校（不用参照）': ['__baselines/nearest_school'], '梯度提升树同表版（通用学习器）': ['__gbt/gbt', '__gbt/gbt_policy']}


def report():
    res = {}; acc = lambda r: float(np.mean([x['ok'] for x in r]) * 100); allrows = {}
    for k, seed in BATCHES:
        c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; ev = c['evaluate']

        def rows_of(odir):
            rr = []
            for d in D:
                f = ROOT / odir / 'probs' / f'{d}.npz'
                if not f.exists(): return None
                rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
            return rr
        meth = {}
        for m_, paths in {**OTHERS, **CANDS}.items():
            vs = [v for v in (rows_of(name + p) for p in paths) if v is not None]; meth[m_] = max(vs, key=acc)      # 原样/加排除取评价上较高者
        allrows[name] = (meth, ev)
        res[name] = dict(n=len(meth['完整模型']), accuracy={m_: evaluate.summary(v) for m_, v in meth.items()})
    k1 = [n for n in res if n.startswith('k1')]; mean1 = {m_: float(np.mean([res[n]['accuracy'][m_]['两城']['accuracy'] for n in k1])) for m_ in CANDS}
    order = sorted(CANDS, key=lambda m_: -mean1[m_]); cmp = {}
    pairs = [('完整模型', 'RFsp'), ('完整模型', '道路街区多数票'), ('RFsp', '抄最近已知'), ('道路街区多数票', '抄最近已知')] + [(order[i], order[i + 1]) for i in range(len(order) - 1)] + [('完整模型', order[0])]
    for a, b in pairs:
        cmp[f'{a} − {b}'] = {n: evaluate.compare(allrows[n][0][a], allrows[n][0][b], allrows[n][1]) for n in res}
    (DOC / '结果.json').write_text(json.dumps(dict(accuracy=res, k1_mean=mean1, order=order, compare=cmp), ensure_ascii=False, indent=1), encoding='utf-8')
    L = ['# "根据参照推测"方向对照：结果（pre_v1 开发数字）', '', '## 准确率（两城｜长沙｜杭州；各方法取原样/加排除较高者）', '', '| 方法 | ' + ' | '.join(res) + ' | K=1 平均 |', '|---|' + '---|' * (len(res) + 1)]
    for m_ in list(OTHERS) + order:
        a = [res[n]['accuracy'][m_] for n in res]; mk1 = np.mean([res[n]['accuracy'][m_]['两城']['accuracy'] for n in k1])
        L.append(f'| {m_} | ' + ' | '.join(f"{x['两城']['accuracy']:.2f}｜{x['长沙']['accuracy']:.2f}｜{x['杭州']['accuracy']:.2f}" for x in a) + f' | {mk1:.2f} |')
    L += ['', '## 比较（两城差值 [95% 区间]；长沙；杭州）', '']
    for kk, v in cmp.items():
        L.append(f'### {kk}'); L.append('')
        for n, x in v.items(): L.append(f"- {n}：两城 {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]（{x['两城']['fixed']}/{x['两城']['broken']}）；长沙 {x['长沙']['delta']:+.2f} [{x['长沙']['ci'][0]:+.2f}, {x['长沙']['ci'][1]:+.2f}]；杭州 {x['杭州']['delta']:+.2f} [{x['杭州']['ci'][0]:+.2f}, {x['杭州']['ci'][1]:+.2f}]")
        L.append('')
    ch = json.loads((ROOT / 'k0_s1__spatial' / 'chosen_rfsp.json').read_text(encoding='utf-8')); L += ['## RFsp 选中的变体（K=0 支持集留出）', ''] + [f"- {d}：{x['variant']}（{x['trials']}）" for d, x in ch.items()]
    cb = json.loads((ROOT / 'k1_s1__spatial' / 'chosen_block.json').read_text(encoding='utf-8')); L += ['', '## 街区多数票（K=1 第 1 批）', ''] + [f"- {d}：{x['road_set']}，评价题里街区内有参照的占 {x['eval_with_block_refs']:.2f}" for d, x in cb.items()]
    (DOC / '结果.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int); ap.add_argument('--select', action='store_true'); ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report: report()
    else: run(a.k, a.seed, a.select)


if __name__ == '__main__':
    main()
