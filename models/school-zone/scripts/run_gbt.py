"""同特征梯度提升树对照（科研推进/梯度提升树同条件验证_20261111/方案与事先判定.md；照 科研推进/结构减法_20261106/gbt.py 移植）。不改默认流程。
  python scripts/run_gbt.py --k 0 --seed 1 --select   # K=0：每个目标区在验证区上选超参数，再全来源重训
  python scripts/run_gbt.py --k 1 --seed 2            # K>0：沿用 K=0 选好的超参数
  python scripts/run_gbt.py --report                  # 汇总四批，写到 科研推进 目录
每个"小区×候选学校"一行：学校一路证据、该校周边统计（同 model.context）、该校最近可用参照的成对证据、最近/次近参照距离、200/500/1000 米内参照数、
两种距离排名、有没有参照。候选＝min(最近参照距离, 到校距离) 最小的 10 所。训练题同主模型 train.items，同楼盘等权。
输出 outputs/main_pre_v1/k<K>_s<seed>__gbt/<gbt|gbt_orig>[_policy]/probs/<区>.npz、eval.json；chosen.json；recall.json。
"""
import sys, json, time, argparse, itertools
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from zoning import evaluate, evidence, model as MD, matching as M
from zoning.baselines import constrain
from run_matching import setup, BATCHES

ROOT = REPO / 'outputs/main_pre_v1'; K_CAND = 10
GRID = [dict(learning_rate=lr, max_leaf_nodes=lf, max_iter=it) for lr, lf, it in itertools.product((0.03, 0.1), (15, 31, 63), (200, 600))]
ORIG = dict(learning_rate=0.05, max_leaf_nodes=31, max_iter=300)            # 11-06 gbt.py 的超参数
DOC = REPO.parent / '科研推进' / '梯度提升树同条件验证_20261111'


def features(x, ids, chunk=128):
    """返回候选 [len(ids), 10] 和特征 [len(ids), 10, F]。"""
    Mem = x.membership[:len(x.ref_sites)].numpy().astype(bool); R, S = Mem.shape; cands, feats = [], []
    for st in range(0, len(ids), chunk):
        ch = np.asarray(ids[st:st + chunk], int); t = MD.batch(x, ch)
        pair = t['pair'].cpu().numpy(); anchor = t['anchor'].cpu().numpy(); valid = t['valid'].cpu().numpy(); ctx = t['ax'].cpu().numpy()[:, :, x.na:]
        dist = np.where(valid, pair[:, :, 0] * 300, np.inf); dsch = anchor[:, :, 0] * 1000; q = len(ch)
        if R:
            Dm = np.where(Mem[None], dist[:, :, None], np.inf); srt = np.sort(Dm, axis=1); j0 = Dm.argmin(1)
            d1 = np.minimum(srt[:, 0], 15000.); d2 = np.minimum(srt[:, 1], 15000.) if R > 1 else np.full((q, S), 15000.); has = np.isfinite(srt[:, 0])
            n2, n5, n10 = ((Dm < 200).sum(1), (Dm < 500).sum(1), (Dm < 1000).sum(1))
        else:
            d1 = d2 = np.full((q, S), 15000.); has = np.zeros((q, S), bool); j0 = np.zeros((q, S), int); n2 = n5 = n10 = np.zeros((q, S))
        cand = np.argsort(np.minimum(d1, dsch), 1, kind='stable')[:, :K_CAND]
        r1 = np.argsort(np.argsort(d1, 1, kind='stable'), 1); r2 = np.argsort(np.argsort(dsch, 1, kind='stable'), 1)
        ii = np.arange(q)[:, None]; jj = j0[ii, cand]
        pr = np.where(has[ii, cand][..., None], pair[ii, jj] if R else 0, 0)
        F = np.concatenate([anchor[ii, cand], ctx[ii, cand], pr, np.stack([d1[ii, cand], d2[ii, cand], n2[ii, cand], n5[ii, cand], n10[ii, cand], r1[ii, cand], r2[ii, cand], has[ii, cand].astype(float)], -1)], -1)
        cands.append(cand); feats.append(F.astype(np.float32))
    return np.concatenate(cands), np.concatenate(feats)


class Table:
    """每个区全部小区的候选与特征（一次算好）。"""
    def __init__(self, D, R, X):
        self.D, self.R = D, R; self.c, self.f = {}, {}
        for d in D: self.c[d], self.f[d] = features(X[d], np.arange(D[d].n))

    def rows(self, d, lab):
        ids = np.array(sorted(lab), int); y = np.array([lab[q] for q in ids]); cand = self.c[d][ids]; hit = (cand == y[:, None]).any(1)
        _, inv, cnt = np.unique(self.D[d].groups[ids], return_inverse=True, return_counts=True); w = 1.0 / cnt[inv]
        k = np.flatnonzero(hit)
        return (self.f[d][ids[k]].reshape(-1, self.f[d].shape[-1]), (cand[k] == y[k, None]).astype(int).ravel(), np.repeat(w[k], K_CAND)), float(hit.mean()), int((~hit).sum())

    def train_set(self, target, exclude=()):
        Xs, Ys, Ws, rec = [], [], [], {}
        for d in self.D:
            if d in exclude: continue
            lab = dict(M.supervised(self.D[d], self.R[d]))
            if d != target: lab.update(M.uncovered(self.D[d], self.R[d]))
            (x_, y_, w_), r, drop = self.rows(d, lab); Xs.append(x_); Ys.append(y_); Ws.append(w_); rec[d] = dict(train_recall=r, dropped=drop)
        return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws), rec

    def predict(self, m, d):
        n = self.D[d].n; pr = m.predict_proba(self.f[d].reshape(-1, self.f[d].shape[-1]))[:, 1].reshape(n, K_CAND)
        P = np.zeros((n, self.D[d].school_n), np.float32); P[np.arange(n)[:, None], self.c[d]] = pr; return P


def fit(Xt, Yt, Wt, hp):
    return HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False, random_state=0, **hp).fit(Xt, Yt, sample_weight=Wt)


def run(k, seed, select):
    t0 = time.time(); c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; out = ROOT / f'{name}__gbt'; out.mkdir(parents=True, exist_ok=True)
    n_eval = sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in D); ref_n = json.loads((ROOT / name / 'eval.json').read_text(encoding='utf-8'))['summary']['两城']['n']
    assert n_eval == ref_n, f'评价集与主表不一致 {n_eval} vs {ref_n}'
    X = {d: evidence.build(D[d], R[d], c['evidence']) for d in D}; T = Table(D, R, X); print(name, '证据与特征', f'{time.time() - t0:.0f}s', flush=True)
    recall = {d: float(np.mean([bool(set(T.c[d][e['q']]) & set(e['truth'])) for e in evaluate.eval_rows(D[d], R[d].known)])) for d in D}
    (out / 'recall.json').write_text(json.dumps(recall, ensure_ascii=False, indent=1), encoding='utf-8'); print('评价题候选覆盖', {d: round(v, 4) for d, v in recall.items()}, flush=True)
    if select:
        chosen = {}
        for d in D:
            V = M.VALIDATION[d]; Xt, Yt, Wt, _ = T.train_set(d, exclude={V}); vlab = M.uncovered(D[V], R[V]); vq = np.array(sorted(vlab), int); vy = np.array([vlab[q] for q in vq])
            trials = []
            for hp in GRID:
                P = T.predict(fit(Xt, Yt, Wt, hp), V); trials.append(dict(hp=hp, val_acc=float((P[vq].argmax(1) == vy).mean())))
            best = max(trials, key=lambda r: r['val_acc']); chosen[d] = dict(hp=best['hp'], val_acc=best['val_acc'], validation=V, trials=trials)
            print('选', d, best['hp'], f"{best['val_acc']:.3f}", f'{time.time() - t0:.0f}s', flush=True)
        (out / 'chosen.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding='utf-8')
    chosen = json.loads((ROOT / 'k0_s1__gbt' / 'chosen.json').read_text(encoding='utf-8'))
    rows, rec_all = {}, {}
    for d in D:
        Xt, Yt, Wt, rec = T.train_set(d); rec_all[d] = rec; er = evaluate.eval_rows(D[d], R[d].known)
        for m_, hp in (('gbt', chosen[d]['hp']), ('gbt_orig', ORIG)):
            P = T.predict(fit(Xt, Yt, Wt, hp), d)
            for suf, PP in (('', P), ('_policy', constrain(P, R[d]))):
                od = out / (m_ + suf) / 'probs'; od.mkdir(parents=True, exist_ok=True); np.savez_compressed(od / f'{d}.npz', probs=PP[None].astype(np.float32))
                rows.setdefault(m_ + suf, []).extend(evaluate.correct(PP, D[d], er))
        print(name, d, f'{time.time() - t0:.0f}s', flush=True)
    for m_, rr in rows.items():
        s_ = evaluate.summary(rr); (out / m_ / 'eval.json').write_text(json.dumps(dict(summary=s_, k=k, seed=seed), ensure_ascii=False, indent=1), encoding='utf-8')
        print(m_, {kk: round(s_[kk]['accuracy'], 2) for kk in ['两城', '长沙', '杭州']}, flush=True)
    (out / 'train_recall.json').write_text(json.dumps(rec_all, ensure_ascii=False, indent=1), encoding='utf-8')


def report():
    res = {}; L = ['# 梯度提升树同条件验证：结果（pre_v1 开发数字）', '']
    for k, seed in BATCHES:
        c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; ev = c['evaluate']

        def rows_of(odir):
            rr = []
            for d in D:
                f = ROOT / odir / 'probs' / f'{d}.npz'
                if not f.exists(): return None
                rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
            return rr
        meth = {'完整模型': rows_of(name)}
        for m_ in ['gbt', 'gbt_policy', 'gbt_orig', 'gbt_orig_policy']: meth[m_] = rows_of(f'{name}__gbt/{m_}')
        for m_ in ['mn_pair', 'mn_pair_policy']: meth['同监督' + m_] = rows_of(f'{name}__matching/{m_}')
        for m_ in ['nearest_school', 'copy_nearest']: meth[m_] = rows_of(f'{name}__baselines/{m_}')
        meth = {a: b for a, b in meth.items() if b is not None}
        acc = lambda r: float(np.mean([x['ok'] for x in r]) * 100)
        best = lambda cands: max([v for v in cands if v in meth], key=lambda v: acc(meth[v]))
        g, pair = best(['gbt', 'gbt_policy']), best(['同监督mn_pair', '同监督mn_pair_policy'])
        full = meth['完整模型']
        cmp = {'T1 完整模型−梯度提升树': (full, g, '完整模型'), 'T2 梯度提升树−MN成对适配版': (meth[g], pair, g), '描述 完整模型−MN成对适配版': (full, pair, '完整模型'),
               '描述 梯度提升树−直线最近学校': (meth[g], 'nearest_school', g)}
        res[name] = dict(n=len(full), accuracy={m_: evaluate.summary(v) for m_, v in meth.items()},
                         compare={kk: dict(evaluate.compare(a, meth[b], ev), which=f'{an} vs {b}') for kk, (a, b, an) in cmp.items()})
        flags = []
        for d in D:
            has = {s for q, ss in R[d].known.items() if q not in D[d].not_reference for s in ss}
            flags += [any(s in has for s in e['truth']) for e in evaluate.eval_rows(D[d], R[d].known)]
        flags = np.array(flags)
        res[name]['coverage'] = {m_: dict(有参照=float(np.mean([r['ok'] for r, f in zip(v, flags) if f]) * 100), 无参照=(float(np.mean([r['ok'] for r, f in zip(v, flags) if not f]) * 100) if (~flags).any() else None), n有=int(flags.sum()), n无=int((~flags).sum())) for m_, v in meth.items()}
        res[name]['recall'] = json.loads((ROOT / f'{name}__gbt' / 'recall.json').read_text(encoding='utf-8'))
    (DOC / '结果.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    names = list(next(iter(res.values()))['accuracy'])
    L += ['## 准确率（两城｜长沙｜杭州）', '', '| 方法 | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
    for m_ in names:
        L.append(f'| {m_} | ' + ' | '.join((lambda a: f"{a['两城']['accuracy']:.2f}｜{a['长沙']['accuracy']:.2f}｜{a['杭州']['accuracy']:.2f}")(r['accuracy'][m_]) if m_ in r['accuracy'] else '—' for r in res.values()) + ' |')
    L += ['', '## 比较（两城差值 [95% 区间]（改对/改坏）；长沙；杭州）', '']
    for kk in next(iter(res.values()))['compare']:
        L.append(f'### {kk}'); L.append('')
        for name, r in res.items():
            x = r['compare'][kk]; L.append(f"- {name}（{x['which']}）：两城 {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]（{x['两城']['fixed']}/{x['两城']['broken']}）；长沙 {x['长沙']['delta']:+.2f} [{x['长沙']['ci'][0]:+.2f}, {x['长沙']['ci'][1]:+.2f}]；杭州 {x['杭州']['delta']:+.2f} [{x['杭州']['ci'][0]:+.2f}, {x['杭州']['ci'][1]:+.2f}]")
        L.append('')
    L += ['## 正确学校在支持集里有/没有参照（两城准确率）', '']
    for name, r in res.items():
        cv = r['coverage']; any_ = next(iter(cv.values())); L.append(f"### {name}（有 {any_['n有']}、无 {any_['n无']}）"); L.append('')
        for m_, v in cv.items(): L.append(f"- {m_}：有参照 {v['有参照']:.2f}" + (f"；无参照 {v['无参照']:.2f}" if v['无参照'] is not None else ''))
        L.append('')
    L += ['## 评价题候选覆盖（正确学校在前 10 候选里的比例）', ''] + [f"- {name}：" + '；'.join(f'{d} {v:.3f}' for d, v in r['recall'].items()) for name, r in res.items()]
    ch = json.loads((ROOT / 'k0_s1__gbt' / 'chosen.json').read_text(encoding='utf-8'))
    L += ['', '## 选中的超参数（K=0 验证区上选，K=1 沿用）', ''] + [f"- {d}（验证区 {x['validation']}）：{x['hp']}，验证准确率 {x['val_acc']:.3f}" for d, x in ch.items()]
    (DOC / '结果.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int); ap.add_argument('--select', action='store_true'); ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report: report()
    else: run(a.k, a.seed, a.select)


if __name__ == '__main__':
    main()
