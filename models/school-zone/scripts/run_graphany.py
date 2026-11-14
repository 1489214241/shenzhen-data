"""GraphAny 同条件验证（科研推进/GraphAny同条件验证_20261111/方案与事先判定.md）。不改默认流程。
  python scripts/run_graphany.py select --cfg 0        # K=0 s1：设置 0..3 各一个进程，每个目标区在验证区上选步数，写 trials_<cfg>.json
  python scripts/run_graphany.py merge                 # 合并：每个目标区选 GraphAny 设置和最好单通道，写 chosen.json
  python scripts/run_graphany.py final --k 1 --seed 2  # 按 chosen 重训（同监督 3 种子、原生协议 1 种子）并预测，单通道直接预测
  python scripts/run_graphany.py report                # 汇总四批，写到 科研推进 目录
输出：outputs/main_pre_v1/k<K>_s<seed>__graphany/<ga|ga_src|chan>[_policy]/probs/<区>.npz、eval.json。
"""
import sys, json, time, argparse, copy
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import torch
torch.set_num_threads(2)
from zoning import evaluate, graphany as G, matching as M
from zoning.baselines import constrain
from run_matching import setup, BATCHES

ROOT = REPO / 'outputs/main_pre_v1'; SEL = ROOT / 'k0_s1__graphany'
CONFIGS = [dict(view=v, graph=g) for v in ('geometry', 'policy') for g in ('knn', 'knn_cut')]
SEEDS = [20261201, 20261202, 20261203]
DOC = REPO.parent / '科研推进' / 'GraphAny同条件验证_20261111'


def labeled(d, D, R, role):
    lab = dict(M.supervised(D[d], R[d]))
    if role == 'source': lab.update(M.uncovered(D[d], R[d]))
    return lab


class Cache:
    """同一设置下各区的 Graph（内积矩阵只算一次），按角色换训练题。"""
    def __init__(self, D, R): self.D, self.R, self.g = D, R, {}

    def get(self, d, cfg, lab=None):
        key = (d, cfg['view'], cfg['graph'])
        if key not in self.g: self.g[key] = G.Graph(self.D[d], self.R[d], cfg['view'], cfg['graph'])
        g = copy.copy(self.g[key]); g.labeled = lab
        if lab: g.lq = np.array(sorted(lab), int)
        return g


def train_graphs(cache, target, cfg, exclude=(), with_target=True):
    D, R = cache.D, cache.R; out = []
    for d in D:
        if d in exclude or (d == target and not with_target): continue
        lab = labeled(d, D, R, 'target' if d == target else 'source')
        if lab: out.append(cache.get(d, cfg, lab))
    return out


def select(ci):
    torch.use_deterministic_algorithms(True, warn_only=True); t0 = time.time(); cfg = CONFIGS[ci]
    c, D, R = setup(0, 1); cache = Cache(D, R); SEL.mkdir(parents=True, exist_ok=True); res = {}
    for d in D:
        V = M.VALIDATION[d]; vlab = M.uncovered(D[V], R[V]); vq = np.array(sorted(vlab), int)
        gs = train_graphs(cache, d, cfg, exclude={V}); vg = cache.get(V, cfg)
        _, (acc, nll, step) = G.train(gs, SEEDS[0], val=(vg, vq, [vlab[q] for q in vq]))
        lg = vg.predict_all(); chan = {ch: float((lg[ch][vq].argmax(1) == np.array([vlab[q] for q in vq])).mean()) for ch in G.CHANNELS}
        res[d] = dict(cfg=cfg, validation=V, val_acc=acc, val_nll=nll, steps=step, channel_val_acc=chan)
        print('选', ci, d, cfg, step, f'{acc:.3f}', {k: round(v, 3) for k, v in chan.items()}, f'{time.time() - t0:.0f}s', flush=True)
        (SEL / f'trials_{ci}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


def merge():
    T = [json.loads((SEL / f'trials_{i}.json').read_text(encoding='utf-8')) for i in range(len(CONFIGS))]; out = {}
    for d in T[0]:
        rows = [t[d] for t in T]
        best = max(rows, key=lambda r: (round(r['val_acc'], 9), -r['val_nll']))
        chans = [(r['channel_val_acc'][ch], -i, ch, r['cfg']) for i, r in enumerate(rows) for ch in G.CHANNELS]
        cb = max(chans, key=lambda x: (round(x[0], 9), x[1]))
        out[d] = dict(ga=dict(cfg=best['cfg'], steps=max(best['steps'], 1), val_acc=best['val_acc']), channel=dict(cfg=cb[3], channel=cb[2], val_acc=cb[0]),
                      all=[dict(cfg=r['cfg'], steps=r['steps'], val_acc=r['val_acc'], channel_val_acc=r['channel_val_acc']) for r in rows])
        print(d, out[d]['ga'], out[d]['channel'])
    (SEL / 'chosen.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


def final(k, seed):
    torch.use_deterministic_algorithms(True, warn_only=True); t0 = time.time()
    c, D, R = setup(k, seed); cache = Cache(D, R); name = f'k{k}_s{seed}'; out = ROOT / f'{name}__graphany'
    n_eval = sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in D); ref_n = json.loads((ROOT / name / 'eval.json').read_text(encoding='utf-8'))['summary']['两城']['n']
    assert n_eval == ref_n, f'评价集与主表不一致 {n_eval} vs {ref_n}'
    ch = json.loads((SEL / 'chosen.json').read_text(encoding='utf-8')); rows = {}; att_log = {}
    for d in D:
        er = evaluate.eval_rows(D[d], R[d].known); cg = ch[d]['ga']; Ps = {}
        tg = cache.get(d, cg['cfg']); lg = tg.predict_all()
        P = []
        for s in SEEDS:
            model, _ = G.train(train_graphs(cache, d, cg['cfg']), s, steps=cg['steps']); p, att = G.predict(model, tg, lg); P.append(p); att_log.setdefault(d, []).append(att.tolist())
        Ps['ga'] = np.mean(P, 0)
        model, _ = G.train(train_graphs(cache, d, cg['cfg'], with_target=False), SEEDS[0], steps=cg['steps']); Ps['ga_src'] = G.predict(model, tg, lg)[0]
        cc = ch[d]['channel']; Ps['chan'] = G.softmax_np(cache.get(d, cc['cfg']).predict_all([cc['channel']])[cc['channel']])
        for m, P_ in Ps.items():
            for suf, PP in (('', P_), ('_policy', constrain(P_, R[d]))):
                od = out / (m + suf) / 'probs'; od.mkdir(parents=True, exist_ok=True); np.savez_compressed(od / f'{d}.npz', probs=PP[None].astype(np.float32))
                rows.setdefault(m + suf, []).extend(evaluate.correct(PP, D[d], er))
        print(name, d, f'{time.time() - t0:.0f}s', flush=True)
    for m, rr in rows.items():
        s_ = evaluate.summary(rr); (out / m / 'eval.json').write_text(json.dumps(dict(summary=s_, k=k, seed=seed), ensure_ascii=False, indent=1), encoding='utf-8')
        print(m, {kk: round(s_[kk]['accuracy'], 2) for kk in ['两城', '长沙', '杭州']}, flush=True)
    (out / 'attention.json').write_text(json.dumps(att_log, ensure_ascii=False, indent=1), encoding='utf-8')


def report():
    res = {}; L = ['# GraphAny 同条件验证：结果（pre_v1 开发数字）', '']
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
        for m in ['ga', 'ga_src', 'chan']:
            for suf in ('', '_policy'): meth[f'{m}{suf}'] = rows_of(f'{name}__graphany/{m}{suf}')
        for m in ['modernnca_raw', 'modernnca_policy', 'copy_nearest', 'label_propagation', 'protonet_policy']: meth[m] = rows_of(f'{name}__baselines/{m}')
        for m in ['mnca', 'mnca_policy', 'mn_pair', 'mn_pair_policy']: meth['同监督' + m] = rows_of(f'{name}__matching/{m}')
        meth['去掉学校当候选(D2)'] = rows_of(f'{name}__D2')
        meth = {a: b for a, b in meth.items() if b is not None}
        acc = lambda r: float(np.mean([x['ok'] for x in r]) * 100)
        best = lambda cands: max([v for v in cands if v in meth], key=lambda v: acc(meth[v]))
        ga, chan, nca = best(['ga', 'ga_policy']), best(['chan', 'chan_policy']), best(['modernnca_raw', 'modernnca_policy', '同监督mnca', '同监督mnca_policy'])
        pair, src = best(['同监督mn_pair', '同监督mn_pair_policy']), best(['ga_src', 'ga_src_policy'])
        full = meth['完整模型']
        cmp = {'G1 完整模型−GraphAny': (full, ga, None), 'G2 GraphAny−ModernNCA': (meth[ga], nca, ga), 'G3 GraphAny−最好单通道': (meth[ga], chan, ga),
               '描述 GraphAny−抄最近': (meth[ga], 'copy_nearest', ga), '描述 GraphAny−标签传播': (meth[ga], 'label_propagation', ga),
               '描述 GraphAny−MN成对适配版': (meth[ga], pair, ga), '描述 GraphAny同监督−原生协议': (meth[ga], src, ga)}
        res[name] = dict(n=len(full), accuracy={m: evaluate.summary(v) for m, v in meth.items()}, compare={})
        for kk, (a, b, an) in cmp.items():
            res[name]['compare'][kk] = dict(evaluate.compare(a, meth[b], ev), which=f"{an or '完整模型'} vs {b}")
        flags = []
        for d in D:
            has = {s for q, ss in R[d].known.items() if q not in D[d].not_reference for s in ss}
            flags += [any(s in has for s in e['truth']) for e in evaluate.eval_rows(D[d], R[d].known)]
        flags = np.array(flags)
        res[name]['coverage'] = {m: dict(有参照=float(np.mean([r['ok'] for r, f in zip(v, flags) if f]) * 100), 无参照=(float(np.mean([r['ok'] for r, f in zip(v, flags) if not f]) * 100) if (~flags).any() else None), n有=int(flags.sum()), n无=int((~flags).sum())) for m, v in meth.items()}
    (DOC / '结果.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    names = list(next(iter(res.values()))['accuracy'])
    L += ['## 准确率（两城｜长沙｜杭州）', '', '| 方法 | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
    for m in names:
        L.append(f'| {m} | ' + ' | '.join((lambda a: f"{a['两城']['accuracy']:.2f}｜{a['长沙']['accuracy']:.2f}｜{a['杭州']['accuracy']:.2f}")(r['accuracy'][m]) if m in r['accuracy'] else '—' for r in res.values()) + ' |')
    L += ['', '## 比较（两城差值 [95% 区间]（改对/改坏）；长沙；杭州）', '']
    for kk in next(iter(res.values()))['compare']:
        L.append(f'### {kk}'); L.append('')
        for name, r in res.items():
            x = r['compare'][kk]; L.append(f"- {name}（{x['which']}）：两城 {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]（{x['两城']['fixed']}/{x['两城']['broken']}）；长沙 {x['长沙']['delta']:+.2f} [{x['长沙']['ci'][0]:+.2f}, {x['长沙']['ci'][1]:+.2f}]；杭州 {x['杭州']['delta']:+.2f} [{x['杭州']['ci'][0]:+.2f}, {x['杭州']['ci'][1]:+.2f}]")
        L.append('')
    L += ['## 正确学校在支持集里有/没有参照（两城准确率）', '']
    for name, r in res.items():
        cv = r['coverage']; any_ = next(iter(cv.values())); L.append(f"### {name}（有 {any_['n有']}、无 {any_['n无']}）"); L.append('')
        for m, v in cv.items(): L.append(f"- {m}：有参照 {v['有参照']:.2f}" + (f"；无参照 {v['无参照']:.2f}" if v['无参照'] is not None else ''))
        L.append('')
    ch = json.loads((SEL / 'chosen.json').read_text(encoding='utf-8')); L += ['## 选中的设置（K=0 验证区上选，K=1 沿用）', '']
    for d, x in ch.items():
        L.append(f"- {d}：GraphAny {x['ga']['cfg']}、{x['ga']['steps']} 步、验证 {x['ga']['val_acc']:.3f}；最好单通道 {x['channel']['cfg']} {x['channel']['channel']}、验证 {x['channel']['val_acc']:.3f}；"
                 + '；'.join(f"{a['cfg']['view']}/{a['cfg']['graph']} GA {a['val_acc']:.3f}" for a in x['all']))
    att = json.loads((ROOT / 'k1_s1__graphany' / 'attention.json').read_text(encoding='utf-8'))
    L += ['', '## 注意力权重（K=1 第 1 批，X / L1 / L2，3 种子平均）', ''] + [f"- {d}：" + ' / '.join(f'{v:.3f}' for v in np.mean(a, 0)) for d, a in att.items()]
    (DOC / '结果.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['select', 'merge', 'final', 'report']); ap.add_argument('--cfg', type=int); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    {'select': lambda: select(a.cfg), 'merge': merge, 'final': lambda: final(a.k, a.seed), 'report': report}[a.cmd]()


if __name__ == '__main__':
    main()
