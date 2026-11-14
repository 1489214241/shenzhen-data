"""Matching Networks 同监督验证（科研推进/MatchingNetworks同监督验证_20261111/方案与事先判定.md）。不改默认流程。
  python scripts/run_matching.py --k 0 --seed 1 --select          # K=0：每个目标区在验证区上选设置，再全来源重训
  python scripts/run_matching.py --k 1 --seed 2                   # K>0：沿用 k0_s1 选好的设置
  python scripts/run_matching.py --report                         # 汇总四批，写到 科研推进 目录
输出：outputs/main_pre_v1/k<K>_s<seed>__matching/<方法>[_policy]/probs/<区>.npz、eval.json；chosen.json（选设置的全部试验）。
方法：mn（MN 族：网格含 FCE 开关和 τ）、proto、mnca、mn_pair（成对证据适配版）、mn_t0（MN 选中设置、只用目标区规则已知训练）。
"""
import sys, json, time, argparse, copy
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import torch
from zoning import rules, evaluate, references, evidence, matching as M
from zoning.baselines import constrain
from run import load_districts

TAG = 'pre_v1'; ROOT = REPO / f'outputs/main_{TAG}'
SEEDS = [20261191, 20261192, 20261193]
METHODS = ['mn', 'proto', 'mnca', 'mn_pair']
DOC = REPO.parent / '科研推进' / 'MatchingNetworks同监督验证_20261111'


def setup(k, seed):
    base = json.loads((REPO / 'configs/default.json').read_text(encoding='utf-8'))
    c = copy.deepcopy(base); c['references'] = dict(k=k, seed=seed)
    D = load_districts(c)
    R0 = {d: rules.run(D[d], c['rules']['preset'], c['rules'].get('hangzhou_names', 'all'), c['rules'].get('hangzhou_boundary', 'v1'), c['rules'].get('hangzhou_community_rule', False)) for d in D}
    R, _ = references.apply(c['references'], D, R0)
    return c, D, R


def run(k, seed, select, methods):
    torch.use_deterministic_algorithms(True, warn_only=True); t0 = time.time()
    c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; out = ROOT / f'{name}__matching'; out.mkdir(parents=True, exist_ok=True)
    n_eval = sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in D); ref_n = json.loads((ROOT / name / 'eval.json').read_text(encoding='utf-8'))['summary']['两城']['n']
    assert n_eval == ref_n, f'评价集与主表不一致 {n_eval} vs {ref_n}'
    print(name, '评价', n_eval, flush=True)
    X = None
    if 'mn_pair' in methods:
        X = {d: evidence.build(D[d], R[d], c['evidence']) for d in D}; print('证据', f'{time.time() - t0:.0f}s', flush=True)
    chosen = {}
    for m in methods:
        f = out / f'chosen_{m}.json'
        if f.exists(): chosen[m] = json.loads(f.read_text(encoding='utf-8'))
        if not select:
            prev = json.loads((ROOT / 'k0_s1__matching' / f'chosen_{m}.json').read_text(encoding='utf-8'))
            chosen[m] = {d: dict(cfg=v['cfg'], steps=v['steps']) for d, v in prev.items()}
    for m in methods:
        if select and m not in chosen:
            chosen[m] = {}
            for d in D:
                V = M.VALIDATION[d]; tr = M.tasks_for_training(d, D, R, exclude={V}, X=X if m == 'mn_pair' else None)
                lab = M.uncovered(D[V], R[V]); qs = sorted(lab); val = M.Task(D[V], R[V], qs, [lab[q] for q in qs], X[V] if m == 'mn_pair' else None)
                trials = []
                for g in M.grid(m):
                    _, st, acc, nll = M.fit(m, g, tr, SEEDS[0], val=val); trials.append(dict(cfg=g, steps=st, val_acc=acc, val_nll=nll))
                best = max(trials, key=lambda r: (round(r['val_acc'], 9), -r['val_nll']))
                chosen[m][d] = dict(cfg=best['cfg'], steps=max(best['steps'], 1), validation=V, val_acc=best['val_acc'], trials=trials)
                print('选', m, d, best['cfg'], best['steps'], f"{best['val_acc']:.3f}", f'{time.time() - t0:.0f}s', flush=True)
            (out / f'chosen_{m}.json').write_text(json.dumps(chosen[m], ensure_ascii=False, indent=1), encoding='utf-8')
        variants = [(m, {})] + ([('mn_t0', dict(only_target=True))] if m == 'mn' else [])
        for vname, opt in variants:
            rows = {'': [], '_policy': []}
            for d in D:
                ch = chosen[m][d]
                tr = M.tasks_for_training(d, D, R, exclude=[o for o in D if o != d] if opt.get('only_target') else (), X=X if m == 'mn_pair' else None)
                task = M.Task(D[d], R[d], np.arange(D[d].n), None, X[d] if m == 'mn_pair' else None); P = []
                for s in SEEDS:
                    model, _, _, _ = M.fit(m, ch['cfg'], tr, s, steps=ch['steps']); P.append(M.predict(model, task))
                P = np.mean(P, 0); er = evaluate.eval_rows(D[d], R[d].known)
                for suf, PP in (('', P), ('_policy', constrain(P, R[d]))):
                    od = out / (vname + suf) / 'probs'; od.mkdir(parents=True, exist_ok=True); np.savez_compressed(od / f'{d}.npz', probs=PP[None].astype(np.float32))
                    rows[suf] += evaluate.correct(PP, D[d], er)
                del tr, task; torch.cuda.empty_cache()
            for suf, rr in rows.items():
                s_ = evaluate.summary(rr); (out / (vname + suf) / 'eval.json').write_text(json.dumps(dict(summary=s_, k=k, seed=seed), ensure_ascii=False, indent=1), encoding='utf-8')
                print(vname + suf, {kk: round(s_[kk]['accuracy'], 2) for kk in ['两城', '长沙', '杭州']}, f'{time.time() - t0:.0f}s', flush=True)
    if not select:
        for m in methods: (out / f'chosen_{m}.json').write_text(json.dumps(chosen[m], ensure_ascii=False, indent=1), encoding='utf-8')


BATCHES = [(0, 1), (1, 1), (1, 2), (1, 3)]


def report():
    res = {}; L = ['# Matching Networks 同监督验证：结果（pre_v1 开发数字）', '']
    cov_tab = {}
    for k, seed in BATCHES:
        c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; ev = c['evaluate']

        def rows_of(odir):
            rr = []
            for d in D:
                f = ROOT / odir / 'probs' / f'{d}.npz'
                if not f.exists(): return None
                rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
            return rr
        meth = {'完整模型': rows_of(name), '去掉学校当候选(D2)': rows_of(f'{name}__D2'), '抄最近': rows_of(f'{name}__baselines/copy_nearest'),
                'ProtoNet只本区+排除': rows_of(f'{name}__baselines/protonet_policy'), 'ModernNCA只本区+排除': rows_of(f'{name}__baselines/modernnca_policy')}
        for m in ['mn', 'mn_t0', 'proto', 'mnca', 'mn_pair']:
            for suf in ('', '_policy'): meth[m + suf] = rows_of(f'{name}__matching/{m}{suf}')
        meth = {a: b for a, b in meth.items() if b is not None}
        acc = lambda r: float(np.mean([x['ok'] for x in r]) * 100)
        best = lambda m: max([m, m + '_policy'], key=lambda v: acc(meth[v]) if v in meth else -1)
        mn_b, pair_b, pro_b, nca_b = best('mn'), best('mn_pair'), best('proto'), best('mnca')
        other_b = max([pro_b, nca_b], key=lambda v: acc(meth[v]))
        full = meth['完整模型']
        cmp = {'R1 完整模型−MN': (full, meth[mn_b], mn_b), 'R2 完整模型−MN成对': (full, meth[pair_b], pair_b), 'R3 MN−max(Proto,MNCA)': (meth[mn_b], meth[other_b], f'{mn_b} vs {other_b}'),
               '描述 MN同监督−MN只本区': (meth['mn'], meth['mn_t0'], 'mn vs mn_t0'), '描述 完整模型−D2': (full, meth['去掉学校当候选(D2)'], ''),
               '描述 MN成对−MN': (meth[pair_b], meth[mn_b], f'{pair_b} vs {mn_b}')}
        res[name] = dict(n=len(full), accuracy={m: evaluate.summary(v) for m, v in meth.items()},
                         compare={kk: dict(evaluate.compare(a, b, ev), which=w) for kk, (a, b, w) in cmp.items()})
        # 正确学校在支持集里有没有参照
        flags = []
        for d in D:
            has = {s for q, ss in R[d].known.items() if q not in D[d].not_reference for s in ss}
            flags += [any(s in has for s in e['truth']) for e in evaluate.eval_rows(D[d], R[d].known)]
        flags = np.array(flags); cov_tab[name] = {m: dict(有参照=float(np.mean([r['ok'] for r, f in zip(v, flags) if f]) * 100), 无参照=float(np.mean([r['ok'] for r, f in zip(v, flags) if not f]) * 100) if (~flags).any() else None, n有=int(flags.sum()), n无=int((~flags).sum())) for m, v in meth.items()}
        res[name]['coverage'] = cov_tab[name]
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
    for name, r in cov_tab.items():
        any_ = next(iter(r.values())); L.append(f"### {name}（有 {any_['n有']}、无 {any_['n无']}）"); L.append('')
        for m, v in r.items(): L.append(f"- {m}：有参照 {v['有参照']:.2f}；无参照 {v['无参照']:.2f}" if v['无参照'] is not None else f"- {m}：有参照 {v['有参照']:.2f}")
        L.append('')
    ch = {m: json.loads((ROOT / 'k0_s1__matching' / f'chosen_{m}.json').read_text(encoding='utf-8')) for m in METHODS}
    L += ['## 选中的设置（K=0 验证区上选，K=1 沿用）', '']
    for m, v in ch.items():
        for d, x in v.items():
            extra = ''
            if m == 'mn':
                tr = x['trials']; fo = max(t['val_acc'] for t in tr if t['cfg']['fce']); ff = max(t['val_acc'] for t in tr if not t['cfg']['fce'])
                tl = max(t['val_acc'] for t in tr if t['cfg']['tau'] == 'learn'); tf = max(t['val_acc'] for t in tr if t['cfg']['tau'] == 'fixed')
                extra = f"；验证最好：FCE开 {fo:.3f} / 关 {ff:.3f}，τ可学 {tl:.3f} / 固定1 {tf:.3f}"
            L.append(f"- {m} {d}（验证区 {x['validation']}）：{x['cfg']}，{x['steps']} 步，验证准确率 {x['val_acc']:.3f}{extra}")
    (DOC / '结果.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int); ap.add_argument('--select', action='store_true')
    ap.add_argument('--methods', nargs='*', default=METHODS); ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report: report()
    else: run(a.k, a.seed, a.select, a.methods)


if __name__ == '__main__':
    main()
