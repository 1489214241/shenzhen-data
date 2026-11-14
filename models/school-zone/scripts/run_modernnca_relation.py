"""Frozen, development-only ModernNCA relation experiment, 2026-11-12.

prepare -> select (K=0 validation districts only) -> fit -> score.
No default model/config/data mutations. Each phase is resumable.
"""
import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import torch
from zoning import DEVICE, evidence, model as MD, train, evaluate, matching as M
from zoning.modernnca_relation import ModernNCARelation
from zoning.baselines import constrain
from run_matching import setup, BATCHES

OUT = REPO / 'outputs/modernnca_relation_20261112'
DOC = REPO.parent / '科研推进/ModernNCA关系证据改进_20261112'
SEEDS = [20261221, 20261222, 20261223]
KINDS = ['reference', 'nca_relation', 'nca_relation_sns']
LRS = [.001, .003]
CHECKPOINTS = [20, 40, 80, 120, 160]
BS = 128


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def freeze():
    paths = [Path(__file__), REPO / 'zoning/modernnca_relation.py',
             REPO / 'zoning/model.py', REPO / 'zoning/train.py',
             REPO / 'zoning/evidence.py', REPO / 'zoning/evaluate.py',
             REPO / 'zoning/data.py', REPO / 'zoning/rules.py', REPO / 'zoning/references.py',
             REPO / 'zoning/matching.py', REPO / 'scripts/run_matching.py',
             REPO / 'configs/default.json',
             REPO / 'third_party/TALENT/model/models/modernNCA.py']
    files = {str(p.relative_to(REPO)): digest(p) for p in paths}
    plan = dict(files=files, seeds=SEEDS, kinds=KINDS, lrs=LRS,
                checkpoints=CHECKPOINTS, minibatch=BS, batches=BATCHES,
                primary='raw', selection='K0 fixed validation district accuracy; tie NLL then earlier checkpoint',
                decision='K1 three batches: positive difference with 95% spatial-block CI above zero in all three, no city with CI entirely below zero; K0 separately',
                note='Development only; hybrid ModernNCA plus reference model relation/anchor scorer, not a pure backbone substitution.')
    plan = json.loads(json.dumps(plan))
    p = OUT / 'frozen.json'
    if p.exists(): assert json.loads(p.read_text(encoding='utf-8')) == plan, 'Frozen source/config changed'
    else:
        save(p, plan)
        for src in paths:
            dest = OUT / 'code' / src.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(src.read_bytes())


def prepared(k, s):
    p = OUT / f'k{k}_s{s}/inputs.pkl'
    if p.exists():
        manifest = json.loads((p.parent / 'inputs.json').read_text(encoding='utf-8'))
        assert digest(p) == manifest['input_sha256'], 'Input cache changed'
        with open(p, 'rb') as f: return pickle.load(f)
    tic = time.time(); c, D, R = setup(k, s)
    X = {}
    for d in D:
        X[d] = evidence.build(D[d], R[d], c['evidence'])
        print('EVIDENCE', k, s, d, round(time.time()-tic, 1), flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix('.tmp'), 'wb') as f: pickle.dump((c, D, R, X), f, protocol=5)
    p.with_suffix('.tmp').replace(p)
    manifest = dict(input_sha256=digest(p), districts={})
    for d in D:
        manifest['districts'][d] = dict(n=D[d].n, schools=D[d].school_n,
            references=len(X[d].ref_sites), known=len(R[d].known), supervised=len(X[d].supervised),
            query_ids=[r['q'] for r in evaluate.eval_rows(D[d], R[d].known)])
    old = REPO / 'outputs/main_pre_v1' / f'k{k}_s{s}' / 'eval.json'
    expected = json.loads(old.read_text(encoding='utf-8'))['summary']['两城']['n']
    assert sum(len(z['query_ids']) for z in manifest['districts'].values()) == expected
    save(p.parent / 'inputs.json', manifest)
    print('PREPARED', k, s, round(time.time()-tic, 1), flush=True)
    return c, D, R, X


def cpu_task(d, ids, D, X, labels=None):
    t = MD.batch(X[d], ids, labels)
    t = {k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()}
    t['coords'] = torch.tensor((D[d].coords - D[d].coords.mean(0)) / 1000, dtype=torch.float32)
    t['refids'] = torch.tensor(X[d].ref_sites, dtype=torch.long)
    t['qids'] = torch.tensor(np.asarray(ids), dtype=torch.long)
    return t


ROW_KEYS = {'pair', 'anchor', 'valid', 'y', 'weight', 'px', 'ax', 'qids'}


def device_batch(t, ix):
    return {k: (v[ix] if k in ROW_KEYS else v).to(DEVICE)
            for k, v in t.items() if torch.is_tensor(v)}


def tasks(target, D, R, X, exclude=()):
    out = []
    for d in D:
        if d in exclude: continue
        lab = {int(q): int(X[d].y[q]) for q in X[d].supervised}
        if d != target: lab.update(train.uncovered_answers(D[d], X[d]))
        ids = sorted(lab)
        if ids: out.append((d, cpu_task(d, ids, D, X, lab)))
    total = sum(len(set(t['groups'])) for _, t in out)
    for _, t in out:
        # Exactly the original full-batch objective in expectation.
        t['weight'] *= len(out) * len(set(t['groups'])) / total
    # Keep immutable training tensors on GPU when there is ample space, avoiding
    # repeated PCIe copies. This changes neither sampling nor arithmetic.
    size = sum(v.numel() * v.element_size() for _, t in out for v in t.values() if torch.is_tensor(v))
    if DEVICE == 'cuda' and 2 * size + 2 * 1024**3 < torch.cuda.mem_get_info()[0]:
        out = [(d, {k: v.to(DEVICE) if torch.is_tensor(v) else v for k,v in t.items()}) for d,t in out]
        print('TASK CACHE GPU MB', round(size/2**20), flush=True)
    return out


def make(kind, x, start, seed):
    if kind == 'reference': return MD.RetrievalScorer(x.np_, x.na, start, seed, 32).to(DEVICE)
    return ModernNCARelation(x.np_, x.na, start, seed,
                            sample_rate=.5 if kind.endswith('sns') else 1.).to(DEVICE)


def predict(m, task):
    m.eval(); out = []
    with torch.no_grad():
        for st in range(0, len(task['ids']), BS):
            out.append(m(device_batch(task, slice(st, st+BS))).cpu().numpy())
    return np.concatenate(out)


def fit(kind, lr, seed, steps, target, X, tr, start, val=None, log=None):
    tic = time.time(); m = make(kind, X[target], start, seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed); history = []; best = None
    for step in range(1, steps+1):
        m.train(); opt.zero_grad(); reg = m.regularization(.01); reg.backward()
        total_loss = float(reg.detach())
        for _, t in tr:
            n = len(t['ids']); ids = rng.choice(n, min(BS, n), replace=False)
            b = device_batch(t, ids); p = m(b)
            loss = -(p[torch.arange(len(ids), device=DEVICE), b['y']].clamp_min(1e-30).log()
                     * b['weight']).sum() * n / len(ids)
            assert torch.isfinite(loss), 'nonfinite training loss'
            loss.backward(); total_loss += float(loss.detach())
            del b, p, loss
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        if step in CHECKPOINTS and val is not None:
            P = predict(m, val)
            y = val['y'].numpy(); acc = float((P.argmax(1) == y).mean())
            nll = float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-30)).mean())
            row = dict(step=step, acc=acc, nll=nll, loss=total_loss,
                       seconds=round(time.time()-tic, 2))
            history.append(row)
            if best is None or (acc, -nll, -step) > (best['acc'], -best['nll'], -best['step']): best = row
            print('VAL', target, kind, lr, row, flush=True)
            if log: save(log, dict(history=history, best=best))
    diag = dict(seconds=time.time()-tic, last_loss=total_loss)
    if hasattr(m, 'log_metric_scale'): diag['metric_scale'] = float(m.log_metric_scale.exp().detach())
    return m, dict(best=best, history=history, diagnostics=diag)


def select(targets):
    c, D, R, X = prepared(0, 1)
    for d in targets or list(D):
        p = OUT / 'selection' / f'{d}.json'
        if p.exists(): continue
        v = M.VALIDATION[d]; tr = tasks(d, D, R, X, exclude={v})
        vl = M.uncovered(D[v], R[v]); val = cpu_task(v, sorted(vl), D, X, vl)
        start = train.warm_start(X[d]); results = {}
        for kind in KINDS:
            trials = []
            for lr in LRS:
                log = OUT / 'selection/trials' / f'{d}_{kind}_{lr}.json'
                if log.exists():
                    r = json.loads(log.read_text(encoding='utf-8'))
                    if len(r['history']) == len(CHECKPOINTS):
                        trials.append(dict(lr=lr, **r)); continue
                m, r = fit(kind, lr, SEEDS[0], max(CHECKPOINTS), d, X, tr, start, val, log)
                save(log, r); trials.append(dict(lr=lr, **r)); del m
            best = max(trials, key=lambda z: (z['best']['acc'], -z['best']['nll'], -z['best']['step']))
            results[kind] = dict(lr=best['lr'], steps=best['best']['step'], validation=v,
                                 acc=best['best']['acc'], nll=best['best']['nll'], trials=trials)
        results['selected_nca'] = max(KINDS[1:], key=lambda k: (results[k]['acc'], -results[k]['nll']))
        save(p, results); print('SELECTED', d, results['selected_nca'], flush=True)
        del tr, val; torch.cuda.empty_cache()


def run(k, s, targets):
    c, D, R, X = prepared(k, s); root = OUT / f'k{k}_s{s}'
    assert all((OUT / 'selection' / f'{d}.json').exists() for d in (targets or list(D))), 'Select first'
    for d in targets or list(D):
        chosen = json.loads((OUT / 'selection' / f'{d}.json').read_text(encoding='utf-8'))
        tr = tasks(d, D, R, X); start = train.warm_start(X[d])
        task = cpu_task(d, np.arange(D[d].n), D, X)
        audit = {dd: dict(ids=t['ids'].tolist(), labels=t['y'].tolist(), groups=t['groups'].tolist()) for dd, t in tr}
        save(root / 'training_audit' / f'{d}.json', audit)
        # The new method's SNS variant is selected on validation only, not target scores.
        for name, kind in [('reference_matched', 'reference'), ('modernnca_selected', chosen['selected_nca'])]:
            hp = chosen[kind]; folder = root / name; probs = []
            for seed in SEEDS:
                f = folder / 'seeds' / f'{d}_{seed}.npz'
                if f.exists():
                    probs.append(np.load(f)['P']); continue
                m, info = fit(kind, hp['lr'], seed, hp['steps'], d, X, tr, start)
                P = predict(m, task); assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=2e-5)
                f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, P=P)
                torch.save(m.cpu().state_dict(), f.with_suffix('.pt'))
                save(f.with_suffix('.json'), dict(kind=kind, hp=hp, seed=seed, **info))
                probs.append(P); del m
                print('FIT', k, s, d, name, seed, info['diagnostics'], flush=True)
            f = folder / 'probs' / f'{d}.npz'; f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, probs=np.stack(probs))
        del tr, task; torch.cuda.empty_cache()
    # Predictions are sealed before the separate score phase reads query labels.
    files = sorted(root.glob('*/probs/*.npz'))
    save(root / 'prediction_hashes.json', {str(p.relative_to(root)): digest(p) for p in files})


def score():
    results = {}; lines = ['ModernNCA 关系证据适配：开发验证', '',
        '保留 ModernNCA 编码器/欧氏距离/标签汇总，加入现有参照模型的关系与学校打分。是混合适配版，不是原版 ModernNCA。',
        'K=0 验证区选择学习率、步数与 SNS 版本；K=1 沿用。对称重训的参照模型为主对手，旧 10 种子完整模型另列。', '']
    for k, s in BATCHES:
        root = OUT / f'k{k}_s{s}'
        if not (root / 'prediction_hashes.json').exists(): continue
        c, D, R = setup(k, s); name = f'k{k}_s{s}'
        frozen = json.loads((root / 'prediction_hashes.json').read_text(encoding='utf-8'))
        assert all(digest(root/p) == h for p, h in frozen.items())
        methods = {}; Ps = {}; seed_summaries = {}; source_hashes = {}
        for meth in ['reference_matched', 'modernnca_selected', 'historical_reference', 'historical_modernnca_rows']:
            folder = (root / meth if meth in ['reference_matched', 'modernnca_selected'] else
                      REPO / 'outputs/main_pre_v1' / (name if meth == 'historical_reference' else name+'__tabular/mnca_rows'))
            if not all((folder / 'probs' / f'{d}.npz').exists() for d in D): continue
            rr = []; P = {}; seed_rows = None
            for d in D:
                pf = folder / 'probs' / f'{d}.npz'; allp = np.load(pf)['probs']
                source_hashes[str(pf.relative_to(REPO))] = digest(pf)
                P[d] = allp.mean(0)
                er = evaluate.eval_rows(D[d], R[d].known)
                rr += evaluate.correct(P[d], D[d], er)
                if seed_rows is None: seed_rows = [[] for _ in allp]
                for j, pp in enumerate(allp): seed_rows[j] += evaluate.correct(pp, D[d], er)
            methods[meth] = rr; Ps[meth] = P
            seed_summaries[meth] = [evaluate.summary(r) for r in seed_rows]
        if len(methods) < 4: continue
        res = dict(accuracy={m: evaluate.summary(r) for m, r in methods.items()}, comparisons={},
                   per_seed=seed_summaries, probability_hashes=source_hashes)
        for other in ['reference_matched', 'historical_reference', 'historical_modernnca_rows']:
            res['comparisons'][other] = evaluate.compare(methods['modernnca_selected'], methods[other], c['evaluate'])
        # Reproducible per-question outcomes and corrected/harmed identities.
        cases = []
        for d in D:
            for e in evaluate.eval_rows(D[d], R[d].known):
                cases.append(dict(district=d, **e, predictions={m: int(p[d][e['q']].argmax()) for m,p in Ps.items()}))
        save(root / 'scored_cases.json', cases); results[name] = res
        lines.append(name)
        for m, z in res['accuracy'].items(): lines.append(f"  {m}: 两城 {z['两城']['accuracy']:.2f}, 长沙 {z['长沙']['accuracy']:.2f}, 杭州 {z['杭州']['accuracy']:.2f}")
        for m, z in res['comparisons'].items():
            a = z['两城']; lines.append(f"  新版减 {m}: {a['delta']:+.2f} [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}], 改对/改坏 {a['fixed']}/{a['broken']}")
        lines.append('')
    verdict = {}
    for comparator in ['reference_matched', 'historical_reference']:
        names = ['k1_s1', 'k1_s2', 'k1_s3']
        if all(n in results for n in names):
            win = all(results[n]['comparisons'][comparator]['两城']['ci'][0] > 0 for n in names)
            harm = any(results[n]['comparisons'][comparator][city]['ci'][1] < 0
                       for n in names for city in ['长沙', '杭州'])
            verdict[comparator] = dict(confirmed=win and not harm, every_batch_positive_ci=win,
                                      any_city_confirmed_harm=harm)
    save(DOC / '判定.json', verdict)
    lines.extend(['判定（K=1 三批）', json.dumps(verdict, ensure_ascii=False, indent=2)])
    save(DOC / '结果.json', results)
    (DOC / '结果.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def smoke():
    c, D, R, X = prepared(0, 1); d = '天心区'
    t = cpu_task(d, X[d].supervised[:12], D, X); b = device_batch(t, slice(None))
    start = train.warm_start(X[d]); base = make('reference', X[d], start, SEEDS[0])
    m = make('nca_relation', X[d], start, SEEDS[0]); base.eval(); m.eval()
    with torch.no_grad():
        # Turning off the metric must recover the reference scorer exactly.
        m.log_metric_scale.fill_(-100)
        delta = (m(b) - base(b)).abs().max().item()
        assert delta < 1e-6, delta
        # Official forward versus our distance/label aggregation, without relation/anchor additions.
        coord = b['coords']; z = m.embed(coord); qi = b['qids']; ri = b['refids']
        labels = torch.arange(len(ri), device=DEVICE) % 2
        expected = m.metric(coord[qi], None, coord[ri], labels, False).exp()
        actual = torch.softmax(-torch.cdist(z[qi], z[ri]), 1) @ torch.nn.functional.one_hot(labels, 2).float()
        core_delta = (expected - actual).abs().max().item()
        print('CORE CHECK', dict(recovery_delta=delta, official_core_delta=core_delta), flush=True)
        # Different GEMM batch shapes can perturb cdist at coincident coordinates.
        # Confirm the algorithm in float64 as well as bounding float32 error.
        m = m.double(); coord64 = coord.double(); z64 = m.embed(coord64)
        expected64 = m.metric(coord64[qi], None, coord64[ri], labels, False).exp()
        actual64 = torch.softmax(-torch.cdist(z64[qi], z64[ri]), 1) @ torch.nn.functional.one_hot(labels, 2).double()
        delta64 = (expected64 - actual64 - 1e-7).abs().max().item()
        assert delta64 < 1e-8, delta64
        assert core_delta < 5e-5, core_delta
        m = m.float()
        p = m(b); assert p.shape[1] == D[d].school_n and torch.isfinite(p).all()
    del base, m, b
    tr = tasks(d, D, R, X, exclude={M.VALIDATION[d]})
    m, info = fit('nca_relation', .001, SEEDS[0], 3, d, X, tr, start)
    save(OUT / 'smoke.json', dict(recovery_delta=delta, official_core_delta=core_delta,
                                official_core_float64_delta_without_epsilon=delta64, **info))
    print('SMOKE OK', delta, core_delta, info, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'smoke', 'select', 'fit', 'score'])
    p.add_argument('--k', type=int, default=0); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--targets', nargs='*'); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    if a.phase == 'smoke': smoke(); return
    freeze()
    if a.phase == 'prepare': prepared(a.k, a.seed)
    elif a.phase == 'select': select(a.targets)
    elif a.phase == 'fit': run(a.k, a.seed, a.targets)
    elif a.phase == 'score': score()


if __name__ == '__main__': main()
