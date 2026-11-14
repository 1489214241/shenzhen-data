"""Full-evidence independent-family comparison. Frozen inputs, selection, fit, then score."""
import argparse
import copy
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
from zoning import DEVICE, model as MD, train, evaluate
from zoning.full_evidence import FullEvidenceModel, tokens
from run_modernnca_relation import cpu_task, tasks, device_batch, predict, digest, save
from zoning.matching import VALIDATION

SOURCE = REPO / 'outputs/modernnca_relation_20261112'
OUT = REPO / 'outputs/full_evidence_20261112'
DOC = REPO.parent / '科研推进/完整同信息参照比较_20261112'
METHODS = ['reference', 'matching', 'modernnca', 'proto', 'deepsets']
SEEDS = [20261231, 20261232, 20261233]
BATCHES = [(0, 1), (1, 1), (1, 2), (1, 3)]
GRID = [dict(lr=lr, reg=reg) for lr in [.001, .003] for reg in [.001, .01]]
CHECKPOINTS = [20, 40, 80, 160, 320]
EXTENSIONS = [640, 1280]
BS = 64


def loaded(k, s):
    f = SOURCE / f'k{k}_s{s}/inputs.pkl'
    assert digest(f) == json.loads(f.with_suffix('.json').read_text(encoding='utf8'))['input_sha256']
    with f.open('rb') as stream:
        c, D, R, X = pickle.load(stream)
    return c, D, R, X


def make(kind, x, start, seed):
    if kind == 'reference':
        return MD.RetrievalScorer(x.np_, x.na, start, seed, width=32).to(DEVICE)
    return FullEvidenceModel(kind, x.np_, x.na, seed, dim=32).to(DEVICE)


def freeze():
    paths = [Path(__file__), REPO / 'zoning/full_evidence.py', REPO / 'zoning/model.py',
             REPO / 'zoning/train.py', REPO / 'zoning/evaluate.py', REPO / 'zoning/matching.py',
             REPO / 'scripts/run_modernnca_relation.py',
             REPO / 'third_party/TALENT/model/models/modernNCA.py']
    plan = dict(methods=METHODS, seeds=SEEDS, grid=GRID, checkpoints=CHECKPOINTS, extensions=EXTENSIONS,
                batch_size=BS, batches=BATCHES, validation=VALIDATION,
                files={str(p.relative_to(REPO)): digest(p) for p in paths},
                input_hashes={f'k{k}_s{s}': json.loads((SOURCE / f'k{k}_s{s}/inputs.json').read_text(encoding='utf8'))['input_sha256'] for k, s in BATCHES},
                selection='K0 validation-only accuracy, then NLL, then earlier step. Same 4 trials for all. If best at 320, extend to 640; if best at 640, extend to 1280. K1 inherits selections.',
                training='Same questions, labels, building weights, masks, full candidate set and minibatch draws for all. Adam, clip 5. No SNS, top-k or target-score tuning. Reference retains its local linear warm start (100 steps); adapters do not inherit it.',
                input='All px/ax/gb/valid/membership. Packing is invertible before trainable encoders; school anchors available to every method. No extra raw coordinate channel.',
                caveat='All four external families are query-conditioned relation-input adapters with school anchors, NOT unmodified paper models. ModernNCA uses official encoder and Euclidean label-weighted readout, not RetrievalScorer.',
                decision='K1 positive paired spatial-block CI in every draw, no city confirmed harm. Bonferroni 98.75% intervals across four comparator families within each draw for confirmed superiority. K0 separate. Report raw model, strict set, and rule+model excluding verified supports.',
                status='pre_v1 development data; all targets previously studied; not independent confirmation')
    plan = json.loads(json.dumps(plan))
    f = OUT / 'frozen.json'
    if f.exists():
        assert json.loads(f.read_text(encoding='utf8')) == plan, 'Frozen experiment changed'
    else:
        save(f, plan)
        for p in paths:
            dest = OUT / 'code' / p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        save(DOC / '方案.json', plan)


def fit(kind, hp, seed, target, X, tr, start, steps=None, val=None, log=None):
    tic = time.time()
    m = make(kind, X[target], start, seed)
    opt = torch.optim.Adam(m.parameters(), lr=hp['lr'])
    rng = np.random.default_rng(seed)
    checkpoints = list(CHECKPOINTS)
    maximum = steps or checkpoints[-1]
    history, best = [], None
    step = 0
    while step < maximum:
        step += 1
        m.train(); opt.zero_grad()
        loss = m.regularization(hp['reg']); loss.backward()
        total = float(loss.detach())
        for _, t in tr:
            n = len(t['ids']); ids = rng.choice(n, min(BS, n), replace=False)
            b = device_batch(t, ids)
            p = m(b)
            loss = -(p[torch.arange(len(ids), device=DEVICE), b['y']].clamp_min(1e-30).log() * b['weight']).sum() * n / len(ids)
            assert torch.isfinite(loss)
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        if val is not None and step in checkpoints:
            P = predict(m, val); y = val['y'].numpy()
            row = dict(step=step, accuracy=float((P.argmax(1) == y).mean()),
                       nll=float(-np.log(P[np.arange(len(y)), y].clip(1e-30)).mean()), loss=total,
                       seconds=time.time()-tic)
            history.append(row)
            if best is None or (row['accuracy'], -row['nll'], -step) > (best['accuracy'], -best['nll'], -best['step']):
                best = row.copy()
            if step == maximum and best['step'] == step and step in [320, 640]:
                maximum *= 2; checkpoints += [maximum // 4 * 3, maximum]
            if log:
                save(log, dict(hp=hp, best=best, history=history, complete=step == maximum))
            print('VAL', target, kind, hp, row, flush=True)
    return m, dict(hp=hp, best=best, history=history, complete=True,
                   seconds=time.time()-tic, steps=step, final_loss=total)


def smoke():
    c, D, R, X = loaded(0, 1)
    d = next(iter(D)); tr = tasks(d, D, R, X)
    start = train.warm_start(X[d]); res = {}
    for kind in METHODS:
        m, info = fit(kind, GRID[0], SEEDS[0], d, X, tr, start, steps=10)
        task = cpu_task(d, np.arange(min(128, D[d].n)), D, X)
        P = predict(m, task)
        assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-5)
        res[kind] = dict(seconds=info['seconds'], loss=info['final_loss'], shape=list(P.shape), parameters=sum(p.numel() for p in m.parameters()))
        print('SMOKE', kind, res[kind], flush=True)
        del m
    save(OUT / 'smoke.json', res)


def select(targets=None, methods=None):
    c, D, R, X = loaded(0, 1)
    for d in targets or D:
        v = VALIDATION[d]
        tr = tasks(d, D, R, X, exclude={v})
        labels = train.uncovered_answers(D[v], X[v])
        val = cpu_task(v, sorted(labels), D, X, labels)
        start = train.warm_start(X[d])
        for kind in methods or METHODS:
            final = OUT / 'selection' / f'{d}_{kind}.json'
            if final.exists(): continue
            trials = []
            for i, hp in enumerate(GRID):
                f = OUT / 'selection/trials' / f'{d}_{kind}_{i}.json'
                if f.exists():
                    old = json.loads(f.read_text(encoding='utf8'))
                    if old.get('complete'):
                        trials.append(old); continue
                m, info = fit(kind, hp, SEEDS[0], d, X, tr, start, val=val, log=f)
                save(f, info); trials.append(info); del m
            chosen = max(trials, key=lambda z: (z['best']['accuracy'], -z['best']['nll'], -z['best']['step']))
            save(final, dict(hp=chosen['hp'], steps=chosen['best']['step'], validation=v,
                             best=chosen['best'], trials=trials))
            print('SELECTED', d, kind, chosen['hp'], chosen['best'], flush=True)
        del tr, val; gc.collect(); torch.cuda.empty_cache()


def run(k, s, targets=None, methods=None):
    c, D, R, X = loaded(k, s)
    folder = OUT / f'k{k}_s{s}'
    for d in targets or D:
        tr = tasks(d, D, R, X)
        start = train.warm_start(X[d]); task = cpu_task(d, np.arange(D[d].n), D, X)
        audit = {dd: dict(ids=t['ids'].tolist(), labels=t['y'].tolist(), groups=t['groups'].tolist()) for dd, t in tr}
        target_ids = set(audit[d]['ids'])
        assert not (target_ids & {row['q'] for row in evaluate.eval_rows(D[d], R[d].known)})
        save(folder / 'training_audit' / f'{d}.json', audit)
        for kind in methods or METHODS:
            choice = json.loads((OUT / 'selection' / f'{d}_{kind}.json').read_text(encoding='utf8'))
            probs = []
            for seed in SEEDS:
                f = folder / kind / 'seeds' / f'{d}_{seed}.npz'
                if f.exists():
                    probs.append(np.load(f)['P']); continue
                m, info = fit(kind, choice['hp'], seed, d, X, tr, start, steps=choice['steps'])
                P = predict(m, task)
                assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=2e-5)
                f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, P=P)
                torch.save(m.cpu().state_dict(), f.with_suffix('.pt'))
                save(f.with_suffix('.json'), info)
                probs.append(P); del m
                print('FIT', k, s, d, kind, seed, info['seconds'], flush=True)
            f = folder / kind / 'probs' / f'{d}.npz'
            f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, probs=np.stack(probs))
        del tr, task; gc.collect(); torch.cuda.empty_cache()
    save(folder / 'prediction_hashes.json', {str(f.relative_to(folder)): digest(f) for f in sorted(folder.glob('*/probs/*.npz'))})


def score():
    results = {}
    for k, s in BATCHES:
        c, D, R, X = loaded(k, s)
        folder = OUT / f'k{k}_s{s}'
        hashes = json.loads((folder / 'prediction_hashes.json').read_text(encoding='utf8'))
        assert len(hashes) == len(D) * len(METHODS)
        assert all(digest(folder / f) == h for f, h in hashes.items())
        scored, whole, Ps, seed_summary = {}, {}, {}, {}
        for kind in METHODS:
            scored[kind], whole[kind], Ps[kind] = [], [], {}
            seed_rows = [[] for _ in SEEDS]
            for d in D:
                P = np.load(folder / kind / 'probs' / f'{d}.npz')['probs']
                Ps[kind][d] = P.mean(0)
                er = evaluate.eval_rows(D[d], R[d].known)
                scored[kind] += evaluate.correct(P.mean(0), D[d], er)
                for j, pp in enumerate(P): seed_rows[j] += evaluate.correct(pp, D[d], er)
                rr = evaluate.overall(P.mean(0), D[d], R[d])
                whole[kind] += [row for q, row in zip(sorted(D[d].answers), rr) if R[d].kind.get(q) != 'verified']
            seed_summary[kind] = [evaluate.summary(rr) for rr in seed_rows]
        comparisons = {m: evaluate.compare(scored[m], scored['reference'], c['evaluate']) for m in METHODS[1:]}
        # Familywise conservative 98.75% bootstrap intervals (Bonferroni across 4)
        # supplement ordinary 95% intervals. Same spatial-block resampling.
        adjusted = {}
        for kind in METHODS[1:]:
            adjusted[kind] = {}
            for city in ['两城', '长沙', '杭州']:
                a = [r for r in scored[kind] if city == '两城' or r['city'] == city]
                b = [r for r in scored['reference'] if city == '两城' or r['city'] == city]
                delta = np.array([float(x['ok'])-float(y['ok']) for x, y in zip(a, b)])
                rng = np.random.default_rng(c['evaluate']['bootstrap_seed']); num = np.zeros(10000); den = np.zeros(10000)
                for d in sorted({r['d'] for r in a}):
                    ix = [i for i, r in enumerate(a) if r['d'] == d]
                    keys = sorted({a[i]['block'] for i in ix}); lk = {v:j for j,v in enumerate(keys)}
                    bi = np.array([lk[a[i]['block']] for i in ix]); n = np.bincount(bi); ds = np.bincount(bi, weights=delta[ix])
                    draw = rng.integers(0, len(keys), (10000, len(keys)))
                    num += ds[draw].sum(1); den += n[draw].sum(1)
                adjusted[kind][city] = (np.quantile(num / den, [.00625, .99375])*100).tolist()
        cases = [dict(district=d, **e, predictions={m:int(Ps[m][d][e['q']].argmax()) for m in METHODS})
                 for d in D for e in evaluate.eval_rows(D[d], R[d].known)]
        save(folder / 'scored_cases.json', cases)
        results[f'k{k}_s{s}'] = dict(model={m:evaluate.summary(r) for m,r in scored.items()},
            overall_excluding_verified={m:evaluate.summary(r) for m,r in whole.items()},
            per_seed=seed_summary, comparisons=comparisons, familywise_ci=adjusted)
        print('SCORED', k, s, {m: results[f'k{k}_s{s}']['model'][m]['两城']['accuracy'] for m in METHODS}, flush=True)
        del c, D, R, X; gc.collect()
    verdict = {m: dict(confirmed=all(results[f'k1_s{s}']['familywise_ci'][m]['两城'][0] > 0 for s in [1,2,3]) and
                      not any(results[f'k1_s{s}']['comparisons'][m][city]['ci'][1] < 0 for s in [1,2,3] for city in ['长沙','杭州'])) for m in METHODS[1:]}
    save(DOC / '结果.json', results); save(DOC / '判定.json', verdict)
    print('VERDICT', verdict, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke', 'freeze', 'select', 'fit', 'score', 'all'])
    parser.add_argument('--k', type=int, default=0); parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--district', nargs='+'); parser.add_argument('--method', nargs='+', choices=METHODS)
    args = parser.parse_args()
    if args.phase == 'smoke': smoke()
    else:
        freeze()
        if args.phase in ['select', 'all']: select(args.district, args.method)
        if args.phase == 'fit': run(args.k, args.seed, args.district, args.method)
        if args.phase == 'all':
            for k, s in BATCHES: run(k, s)
        if args.phase in ['score', 'all']: score()
