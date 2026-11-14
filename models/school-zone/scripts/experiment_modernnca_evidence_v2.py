"""Sealed full-pipeline comparison, full-encoder fine-tuning, source-only selection."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

import experiment_modernnca_shenzhen_multilabel as M
from build_shenzhen_evidence_v2 import OUT as INPUT
from shenzhen_exact_calibration import exact_thresholds, threshold_metrics
from shenzhen_partial_model import predict_sets
from shenzhen_reviewed_supplements import ReviewedSupplements
from zoning.modernnca_evidence_v2 import expanded_model, observed_mean_loss
from zoning.scope_evidence_v2 import apply_verified_constraints

OUT = M.REPO / 'outputs/modernnca_evidence_v2_20261113'
DOC = M.REPO.parent / '科研推进/深圳全链路修复_v2_20261113'
TEACHER = M.OUT
ARMS = ['control', 'routed']
write, read, sha = M.write, M.read, M.F.sha


def verify():
    c = read(OUT / 'protocol.json')
    for p, h in read(OUT / 'inputs_sha256.json').items():
        assert sha(p) == h, p
    for p, h in read(OUT / 'code_sha256.json').items():
        assert sha(M.REPO / p) == h, p
    return c


def check(name):
    for p, h in read(OUT / name).items():
        assert sha(OUT / p) == h, p


def prepare():
    M.guard()
    if OUT.exists():
        raise FileExistsError('Preserve sealed experiment')
    original = M.verify(); receipt = read(INPUT / 'manifest.json')
    links = {str(INPUT / p): h for p, h in receipt['files'].items()}
    links.update(receipt['inputs'])
    links[str(INPUT / 'manifest.json')] = sha(INPUT / 'manifest.json')
    for s in original['seeds']:
        p = TEACHER / 'weights/categorical' / f'{s}.pt'; links[str(p)] = sha(p)
        p = TEACHER / 'probabilities/categorical' / f'{s}.npz'; links[str(p)] = sha(p)
    for d in original['sources'] + original['targets']:
        p = TEACHER / f'{d}_weights.npy'; links[str(p)] = sha(p)
    c = dict(date='2026-11-13', sources=original['sources'], targets=original['targets'], seeds=original['seeds'],
        selection_seed=original['seeds'][0], arms=ARMS, width=32, minibatch=64,
        grid=[dict(lr=lr, reg=.001) for lr in [.0003, .001]], checkpoints=[0, 20, 40, 80],
        warm_start='Frozen categorical Modern teacher for corresponding seed. Lossless zero-column expansion from41/27 to44/30. All encoder/query/matching parameters trainable.',
        loss='Mean BCE over observed schools per property-row; unknown entries zero gradient. Complete source singleton labels penalize every observed extra school. Same loss and source mass in both arms.',
        comparison='Matched compound input/routing intervention; component causal gains not isolated. Separately cross legacy versus verified-only decoders on each model.',
        input_manifest=str(INPUT / 'manifest.json'), source_partial_rows_included=69,
        target_labels='control original partial estimates; routed textual proofs and7explicit negatives, estimates remain features/reference tokens only.',
        output='Independent school membership with source-calibrated threshold; no fixed template inventory or target cardinality; fallback to top1 before explicit constraints. Empty after exclusion is retained as abstention.',
        selection='Source exact full-set subject to singleton no worse than frozen native teacher by0.5pp; tie multi then higher threshold. If no checkpoint passes, report gate failure; never adopt.',
        calibration='Enumerate exact threshold transitions on1246source validation rows. Checkpoint0 included as no-training control; final ensemble recalibrated on source only.',
        target='All1802outputs,1761fixed scored rows,41outside,242semantic-pending retained.',
        prior_target_results_seen=True, independent_validation=False, adopted=False,
        target_gold_read=False, target_polygons_read=False, diagnostic_target_scoring_if_gate_fails=True)
    write(OUT / 'protocol.json', c)
    for p, h in links.items():
        assert sha(p) == h, p
    write(OUT / 'inputs_sha256.json', links)
    files = [Path(__file__), M.REPO / 'zoning/modernnca_evidence_v2.py', M.REPO / 'zoning/scope_evidence_v2.py',
             M.REPO / 'tests/test_modernnca_evidence_v2.py', M.REPO / 'tests/test_scope_evidence_v2.py']
    write(OUT / 'code_sha256.json', {str(p.relative_to(M.REPO)): sha(p) for p in files})
    for p in files:
        dest = OUT / 'code' / p.relative_to(M.REPO); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    write(OUT / 'protocol_receipt.json', dict(sha256=sha(OUT / 'protocol.json'), target_gold_read=False))
    M.log('V2 PROTOCOL SEALED')


def load(c, arm):
    xs = {d: M.pkl(INPUT / d / f'{arm}.pkl') for d in c['sources'] + c['targets']}
    ys = {d: dict(np.load(INPUT / d / f'{arm}_labels.npz')) for d in xs}
    weights = {}
    # Original per-property target unit; preserve source single/multi balance.
    for d, x in xs.items():
        y = ys[d]; w = np.load(TEACHER / f'{d}_weights.npy').copy()
        if d in c['targets']:
            unit = float(np.max(w))
            _, inv, count = np.unique(x.groups[y['train_ids']], return_inverse=True, return_counts=True)
            w[:] = 0; w[y['train_ids']] = unit / count[inv]
        weights[d] = w
    total = sum(w[ys[d]['train_ids']].sum() for d, w in weights.items())
    tr = {d: M.task(xs[d], ys[d]['train_ids'], ys[d]['labels'], weights[d] / total) for d in xs if len(ys[d]['train_ids'])}
    val = {d: M.task(xs[d], ys[d]['valid_ids']) for d in c['sources']}
    size = sum(v.numel() * v.element_size() for t in tr.values() for v in t.values() if torch.is_tensor(v))
    if M.DEVICE == 'cuda' and size + 3 * 1024 ** 3 < torch.cuda.mem_get_info()[0]:
        tr = {d: {k: v.to(M.DEVICE) if torch.is_tensor(v) else v for k, v in t.items()} for d, t in tr.items()}
    M.log('TASKS', arm, 'observed rows', sum(t['n'] for t in tr.values()), 'cache_MB', round(size / 2**20))
    return xs, ys, tr, val


def teacher(c, seed):
    state = torch.load(TEACHER / 'weights/categorical' / f'{seed}.pt', map_location=M.DEVICE, weights_only=True)
    return expanded_model(state, seed, M.DEVICE)


def native_single(c, ys, seeds):
    count = 0
    for d in c['sources']:
        p = np.stack([dict(np.load(TEACHER / 'probabilities/categorical' / f'{s}.npz'))[d] for s in seeds]).mean(0)
        y = ys[d]['valid_labels']; single = y.sum(1) == 1
        count += int((p.argmax(1)[single] == y.argmax(1)[single]).sum())
    return count


def calibrate(probabilities, labels, baseline_single):
    cal = exact_thresholds(probabilities, labels)
    floor = baseline_single - .005 * cal['source_single']
    eligible = [r for r in cal['candidates'] if r['single_correct'] >= floor]
    pool = eligible or cal['candidates']
    best = max(pool, key=lambda r: (r['single_correct'] + r['multi_correct'], r['multi_correct'], r['threshold']))
    metrics = threshold_metrics(probabilities, labels, best['threshold'])
    return dict(**metrics, threshold=best['threshold'], native_single_correct=baseline_single,
                singleton_floor=floor, source_gate_passed=bool(eligible), candidate_thresholds=len(cal['candidates']))


def rank(r):
    return r['source_gate_passed'], r['correct'], r['multi_correct'], -r['step']


def optimize(c, arm, hp, seed, steps, tr, val, ys, folder, selecting):
    model = teacher(c, seed)
    initial = {k: p.detach().clone() for k, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    rng = np.random.default_rng(seed)
    baseline_single = native_single(c, ys, [seed])
    history = []; best = None
    for step in range(steps + 1):
        if step:
            model.train(); opt.zero_grad()
            reg = hp['reg'] * sum((p - initial[k]).square().mean() for k, p in model.named_parameters())
            reg.backward(); loss_sum = float(reg.detach())
            for d, t in tr.items():
                ix = rng.choice(t['n'], min(c['minibatch'], t['n']), replace=False)
                b = M.device_batch(t, ix)
                loss = observed_mean_loss(model(b), b['labels'], b['weight']) * t['n'] / len(ix)
                assert torch.isfinite(loss), (arm, seed, step, d)
                loss.backward(); loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            assert torch.isfinite(norm)
            opt.step()
            if step % 10 == 0:
                M.log('V2 TRAIN', arm, seed, hp, step, round(loss_sum, 6))
        if selecting and step in c['checkpoints']:
            pp = [M.predict(model, val[d], 'multilabel') for d in c['sources']]
            r = calibrate(pp, [ys[d]['valid_labels'] for d in c['sources']], baseline_single)
            r['step'] = step; history.append(r)
            if best is None or rank(r) > rank(best):
                best = r; folder.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), folder / 'best.pt')
            write(folder / 'progress.json', dict(best=best, history=history, complete=False))
            M.log('V2 SOURCE', arm, hp, r)
    folder.mkdir(parents=True, exist_ok=True)
    if selecting:
        result = dict(hp=hp, best=best, history=history, complete=True, budget_edge=best['step'] == steps)
        write(folder / 'trial.json', result); return result
    torch.save(model.state_dict(), folder / f'{seed}.pt')
    write(folder / f'{seed}.json', dict(seed=seed, hp=hp, steps=steps, trainable_parameters='all'))


def train():
    M.guard(); c = verify()
    for arm in ARMS:
        xs, ys, tr, val = load(c, arm)
        trials = []
        for j, hp in enumerate(c['grid']):
            folder = OUT / 'selection' / arm / f'trial{j}'
            r = read(folder / 'trial.json') if (folder / 'trial.json').exists() else optimize(c, arm, hp, c['selection_seed'], max(c['checkpoints']), tr, val, ys, folder, True)
            trials.append(r)
        j = max(range(len(trials)), key=lambda j: rank(trials[j]['best']))
        selected = dict(trial=j, **trials[j]); write(OUT / 'selection' / f'{arm}.json', selected)
        for seed in c['seeds']:
            folder = OUT / 'weights' / arm
            if not (folder / f'{seed}.pt').exists():
                optimize(c, arm, selected['hp'], seed, selected['best']['step'], tr, val, ys, folder, False)
                M.log('V2 FIT COMPLETE', arm, seed)
        del xs, ys, tr, val
        if M.DEVICE == 'cuda':
            torch.cuda.empty_cache()
    for name, folder in [('selection_freeze.json', 'selection'), ('weights_freeze.json', 'weights')]:
        write(OUT / name, {str(p.relative_to(OUT)): sha(p) for p in (OUT / folder).rglob('*') if p.is_file()})
    M.log('V2 ALL TRAINING COMPLETE')


def infer():
    M.guard(); c = verify(); check('selection_freeze.json'); check('weights_freeze.json')
    calibration = {}; replay = {}
    for arm in ARMS:
        xs = {d: M.pkl(INPUT / d / f'{arm}.pkl') for d in c['sources'] + c['targets']}
        ys = {d: dict(np.load(INPUT / d / f'{arm}_labels.npz')) for d in xs}
        payload = []
        for seed in c['seeds']:
            model = teacher(c, seed)
            model.load_state_dict(torch.load(OUT / 'weights' / arm / f'{seed}.pt', map_location=M.DEVICE, weights_only=True))
            pp = {}
            for d, x in xs.items():
                ids = np.arange(x.n) if d in c['targets'] else ys[d]['valid_ids']
                t = M.task(x, ids); pp[d] = M.predict(model, t, 'multilabel')
                if d in c['targets']:
                    np.testing.assert_array_equal(pp[d], M.predict(model, t, 'multilabel'))
            p = OUT / 'probabilities' / arm / f'{seed}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **pp); payload.append(pp)
            replay[f'{arm}/{seed}'] = True; M.log('V2 PROBABILITIES', arm, seed)
        mean = {d: np.stack([p[d] for p in payload]).mean(0) for d in xs}
        cal = calibrate([mean[d] for d in c['sources']], [ys[d]['valid_labels'] for d in c['sources']], native_single(c, ys, c['seeds']))
        calibration[arm] = cal
        for label, pp in [('ensemble', mean)] + list(zip(map(str, c['seeds']), payload)):
            for d in c['targets']:
                meta = read(INPUT / d / 'metadata.json'); cons = read(INPUT / d / 'constraints.json')
                policy = read(M.ROOT / 'round08_reference_retraining/policies' / f'{d}.json')
                graph = read(M.OLD / 'source_graphs' / f'{d}.json')
                ids = [s['school_id'] for s in meta['schools']]
                outputs = {k: [] for k in ['raw', 'legacy', 'verified']}
                for q, site in enumerate(meta['sites']):
                    p = np.asarray(pp[d][q], np.float64)
                    chosen = np.flatnonzero(p >= cal['threshold']).tolist() or [int(p.argmax())]
                    base = dict(district=d, site_id=site['coordinate_group_id'], building_name=site['building_name'],
                                complete_school_set=False, enrollment_eligibility=False)
                    raw = sorted(ids[j] for j in chosen)
                    outputs['raw'].append(dict(base, after=raw))
                    k = cons[base['site_id']]
                    after = apply_verified_constraints(raw, k['positive'], k['negative'])
                    outputs['verified'].append(dict(base, after=after, abstained=not after))
                    known = [ids[j] for j in meta['known'].get(str(q), [])]
                    decision = predict_sets(p, cal['threshold'], ids, policy, graph, site, known)
                    outputs['legacy'].append(dict(base, after=decision['predicted_school_ids'], decision=decision))
                outputs['legacy'] = ReviewedSupplements(d).apply(outputs['legacy'])
                for decoder, rr in outputs.items():
                    write(OUT / 'predictions' / f'{arm}_{decoder}' / label / f'{d}.json', rr)
        M.log('V2 ENSEMBLE SOURCE', arm, cal)
    write(OUT / 'source_calibration.json', calibration); write(OUT / 'inference_replay.json', replay)
    files = [p for folder in ['predictions', 'probabilities'] for p in (OUT / folder).rglob('*') if p.is_file()]
    files += [OUT / 'source_calibration.json', OUT / 'inference_replay.json']
    write(OUT / 'prediction_freeze.json', {str(p.relative_to(OUT)): sha(p) for p in files})
    M.log('V2 PREDICTIONS SEALED')


def score():
    c = verify(); check('prediction_freeze.json')
    truth = read(M.REPO / 'outputs/modernnca_partial_sets_20261113/scored_rows/categorical_policy.json')
    byid = {(r['d'], r['site_id']): r for r in truth}
    groups = {'全部': lambda r: True, **{d: (lambda r, d=d: r['d'] == d) for d in c['targets']},
              '单校': lambda r: r['size'] == 1, '多校': lambda r: r['size'] > 1,
              '两校': lambda r: r['size'] == 2, '三校及以上': lambda r: r['size'] >= 3,
              '语义待核': lambda r: r['semantic_pending']}
    variants = [f'{a}_{b}' for a in ARMS for b in ['raw', 'legacy', 'verified']]
    scores = {}; rows = {}; seeds = {}
    for arm in variants:
        seeds[arm] = {}
        for label in ['ensemble'] + list(map(str, c['seeds'])):
            rr = []; outside = 0
            for d in c['targets']:
                pred = read(OUT / 'predictions' / arm / label / f'{d}.json')
                expected = {s['coordinate_group_id'] for s in read(INPUT / d / 'metadata.json')['sites']}
                assert len(pred) == len(expected) and {r['site_id'] for r in pred} == expected
                for r in pred:
                    key = (d, r['site_id'])
                    if key not in byid:
                        outside += 1; continue
                    old = byid[key]; g = set(old['truth']); p = set(r['after'])
                    rr.append(dict(old, prediction=sorted(p), correct=p == g, tp=len(p & g), fp=len(p-g), fn=len(g-p)))
            assert len(rr) == 1761 and outside == 41
            seeds[arm][label] = {g: M.metric([r for r in rr if f(r)]) for g, f in groups.items()}
            if label == 'ensemble':
                rows[arm] = rr; scores[arm] = seeds[arm][label]; write(OUT / 'scored_rows' / f'{arm}.json', rr)
    rows['native_modern_policy'] = truth; scores['native_modern_policy'] = {g: M.metric([r for r in truth if f(r)]) for g, f in groups.items()}
    pairs = [('routed_verified', 'control_legacy'), ('routed_verified', 'control_verified'),
             ('routed_legacy', 'control_legacy'), ('control_verified', 'control_legacy'),
             ('routed_verified', 'routed_legacy'), ('routed_verified', 'native_modern_policy')]
    comparisons = {}
    for a, b in pairs:
        before = {(r['d'], r['site_id']): r for r in rows[b]}
        comparisons[f'{a} minus {b}'] = {}
        for g, f in groups.items():
            rr = [r for r in rows[a] if f(r)]
            delta = [int(r['correct']) - int(before[r['d'], r['site_id']]['correct']) for r in rr]
            comparisons[f'{a} minus {b}'][g] = dict(delta_pp=100*float(np.mean(delta)), corrected=delta.count(1), harmed=delta.count(-1), ci95=M.evaluate.bootstrap(rr, delta, 20261104, 3000))
    gold = read(M.REPO / 'outputs/modernnca_partial_sets_20261113/results.json')['gold_sha256']
    for d, h in gold.items():
        assert sha(M.F.SZ.DATA / d / 'Gold/小区_学校Gold.json') == h
    result = dict(metrics=scores, seeds=seeds, comparisons=comparisons, gold_sha256=gold,
        source_calibration=read(OUT / 'source_calibration.json'), selection={a: read(OUT / 'selection' / f'{a}.json') for a in ARMS},
        adopted=False, independent_validation=False, primary='routed_verified minus control_legacy')
    write(OUT / 'results.json', result); write(DOC / '结果.json', result)
    M.log('V2 RESULTS', {a: m['全部']['correct'] for a, m in scores.items()}, comparisons['routed_verified minus control_legacy']['全部'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['prepare', 'train', 'infer', 'score'])
    globals()[ap.parse_args().phase]()
