"""Source-gated round32 target trial, with the complete frozen round31 rules."""
import argparse
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch
import experiment_shenzhen_explicit_negative_training as E
from experiment_shenzhen_resident_replay import decode
from experiment_shenzhen_relative_decode import relative
from shenzhen_current_supplements import CurrentSupplements

F, R = E.F, E.R
OUT = R.ROOT/'round32_explicit_negative_target'
LATEST = R.ROOT/'round31_explicit_scope/combined'
SOURCE_AUDIT = R.ROOT/'round32_independent_completion_audit.json'
BANKS = E.ARMS + ['best_development', 'parser_repaired_development']
COMPARISONS = [('features_only', 'scalar_control'), ('features_labels', 'scalar_control'),
               ('features_labels', 'features_only'), ('features_labels', 'best_development'),
               ('features_labels', 'parser_repaired_development')]


def target_inputs(c, arm):
    xs, _ = E.load(c, arm)
    result = {d: xs[d] for d in c['targets']}
    for d, x in result.items():
        assert not len(x.supervised) and (x.y == -1).all()
        np.testing.assert_array_equal(x.self_mask, x.groups[:, None] == x.groups[x.ref_sites][None])
        assert x.n == len(F.read(R.OLD/'inputs'/f'{d}.json')['sites'])
    return result


def predict(model, xs, c):
    result = {d: R.probabilities(model, x, np.arange(x.n), c['chunk']) for d, x in xs.items()}
    assert all(np.isfinite(p).all() for p in result.values())
    return result


def checkpoint_model(xs, c, arm, seed):
    model = R.BernoulliRetrieval(41, 27, R.M.LinearRetrieval(next(iter(xs.values()))), seed, c['width']).to(R.DEVICE)
    saved = torch.load(E.OUT/'checkpoints'/f'{arm}_{seed}.pt', map_location=R.DEVICE, weights_only=False)
    assert saved['step'] == c['steps'] == 40
    model.load_state_dict(saved['model']); model.eval()
    return model


def decoded(p, d, t, supplement):
    return supplement.apply(decode(relative(p), d, t))


def prepare():
    F.guard(); c = E.verify(); R.check(E.OUT, 'probability_freeze.json')
    report = F.read(E.OUT/'source_report.json'); audit = F.read(SOURCE_AUDIT)
    assert audit['source_gate_passed'] and audit['all_metrics_recomputed']
    assert report['selected_target_candidate'] == 'features_labels_relative'
    assert F.read(E.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    replay = R.ROOT/'current_supplements_replay_20261113'
    R.check(replay, 'manifest.json')
    assert F.read(replay/'audit.json')['all_round31_predictions_reproduced']
    R.check(LATEST, 'prediction_freeze.json')
    if OUT.exists(): raise FileExistsError('Preserve frozen target trial')
    protocol = dict(round=32, arms=E.ARMS, seeds=c['seeds'], objectives=R.OBJECTIVES,
        primary_arm='features_labels', primary_objective='balanced_exact', decoder='relative',
        source_gate_passed=True, training_steps=0, comparisons=COMPARISONS,
        intervention='Matched three-arm source-trained checkpoints; scalar original input, features_only and features_labels identical repaired target features.',
        policy='Identical round31 supplements: original university areas, explicit shared rules, original-root graph, community, roads, residential, joint, named facts and whole-scope exclusions.',
        supplement_audit='All88 frozen predecessor files and79288 rows reproduced without Gold.',
        reference='best_development is round31 round07; parser_repaired_development is round31 round12. Both descriptive historical comparators, not matched training controls.',
        threshold='Frozen source-only selection for each model arm and objective; no target threshold or cardinality tuning.',
        output_nonempty='Abort if official exclusions empty any predicted set; do not select a new fallback after exclusion.',
        scored_sites=1761, unscored_sites=41, target_gold_read=False, target_polygon_read=False,
        independent_validation=False, adopted=False)
    F.write(OUT/'protocol.json', protocol)
    links = dict(F.read(E.OUT/'input_links.json')['files'])
    for folder, manifest in [(E.OUT, 'probability_freeze.json'), (LATEST, 'prediction_freeze.json'), (replay, 'manifest.json')]:
        links[str(folder/manifest)] = F.sha(folder/manifest)
        links.update({str(folder/p): h for p, h in F.read(folder/manifest)['files'].items()})
    for p in [E.OUT/'source_report.json', SOURCE_AUDIT, E.OUT/'probability_replay_audit.json']:
        links[str(p)] = F.sha(p)
    supplements = {d: CurrentSupplements(d) for d in c['targets']}
    for d, s in supplements.items():
        links.update(s.hashes)
        for p in [R.OLD/'inputs'/f'{d}.json', R.OUT/'policies'/f'{d}.json', R.OLD/'source_graphs'/f'{d}.json']:
            links[str(p)] = F.sha(p)
    code = [Path(__file__), F.REPO/'scripts/shenzhen_current_supplements.py',
            F.REPO/'scripts/experiment_shenzhen_resident_replay.py', F.REPO/'scripts/experiment_shenzhen_relative_decode.py',
            F.REPO/'scripts/shenzhen_partial_model.py', F.REPO/'scripts/shenzhen_reference_retraining.py',
            F.REPO/'scripts/shenzhen_scope_implication.py', F.REPO/'zoning/shenzhen_policy.py']
    for p in code:
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); links[str(p)] = F.sha(p)
    thresholds = {a: report['reports'][a+'_relative']['selections'] for a in c['arms']}
    F.write(OUT/'thresholds.json', thresholds); F.write(OUT/'input_links.json', dict(files=links))
    # Protocol and all dependency identities precede any target inference.
    F.write(OUT/'input_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p)
        for p in OUT.rglob('*') if p.is_file()}, target_gold_read=False))
    for arm in c['arms']:
        xs = target_inputs(c, arm); stored = []
        for seed in c['seeds']:
            model = checkpoint_model(xs, c, arm, seed)
            ps = predict(model, xs, c); stored.append(ps)
            dest = OUT/'probabilities'/arm/f'{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **ps); print('TARGET_PROBABILITIES', arm, seed, flush=True)
        mean = {d: np.stack([p[d] for p in stored]).mean(0) for d in c['targets']}
        for label, ps in [('ensemble', mean)] + [(str(s), p) for s, p in zip(c['seeds'], stored)]:
            for objective in R.OBJECTIVES:
                for d in c['targets']:
                    rows = decoded(ps[d], d, thresholds[arm][objective]['threshold'], supplements[d])
                    F.write(OUT/'predictions'/arm/objective/label/f'{d}.json', rows)
    verify_links()
    F.write(OUT/'prediction_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p)
        for p in OUT.rglob('*') if p.is_file()}, target_gold_read=False))
    print('TARGET_PREDICTIONS_FROZEN', flush=True)


def verify_links():
    for p, h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p) == h, p


def verify():
    verify_links(); R.check(OUT, 'input_freeze.json'); R.check(OUT, 'prediction_freeze.json')
    return F.read(OUT/'protocol.json')


def replay():
    F.guard(); cfg = verify(); c = E.verify(); thresholds = F.read(OUT/'thresholds.json')
    dest = OUT/'prediction_replay_audit.json'
    if dest.exists(): raise FileExistsError('Preserve target replay')
    supplements = {d: CurrentSupplements(d) for d in c['targets']}; count = 0
    for arm in cfg['arms']:
        xs = target_inputs(c, arm); stored = []
        for seed in c['seeds']:
            ps = dict(np.load(OUT/'probabilities'/arm/f'{seed}.npz')); stored.append(ps)
            actual = predict(checkpoint_model(xs, c, arm, seed), xs, c)
            for d in c['targets']: np.testing.assert_array_equal(ps[d], actual[d])
        mean = {d: np.stack([p[d] for p in stored]).mean(0) for d in c['targets']}
        for label, ps in [('ensemble', mean)] + [(str(s), p) for s, p in zip(c['seeds'], stored)]:
            for objective in cfg['objectives']:
                for d in c['targets']:
                    rows = decoded(ps[d], d, thresholds[arm][objective]['threshold'], supplements[d])
                    assert rows == F.read(OUT/'predictions'/arm/objective/label/f'{d}.json')
                    count += len(rows)
        print('TARGET_REPLAYED', arm, flush=True)
    assert count == 1802*3*2*11
    F.write(dest, dict(checkpoints=30, prediction_rows=count, probabilities_bitwise_equal=True,
        all_decisions_reproduced=True, target_gold_read=False, target_polygon_read=False))


def prediction_paths(obj, label, d):
    paths = {a: OUT/'predictions'/a/obj/label/f'{d}.json' for a in E.ARMS}
    paths['best_development'] = LATEST/'predictions/round07'/obj/label/f'{d}.json'
    paths['parser_repaired_development'] = LATEST/'predictions/round12'/obj/label/f'{d}.json'
    return paths


def evaluation_rows(c, obj, label):
    gold = {(r['district'], r['site_id']): r for r in F.read(R.BASE/'scored_cases.json')}
    rows = []; unscored = []
    for d in R.TARGETS:
        with (E.B.OUT/d/'district.pkl').open('rb') as h: district = pickle.load(h)
        xy = dict(zip(district.site_ids, district.coords))
        banks = {a: F.read(p) for a, p in prediction_paths(obj, label, d).items()}
        for rs in banks.values(): assert len(rs) == len({r['site_id'] for r in rs})
        banks = {a: {r['site_id']: r for r in rs} for a, rs in banks.items()}
        assert all(set(b) == set(banks['features_labels']) for b in banks.values())
        for site, r in banks['features_labels'].items():
            g = gold[d, site]
            row = dict(d=d, district=d, site_id=site, building_name=r['building_name'],
                **{a: b[site]['after'] for a, b in banks.items()})
            if not g['has_map_gold']: unscored.append(row); continue
            row.update(gold=set(g['gold']), block=str(tuple(int(z//1000) for z in xy[site])),
                       semantic_pending=bool(g['semantic_pending']))
            rows.append(row)
    assert len(rows) == 1761 and len(unscored) == 41
    return rows, unscored


def groups(rows):
    return [('all', rows), ('single', [r for r in rows if len(r['gold']) == 1]),
            ('multi', [r for r in rows if len(r['gold']) > 1])] + [
            (d, [r for r in rows if r['d'] == d]) for d in R.TARGETS] + [
            (k, [r for r in rows if r['semantic_pending'] == v])
            for k, v in [('semantic_pending', True), ('not_flagged', False)]]


def check_gold():
    hashes = F.read(R.BASE/'results.json')['gold_sha256']
    for d, h in hashes.items(): assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json') == h
    return hashes


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c = verify(); hashes = check_gold()
    assert F.read(OUT/'prediction_replay_audit.json')['all_decisions_reproduced']
    if (OUT/'results.json').exists(): raise FileExistsError('Preserve scored target trial')
    results = {}
    for obj in c['objectives']:
        results[obj] = {}
        for label in ['ensemble'] + list(map(str, c['seeds'])):
            rows, unscored = evaluation_rows(c, obj, label); report = {}
            for name, rs in groups(rows):
                effects = {}
                for after, before in COMPARISONS:
                    delta = [int(set(r[after]) == r['gold'])-int(set(r[before]) == r['gold']) for r in rs]
                    effect = dict(corrected=delta.count(1), harmed=delta.count(-1), delta_pp=float(np.mean(delta)*100))
                    if label == 'ensemble': effect['delta_95_pp'] = bootstrap(rs, delta, seed=20261104, draws=3000)
                    effects[after+'_vs_'+before] = effect
                report[name] = dict(metrics={a: metrics(rs, a) for a in BANKS}, comparisons=effects)
            results[obj][label] = report
            if label == 'ensemble':
                F.write(OUT/'diagnostics_not_training'/f'{obj}.json', [dict(r, gold=sorted(r['gold'])) for r in rows])
                F.write(OUT/'unscored'/f'{obj}.json', unscored)
                print('TARGET_RESULT', obj, report['all'], flush=True)
    F.write(OUT/'results.json', dict(reports=results, gold_sha256=hashes, scored=1761, unscored=41,
        source_gate_passed=True, predictions_frozen_before_scoring=True, independent_validation=False, adopted=False))


def audit():
    from zoning.evaluate import bootstrap
    c = verify(); hashes = check_gold(); result = F.read(OUT/'results.json')
    assert hashes == result['gold_sha256']
    dest = R.ROOT/'round32_target_completion_audit.json'
    if dest.exists(): raise FileExistsError('Preserve completion audit')
    checked = 0
    for obj in c['objectives']:
        for label in ['ensemble'] + list(map(str, c['seeds'])):
            rows, outside = evaluation_rows(c, obj, label)
            for name, rs in groups(rows):
                claim = result['reports'][obj][label][name]
                for a in BANKS:
                    actual = dict(n=len(rs), correct=sum(set(r[a]) == r['gold'] for r in rs),
                        fp=sum(len(set(r[a])-r['gold']) for r in rs), fn=sum(len(r['gold']-set(r[a])) for r in rs))
                    assert all(claim['metrics'][a][k] == v for k, v in actual.items())
                    assert claim['metrics'][a]['exact_accuracy'] == actual['correct']/actual['n']
                    checked += 1
                for after, before in COMPARISONS:
                    delta = [int(set(r[after]) == r['gold'])-int(set(r[before]) == r['gold']) for r in rs]
                    expected = dict(corrected=delta.count(1), harmed=delta.count(-1), delta_pp=float(np.mean(delta)*100))
                    if label == 'ensemble': expected['delta_95_pp'] = bootstrap(rs, delta, seed=20261104, draws=3000)
                    assert expected == claim['comparisons'][after+'_vs_'+before]; checked += 1
    assert checked == 2*11*7*10
    F.write(dest, dict(metric_and_comparison_groups=checked, gold_hashes_unchanged=True,
        all_seeds_districts_single_multi_recounted=True, fp_fn_corrected_harmed_and_intervals_recomputed=True,
        predictions_replayed_before_scoring=True, source_gate_passed=True, scored=1761, unscored=41,
        independent_validation=False, adopted=False, audit_code_sha256=F.sha(__file__)))
    print('TARGET_COMPLETION_AUDIT', checked, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'replay', 'score', 'audit'])
    globals()[p.parse_args().phase]()
