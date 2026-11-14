"""Independent data-route, probability-replay and strict metric verification."""
import argparse
import collections
import json

import numpy as np
import torch

import experiment_modernnca_evidence_v2 as E
from zoning.scope_evidence_v2 import TEXTUAL_KINDS


def inputs():
    E.M.guard(); c = E.verify(); summary = {}
    manifest = E.read(E.INPUT / 'manifest.json')
    for p, h in manifest['inputs'].items():
        assert E.sha(p) == h
    for p, h in manifest['files'].items():
        assert E.sha(E.INPUT / p) == h
    for d in c['sources'] + c['targets']:
        old = E.M.pkl(E.INPUT / d / 'control.pkl'); new = E.M.pkl(E.INPUT / d / 'routed.pkl')
        states = np.load(E.INPUT / d / 'whole_scope.npz')['states']
        expected = np.stack([states == -1, states == 0, states == 1], -1).astype(np.float32)
        np.testing.assert_array_equal(new.anchor[..., -3:], expected)
        np.testing.assert_array_equal(new.pair[..., -3:], expected[:, new.ref_schools])
        assert not old.anchor[..., -3:].any() and not old.pair[..., -3:].any()
        np.testing.assert_array_equal(new.groups, old.groups)
        np.testing.assert_array_equal(new.self_mask, new.groups[:, None] == new.groups[new.ref_sites][None])
        y = dict(np.load(E.INPUT / d / 'routed_labels.npz'))
        original = dict(np.load(E.INPUT / d / 'control_labels.npz'))
        if d in c['sources']:
            for key in y:
                np.testing.assert_array_equal(y[key], original[key])
            summary[d] = dict(source_labels_and_splits_unchanged=True)
            continue
        ledger = E.read(E.INPUT / d / 'evidence_ledger.json')
        rec = np.full_like(y['labels'], -1)
        positive = collections.defaultdict(set); negative = collections.defaultdict(set)
        strong = weak = 0
        for row in ledger:
            kinds = {e['kind'] for e in row['evidence']}
            is_text = bool(kinds & TEXTUAL_KINDS) if row['positive'] else 'explicit_negative' in kinds
            assert row['routes']['training'] == is_text
            assert row['routes']['hard_constraint'] == is_text
            assert not row['routes']['complete_school_set']
            if is_text:
                strong += 1; rec[row['q'], row['school_index']] = int(row['positive'])
                (positive if row['positive'] else negative)[row['site_id']].add(row['school_id'])
            else:
                weak += 1
        np.testing.assert_array_equal(rec, y['labels'])
        np.testing.assert_array_equal(np.flatnonzero((rec >= 0).any(1)), y['train_ids'])
        assert not (rec >= 0).all(1).any()
        cons = E.read(E.INPUT / d / 'constraints.json')
        for site, v in cons.items():
            assert set(v['positive']) == positive[site] and set(v['negative']) == negative[site]
            assert not positive[site] & negative[site]
        branches = E.read(E.INPUT / d / 'base_branches.json')
        assert all(not b['university_member_expansion'] for b in branches)
        summary[d] = dict(ledger_rows=len(ledger), textual_rows=strong, estimate_only_rows=weak,
            no_estimate_as_hard_constraint_or_supervised_label=True,
            typed_positive_missing_from_reference=sum(r['positive'] and r['routes']['training'] and not r['reference_currently_present'] for r in ledger),
            whole_scope_cells_replayed=int(states.size), reference_mask_replayed=True)
    E.write(E.OUT / 'input_route_audit.json', dict(passed=True, districts=summary,
        target_gold_read=False, target_polygons_read=False))
    print('INPUT ROUTE AUDIT PASSED', json.dumps(summary, ensure_ascii=False), flush=True)


def outputs():
    c = E.verify(); E.check('weights_freeze.json'); E.check('prediction_freeze.json')
    arrays = 0; checked = 0
    for arm in E.ARMS:
        # Independently replay every target array from saved full-encoder weights.
        for seed in c['seeds']:
            model = E.teacher(c, seed)
            model.load_state_dict(torch.load(E.OUT / 'weights' / arm / f'{seed}.pt', map_location=E.M.DEVICE, weights_only=True))
            saved = dict(np.load(E.OUT / 'probabilities' / arm / f'{seed}.npz'))
            for d in c['targets']:
                x = E.M.pkl(E.INPUT / d / f'{arm}.pkl')
                now = E.M.predict(model, E.M.task(x, np.arange(x.n)), 'multilabel')
                np.testing.assert_array_equal(now, saved[d]); arrays += 1
        for d in c['targets']:
            meta = E.read(E.INPUT / d / 'metadata.json'); cons = E.read(E.INPUT / d / 'constraints.json')
            ids = [s['school_id'] for s in meta['schools']]
            pp = [dict(np.load(E.OUT / 'probabilities' / arm / f'{s}.npz'))[d] for s in c['seeds']]
            threshold = E.read(E.OUT / 'source_calibration.json')[arm]['threshold']
            for label in ['ensemble'] + list(map(str, c['seeds'])):
                p = np.stack(pp).mean(0) if label == 'ensemble' else pp[c['seeds'].index(int(label))]
                raw = E.read(E.OUT / 'predictions' / f'{arm}_raw' / label / f'{d}.json')
                verified = E.read(E.OUT / 'predictions' / f'{arm}_verified' / label / f'{d}.json')
                for q, (a, b) in enumerate(zip(raw, verified)):
                    members = set(np.flatnonzero(np.asarray(p[q], np.float64) >= threshold)) or {int(p[q].argmax())}
                    expected = {ids[j] for j in members}
                    assert set(a['after']) == expected
                    co = cons[a['site_id']]
                    assert set(b['after']) == (expected | set(co['positive'])) - set(co['negative'])
                    assert b['abstained'] == (len(b['after']) == 0)
                    checked += 2
    result = E.read(E.OUT / 'results.json')
    for arm in result['metrics']:
        if arm == 'native_modern_policy':
            continue
        rr = E.read(E.OUT / 'scored_rows' / f'{arm}.json')
        assert len(rr) == 1761 and sum(r['semantic_pending'] for r in rr) == 242
        correct = fp = fn = 0
        for r in rr:
            p, g = set(r['prediction']), set(r['truth'])
            assert r['correct'] == (p == g)
            correct += p == g; fp += len(p-g); fn += len(g-p)
        m = result['metrics'][arm]['全部']
        assert (correct, fp, fn) == (m['correct'], m['fp'], m['fn'])
    for d, h in result['gold_sha256'].items():
        assert E.sha(E.M.F.SZ.DATA / d / 'Gold/小区_学校Gold.json') == h
    for arm in E.ARMS:
        ys = {d: dict(np.load(E.INPUT / d / f'{arm}_labels.npz')) for d in c['sources']}
        pp = [dict(np.load(E.OUT / 'probabilities' / arm / f'{s}.npz')) for s in c['seeds']]
        avg = [np.stack([p[d] for p in pp]).mean(0) for d in c['sources']]
        cc = E.calibrate(avg, [ys[d]['valid_labels'] for d in c['sources']], E.native_single(c, ys, c['seeds']))
        assert cc == result['source_calibration'][arm]
        trials = [E.read(E.OUT / 'selection' / arm / f'trial{j}/trial.json') for j in range(len(c['grid']))]
        assert result['selection'][arm]['trial'] == max(range(len(trials)), key=lambda j: E.rank(trials[j]['best']))
    E.write(E.OUT / 'output_audit.json', dict(passed=True, target_probability_arrays_bitwise_replayed=arrays,
        raw_and_verified_output_rows_checked=checked, all_ensemble_metrics_recounted=True,
        source_selection_and_final_calibration_recounted=True, target_gold_unchanged=True))
    print('OUTPUT AUDIT PASSED', arrays, 'probability arrays;', checked, 'output rows', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['inputs', 'outputs'])
    globals()[ap.parse_args().phase]()
