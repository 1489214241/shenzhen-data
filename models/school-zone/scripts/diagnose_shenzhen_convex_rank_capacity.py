"""Quarantined source-label diagnostics after round37 scoring; never training."""
import numpy as np
import experiment_shenzhen_convex_threshold as C
from shenzhen_adaptive_threshold import choose
F, R, E = C.F, C.R, C.E


def count(z, margins, labels):
    y = np.asarray(labels, bool)
    p = choose(margins)
    separated = np.where(y, z, np.inf).min(1) > np.where(~y, z, -np.inf).max(1)
    correct = (p == y).all(1)
    fp = (p & ~y).any(1); fn = (~p & y).any(1)
    out = dict(n=len(y), rank_separable=int(separated.sum()), exact=int(correct.sum()),
        rank_blocked=int((~separated).sum()),
        separable_too_many=int((separated & fp & ~fn).sum()),
        separable_too_few=int((separated & fn & ~fp).sum()))
    assert not (separated & fp & fn).any()
    assert out['rank_separable'] == out['exact'] + out['separable_too_many'] + out['separable_too_few']
    return out


def run():
    F.guard(); c = C.verify(); R.check(C.OUT, 'prediction_freeze.json')
    assert F.read(R.ROOT/'round37_independent_completion_audit.json')['all_metrics_independently_recomputed']
    dest = C.OUT/'diagnostics_not_training/source_rank_capacity.json'
    if dest.exists(): raise FileExistsError('Preserve diagnostic')
    _, ys = E.load(c, 'features_labels')
    banks = [dict(np.load(C.OUT/'cache'/f'{s}.npz')) for s in c['seeds']]
    payloads = [dict(np.load(C.OUT/'margins/adaptive_head'/f'{s}.npz')) for s in c['seeds']]
    results = {}
    for idx, seed in enumerate(c['seeds'] + ['ensemble']):
        result = {}
        for view in ['train', 'valid']:
            entries = []; district = {}
            for d in c['sources']:
                k = d+'_'+view
                z = banks[idx][k+'_z'].astype(float) if seed != 'ensemble' else np.stack([b[k+'_z'].astype(float) for b in banks]).mean(0)
                m = payloads[idx][k] if seed != 'ensemble' else np.stack([p[k] for p in payloads]).mean(0)
                y = banks[0][k+'_y'] if view == 'train' else ys[d]['valid_labels']
                if view == 'train':
                    for b in banks:
                        np.testing.assert_array_equal(b[k+'_y'], y)
                        np.testing.assert_array_equal(b[k+'_ids'], banks[0][k+'_ids'])
                district[d] = {name: count(z[mask], m[mask], y[mask]) for name, mask in [
                    ('all', np.ones(len(y), bool)), ('single', y.sum(1)==1), ('multi', y.sum(1)>1)]}
                entries.append(district[d])
            aggregate = {name: {k:sum(e[name][k] for e in entries) for k in entries[0][name]} for name in ['all','single','multi']}
            result[view] = dict(aggregate=aggregate, districts=district)
        results[str(seed)] = result
    F.write(dest, dict(results=results, source_labels_used_for_postscore_diagnostic=True,
        no_threshold_fit=True, not_for_training=True, target_gold_read=False,
        limitations='Rank separability uses true source labels. It is an explanatory bound for a scalar threshold on frozen scores, not a deployable decision or universal model upper bound.',
        prediction_manifest_sha256=F.sha(C.OUT/'prediction_freeze.json'), audit_code_sha256=F.sha(__file__)))
    print('SOURCE_RANK_DIAGNOSTIC', results['ensemble'], flush=True)


if __name__ == '__main__': run()
