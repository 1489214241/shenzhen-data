"""Replay the frozen source-only scope ablation and audit paired errors."""
import os
os.environ['OMP_NUM_THREADS'] = '6'
import pickle
import numpy as np
import experiment_shenzhen_composed_probe as E
from zoning.evaluate import bootstrap


def select(p, t):
    result = p >= t
    empty = ~result.any(1)
    result[empty, p[empty].argmax(1)] = True
    return result


def main():
    F, R, P = E.F, E.R, E.P
    F.guard()
    c = E.verify()
    R.check(E.OUT, 'probability_freeze.json')
    reports = F.read(E.OUT/'source_report.json')['reports']
    assert F.sha(E.OUT/'models/pooled_control.pkl') == F.sha(P.OUT/'models/pooled_relations.pkl')
    assert F.sha(E.OUT/'probabilities/pooled_control.npz') == F.sha(P.OUT/'probabilities/pooled_relations.npz')
    assert F.sha(E.OUT/'train_labels_weights.npz') == F.sha(P.OUT/'train_labels_weights.npz')
    train = np.load(E.OUT/'train_features.npy', mmap_mode='r')
    old = np.load(P.OUT/'train_features.npy', mmap_mode='r')
    for start in range(0, len(train), 10000):
        np.testing.assert_array_equal(train[start:start+10000, :159], old[start:start+10000])
    np.testing.assert_array_equal(train[:, 159:].sum(1), np.ones(len(train)))
    models, probs = {}, {}
    for arm in c['arms']:
        with (E.OUT/'models'/f'{arm}.pkl').open('rb') as handle:
            models[arm] = pickle.load(handle)
        assert models[arm].n_iter_ == 200
        probs[arm] = dict(np.load(E.OUT/'probabilities'/f'{arm}.npz'))
    rows = []
    for d in c['sources']:
        for split in ['valid', 'train_multi']:
            v = dict(np.load(E.OUT/split/f'{d}.npz'))
            prev = dict(np.load(P.OUT/split/f'{d}.npz'))
            np.testing.assert_array_equal(v['features'][:, :, :159], prev['features'])
            np.testing.assert_array_equal(v['labels'], prev['labels'])
            np.testing.assert_array_equal(v['ids'], prev['ids'])
            for arm in c['arms']:
                width = c['features'][arm]
                a = v['features'][:, :, :width]
                p = models[arm].predict_proba(a.reshape(-1, width))[:, 1].reshape(a.shape[:2]) if len(a) else np.zeros(a.shape[:2])
                np.testing.assert_array_equal(p, probs[arm][d+'_'+split])
            if split != 'valid':
                continue
            predicted = {arm: select(probs[arm][d+'_valid'], reports[arm]['selections']['balanced_exact']['threshold']) for arm in c['arms']}
            coords = np.load(F.REPO/'data'/d/'arrays.npz')['coords']
            for j, q in enumerate(v['ids']):
                truth = v['labels'][j].astype(bool)
                rows.append(dict(d=d, q=int(q), block=str(tuple(int(z//1000) for z in coords[q])),
                    multi=bool(truth.sum()>1), gold=np.flatnonzero(truth).tolist(),
                    before=np.flatnonzero(predicted['pooled_control'][j]).tolist(),
                    after=np.flatnonzero(predicted['plus_composed_scope'][j]).tolist(),
                    before_ok=bool((predicted['pooled_control'][j]==truth).all()),
                    after_ok=bool((predicted['plus_composed_scope'][j]==truth).all())))
    assert len(rows)==1246 and sum(r['multi'] for r in rows)==60
    groups = [('all',rows), ('single',[r for r in rows if not r['multi']]), ('multi',[r for r in rows if r['multi']])]
    groups += [(d,[r for r in rows if r['d']==d]) for d in c['sources']]
    metrics = {}
    for name, rr in groups:
        delta = [int(r['after_ok'])-int(r['before_ok']) for r in rr]
        metrics[name] = dict(n=len(rr), before_correct=sum(r['before_ok'] for r in rr),
            after_correct=sum(r['after_ok'] for r in rr), corrected=delta.count(1), harmed=delta.count(-1),
            delta_pp=float(np.mean(delta)*100), delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    for arm, key in [('pooled_control','before_ok'),('plus_composed_scope','after_ok')]:
        selection=reports[arm]['selections']['balanced_exact']
        assert selection['correct']==sum(r[key] for r in rows)
        assert selection['multi_correct']==sum(r[key] for r in rows if r['multi'])
        gate=c['source_gate']
        passed=(selection['multi_correct']>=gate['min_multi_correct'] and selection['single_correct']>=gate['min_single_correct'] and reports[arm]['attainable_multi']>=gate['min_attainable_multi'])
        assert passed==reports[arm]['source_gate_passed']
    F.write(E.OUT/'paired_source_rows.json', rows)
    audit=dict(original159features_identical=True, labels_weights_identical=True,
        control_model_and_probabilities_identical=True, all_source_probabilities_replayed_exactly=True,
        observed_training_cells=len(train), source_paired_metrics=metrics,
        target_predictions_generated=False, target_scored=False, adopted=False,
        outcome='Source gate failed; do not evaluate Shenzhen or relax the gate.',
        limitation='Same development holdout selected thresholds; spatial intervals do not establish independent validation.')
    assert not reports['plus_composed_scope']['source_gate_passed']
    F.write(R.ROOT/'round14_independent_completion_audit.json', audit)
    print('AUDIT_COMPLETE',metrics['all'],metrics['single'],metrics['multi'],flush=True)


if __name__=='__main__':
    main()
