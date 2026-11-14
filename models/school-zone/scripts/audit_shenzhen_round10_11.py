"""Independent artifact and paired-source audit for the input replay and probe."""
import os
os.environ['OMP_NUM_THREADS']='6'
import pickle
import numpy as np
import experiment_shenzhen_resident_replay as A
import experiment_shenzhen_relation_probe as B
from shenzhen_relation_probe import school_vectors
from zoning.evaluate import bootstrap

F=A.F
R=A.R


def sets(probabilities,threshold):
    pred=probabilities>=threshold
    empty=~pred.any(1);pred[empty,probabilities[empty].argmax(1)]=True
    return pred


def main():
    F.guard()
    A.verify_links();R.check(A.OUT,'prediction_freeze.json')
    c=F.read(A.OUT/'protocol.json');counts={}
    for objective in c['objectives']:
        counts[objective]={}
        for label in ['ensemble']+[str(s) for s in c['seeds']]:
            changed=rows=0
            for d in R.TARGETS:
                before=F.read(R.OUT/'predictions'/A.ARM/objective/label/f'{d}.json')
                after=F.read(A.OUT/'predictions'/objective/label/f'{d}.json')
                assert len(before)==len(after)
                for old,new in zip(before,after):
                    assert (old['district'],old['site_id'])==(new['district'],new['site_id'])
                    assert old['late_source_positives']==new['late_source_positives']
                    changed+=set(old['after'])!=set(new['after']);rows+=1
            assert rows==1802
            counts[objective][label]=dict(rows=rows,changed_final_sets=int(changed))
    assert len(list((A.OUT/'predictions').rglob('*.json')))==44
    F.write(R.ROOT/'round10_independent_completion_audit.json',dict(predictions=counts,
        frozen_inputs_and_models_verified=True,prediction_files=44,no_new_thresholds=True,
        source_of_gold_integrity='Separate score stage verified original target Gold hashes; this audit never opens target labels.'))

    c=B.verify();R.check(B.OUT,'probability_freeze.json')
    for rel,h in F.read(R.OLD/'input_audit.json')['source_hashes'].items():assert F.sha(F.REPO/rel)==h,rel
    xs,ys=R.load(F.read(R.OUT/'protocol.json'),B.ARM)
    for d in R.TARGETS:
        with (B.NEW/d/'inputs.pkl').open('rb') as handle:xs[d]=pickle.load(handle)
    inp=F.read(B.OUT/'input_audit.json');train=np.load(B.OUT/'train_features.npy',mmap_mode='r')
    data=dict(np.load(B.OUT/'train_labels_weights.npz'));verification={}
    for d,x in xs.items():
        ids=ys[d]['train_ids'];y=ys[d]['labels'][ids];mask=y>=0
        info=inp['districts'][d];sl=slice(info['cell_start'],info['cell_end'])
        np.testing.assert_array_equal(y[mask],data['labels'][sl])
        w=np.load(R.OUT/'weights'/f'{B.ARM}_{d}.npz')['weights'][ids]
        expected=np.broadcast_to(w[:,None],y.shape)[mask]/inp['weight_normalizer']
        np.testing.assert_array_equal(expected,data['weights'][sl])
        sample=ids[:3];v=school_vectors(x,sample);selected=v[y[:3]>=0]
        np.testing.assert_allclose(train[info['cell_start']:info['cell_start']+len(selected)],selected,
            rtol=1e-6,atol=1e-6,equal_nan=True)
        assert np.array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        verification[d]=dict(labels_exact=True,weight_ratios_exact=True,first_three_feature_rows_rebuilt=True,
            own_property_mask_verified=True,target_negative_cells=int((data['labels'][sl]==0).sum()) if d in R.TARGETS else None)
    report=F.read(B.OUT/'source_report.json')['reports'];source_rows={}
    seeds=F.read(R.OUT/'protocol.json')['seeds']
    old=[dict(np.load(R.OUT/'probabilities'/B.ARM/f'{s}.npz')) for s in seeds]
    for arm in c['arms']:
        payload=dict(np.load(B.OUT/'probabilities'/f'{arm}.npz'));width=c['features'][arm]
        with (B.OUT/'models'/f'{arm}.pkl').open('rb') as handle:model=pickle.load(handle)
        assert model.n_iter_==c['parameters']['max_iter']
        rows=[]
        for d in c['sources']:
            v=dict(np.load(B.OUT/'valid'/f'{d}.npz'));features=v['features']
            p=model.predict_proba(features.reshape(-1,159)[:,:width])[:,1].reshape(features.shape[:2])
            np.testing.assert_array_equal(p,payload[d+'_valid'])
            baseline=np.stack([z[d] for z in old]).mean(0)
            before=sets(baseline,report['baseline']['selections']['balanced_exact']['threshold'])
            after=sets(p,report[arm]['selections']['balanced_exact']['threshold'])
            coords=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
            for j,q in enumerate(v['ids']):
                truth=v['labels'][j]
                rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in coords[q])),
                    gold=np.flatnonzero(truth).tolist(),before=np.flatnonzero(before[j]).tolist(),after=np.flatnonzero(after[j]).tolist(),
                    before_ok=bool((before[j]==truth).all()),after_ok=bool((after[j]==truth).all()),multi=bool(truth.sum()>1)))
        summary={}
        for group,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['after_ok'])-int(r['before_ok']) for r in rr]
            summary[group]=dict(n=len(rr),before_correct=sum(r['before_ok'] for r in rr),after_correct=sum(r['after_ok'] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
                delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
        source_rows[arm]=summary
        F.write(B.OUT/'source_diagnostics'/f'{arm}.json',rows)
    F.write(R.ROOT/'round11_independent_completion_audit.json',dict(input_verification=verification,
        model_predictions_reproduced_exactly=True,source_hashes_unchanged=True,new_models=2,
        source_paired_metrics=source_rows,target_predictions_generated=False,target_scored=False,
        gate_failures_preserved=True,adopted=False,
        uncertainty='Spatial intervals are descriptive development diagnostics; same source holdout selected thresholds, not independent validation.'))
    print('AUDITS_COMPLETE',source_rows,flush=True)


if __name__=='__main__':main()
