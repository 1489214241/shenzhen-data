"""Recheck provenance, masked supervision and delivered set arithmetic."""
import argparse
import numpy as np
import torch
import experiment_conditional_admission as A

def inputs():
    A.M.guard();c=A.verify();summary={}
    for arm in ['control','routed']:
        for d in c['sources']+c['targets']:
            cache=dict(np.load(A.OUT/'cache'/arm/f'{d}.npz'))
            x=A.M.pkl(A.E.INPUT/d/f'{arm}.pkl');y=dict(np.load(A.E.INPUT/d/'routed_labels.npz'))
            labels=y['labels'].copy();labels[np.arange(x.n),cache['prob'].argmax(1)]=-1
            np.testing.assert_array_equal(labels,cache['labels'])
            for k in ['train_ids','valid_ids','valid_labels']:np.testing.assert_array_equal(y[k],cache[k])
            assert not set(x.groups[cache['train_ids']])&set(x.groups[cache['valid_ids']])
            assert np.isfinite(cache['feature']).all()
            # Independent per-school mean recomputation, no einsum/features helper.
            ids=np.linspace(0,x.n-1,min(8,x.n),dtype=int)
            for q in ids:
                for s in range(x.school_n):
                    refs=np.flatnonzero((x.ref_schools==s)&(~x.self_mask[q]))
                    if len(refs):
                        w=np.exp(x.group_bias[refs]);expected=(x.pair[q,refs]*w[:,None]).sum(0)/w.sum()
                    else:expected=np.zeros(44)
                    np.testing.assert_allclose(cache['feature'][q,s,90:134],expected,atol=2e-5,rtol=2e-5)
            summary[arm+'/'+d]=dict(rows=x.n,schools=x.school_n,observed_admission_cells=int((labels>=0).sum()),
                source_splits_unchanged=True,independent_reference_mean_queries=len(ids))
    A.write(A.OUT/'input_audit.json',dict(passed=True,districts=summary,target_gold_read=False))
    print('ADMISSION INPUT AUDIT PASSED',flush=True)

def outputs():
    c=A.verify();A.check('weights_freeze.json');A.check('prediction_freeze.json')
    result=A.read(A.OUT/'results.json');replays=0;checked=0
    for arm in A.ARMS:
        input_arm=arm.split('_')[0];xs=A.data(c,input_arm)
        scale=dict(np.load(A.OUT/'scaling'/f'{input_arm}.npz'));payload=[]
        for seed in c['seeds']:
            model=A.Admission(len(scale['mean']),seed).to(A.M.DEVICE)
            model.load_state_dict(torch.load(A.OUT/'weights'/arm/f'{seed}.pt',map_location=A.M.DEVICE,weights_only=True))
            saved=dict(np.load(A.OUT/'probabilities'/arm/f'{seed}.npz'));payload.append(saved)
            for d in c['targets']:
                now=A.scores(model,xs[d],**scale)
                np.testing.assert_array_equal(now,saved[d]);replays+=1
        pp={d:np.stack([p[d] for p in payload]).mean(0) for d in xs}
        assert A.calibration(c,pp,xs)==result['source'][arm]
        threshold=result['source'][arm]['threshold']
        for label,probs in [('ensemble',pp)]+list(zip(map(str,c['seeds']),payload)):
            for d in c['targets']:
                meta=A.read(A.E.INPUT/d/'metadata.json');cons=A.read(A.E.INPUT/d/'constraints.json')
                ids=[s['school_id'] for s in meta['schools']]
                raw=A.read(A.OUT/'predictions'/f'{arm}_raw'/label/f'{d}.json')
                ver=A.read(A.OUT/'predictions'/f'{arm}_verified'/label/f'{d}.json')
                for q,(a,b) in enumerate(zip(raw,ver)):
                    p={ids[j] for j in np.flatnonzero(probs[d][q]>=threshold)}
                    assert set(a['after'])==p and ids[int(xs[d]['prob'][q].argmax())] in p
                    cc=cons[a['site_id']]
                    assert set(b['after'])==(p|set(cc['positive']))-set(cc['negative']);checked+=2
    for arm,mm in result['metrics'].items():
        rr=A.read(A.OUT/'scored_rows'/f'{arm}.json');assert len(rr)==1761
        counts=dict(correct=0,fp=0,fn=0)
        for r in rr:
            p=set(r['prediction']);g=set(r['truth'])
            assert r['correct']==(p==g)
            counts['correct']+=int(p==g);counts['fp']+=len(p-g);counts['fn']+=len(g-p)
        for k,v in counts.items():assert v==mm['全部'][k]
    for d,h in result['gold_sha256'].items():assert A.sha(A.M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    A.write(A.OUT/'output_audit.json',dict(passed=True,probability_arrays_replayed=replays,
        raw_verified_rows_checked=checked,all_ensemble_metrics_recounted=True,gold_unchanged=True,
        native_baseline_exact_replay=True,independent_validation=False))
    print('ADMISSION OUTPUT AUDIT PASSED',replays,checked,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['inputs','outputs']);globals()[ap.parse_args().phase]()
