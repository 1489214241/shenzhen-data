"""Replay the frozen set decoder and metrics through an independent implementation."""
import json
import math
import sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import experiment_modernnca_conditional_sets as E


def independent_pick(p,candidates,ids,beta):
    index={s:i for i,s in enumerate(ids)}
    def key(c):
        ss=c['school_ids']
        v=sum(math.log(max(float(p[index[s]]),1e-30)) for s in ss)/len(ss)+beta*math.log(len(ss))
        return (-v,len(ss),ss)
    return min(candidates,key=key)['school_ids']


def main():
    c=E.verify();E.check('candidate_freeze.json');E.check('prediction_freeze.json')
    assert E.read(E.OUT/'protocol_receipt.json')['protocol_sha256']==E.sha(E.OUT/'protocol.json')
    pp=E.probabilities();mean={d:np.stack([p[d] for p in pp]).mean(0) for d in pp[0]}
    beta=E.read(E.OUT/'source_selection.json')['selected']['beta'];checked=0
    for label,probs in [('ensemble',mean)]+list(zip(map(str,c['seeds']),pp)):
        for d in E.M.TARGETS:
            meta=E.read(E.M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
            sites=E.read(E.OUT/'site_candidates'/f'{d}.json')
            old=E.read(E.PARENT/'predictions/categorical_policy'/label/f'{d}.json')
            for arm in E.ARMS:
                rows=E.read(E.OUT/'predictions'/arm/label/f'{d}.json')
                for q,(row,site) in enumerate(zip(rows,sites)):
                    assert row['site_id']==site['site_id']==old[q]['site_id']
                    if arm=='categorical_policy':expected=old[q]['after']
                    elif arm=='categorical_plus_named_evidence':expected=sorted(set(old[q]['after'])|set(site['new_named_ids']))
                    else:
                        cats=site['same_evidence' if arm=='set_same_evidence' else 'plus_named']
                        expected=independent_pick(probs[d][q],cats,ids,beta)
                        assert not (set(expected)&set(site['exclusions']))
                        assert not row['decision']['complete_school_set']
                        assert not row['decision']['enrollment_eligibility']
                    assert row['after']==expected,(arm,label,d,q)
                    checked+=1
    # Independently check source candidates and all source calibration outcomes.
    for d in c['sources']:
        y=dict(np.load(E.M.OLD/'inputs'/f'{d}_labels.npz'))
        train=y['labels'][y['train_ids']];complete=train[(train>=0).all(1)]
        expected={(j,) for j in range(train.shape[1])}|{tuple(np.flatnonzero(r==1)) for r in complete}
        got={tuple(r['school_ids']) for r in E.read(E.OUT/'source_candidates'/f'{d}.json')}
        assert got==expected
    selection=E.read(E.OUT/'source_selection.json')
    for trial in selection['trials']:
        cs=cm=fp=ns=nm=0
        for d in c['sources']:
            y=dict(np.load(E.M.OLD/'inputs'/f'{d}_labels.npz'))['valid_labels'];ids=list(range(y.shape[1]))
            cats=E.read(E.OUT/'source_candidates'/f'{d}.json')
            for p,row in zip(mean[d],y):
                pred=set(independent_pick(p,cats,ids,trial['beta']));gold=set(np.flatnonzero(row))
                if len(gold)>1:nm+=1;cm+=int(pred==gold)
                else:ns+=1;cs+=int(pred==gold)
                fp+=len(pred-gold)
        assert (cs,cm,ns,nm,fp)==tuple(trial[k] for k in ['single_correct','multi_correct','single_n','multi_n','fp'])
    expected=max(selection['trials'],key=lambda r:(r['balanced'],r['correct'],-r['fp'],-r['beta']))
    assert expected==selection['selected']
    result=E.read(E.OUT/'results.json')
    for arm in E.ARMS:
        rows=E.read(E.OUT/'scored_rows'/f'{arm}.json')
        assert len(rows)==1761 and sum(r['semantic_pending'] for r in rows)==242
        correct=fp=fn=0
        for r in rows:
            p=set(r['prediction']);g=set(r['truth'])
            assert r['correct']==(p==g)
            assert (r['tp'],r['fp'],r['fn'])==(len(p&g),len(p-g),len(g-p))
            correct+=int(p==g);fp+=len(p-g);fn+=len(g-p)
        m=result['metrics'][arm]['全部'];assert (correct,fp,fn)==(m['correct'],m['fp'],m['fn'])
    # The baseline must reproduce the prior experiment, not a moving target.
    parent=E.read(E.PARENT/'results.json')['metrics']['categorical_policy']['全部']
    assert parent==result['metrics']['categorical_policy']['全部']
    for d,h in result['gold_sha256'].items():assert E.sha(E.M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    E.write(E.OUT/'audit.json',dict(passed=True,output_rows_independently_replayed=checked,
        source_candidate_pool_train_only=True,source_calibration_recounted=True,
        baseline_exactly_reproduced=True,semantic_pending_retained=242,gold_unchanged=True,
        protocol_and_input_and_prediction_hashes_verified=True,
        limitation='Candidate/AST provenance pins prior parser output; this is not an independent audit of every source document interpretation.'))
    print('AUDIT PASSED',checked,flush=True)


if __name__=='__main__':main()
