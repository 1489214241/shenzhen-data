"""Read-only verification of frozen round08, then write separate diagnostics."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import experiment_shenzhen_reference_retraining as E


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=E.ROOT/'round08_independent_completion_audit.json'
    if out.exists():
        raise FileExistsError('Preserve completed audit')
    c=E.verify()
    for name in ['probability_freeze.json','prediction_freeze.json']:
        E.check(E.OUT,name)
    history=E.F.read(E.OUT/'training_history.json')
    assert len(history)==30
    protocol_hash=sha(E.OUT/'protocol.json')
    for arm in E.ARMS:
        for seed in c['seeds']:
            ck=E.OUT/'checkpoints'/f'{arm}_{seed}.pt'
            item=torch.load(ck,map_location='cpu',weights_only=False)
            assert item['step']==40 and item['protocol_hash']==protocol_hash
            assert len(item['history'])==40
            assert all(np.isfinite(v) for r in item['history'] for v in r.values())
            assert sha(ck)==history[f'{arm}/{seed}']['checkpoint_sha256']
            with np.load(E.OUT/'probabilities'/arm/f'{seed}.npz') as p:
                for key in p.files:
                    assert np.isfinite(p[key]).all()
                    assert ((p[key]>=0)&(p[key]<=1)).all()
    inputs={a:E.load(c,a) for a in E.ARMS}
    bx,by=inputs['continued_rank']
    for a,(xs,ys) in inputs.items():
        for d in c['sources']:
            np.testing.assert_array_equal(np.load(E.OUT/'weights'/f'{a}_{d}.npz')['weights'],
                                          np.load(E.OUT/'weights'/f'continued_rank_{d}.npz')['weights'])
            for k in by[d]:np.testing.assert_array_equal(ys[d][k],by[d][k])
        for d in E.TARGETS:
            x=xs[d]
            assert not (ys[d]['labels']==0).any() and not len(ys[d]['valid_ids'])
            np.testing.assert_array_equal(x.anchor,bx[d].anchor)
            np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
    for d,h in E.F.read(E.BASE/'results.json')['gold_sha256'].items():
        assert sha(E.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in E.F.read(E.BASE/'scored_cases.json')}
    predictions=list((E.OUT/'predictions').rglob('*.json'))
    assert len(predictions)==132
    for p in predictions:
        rows=E.F.read(p)
        d=rows[0]['district']
        expected={k[1] for k in gold if k[0]==d}
        assert len(rows)==len(expected) and {r['site_id'] for r in rows}==expected
    results=E.F.read(E.OUT/'results.json')['reports']
    diagnostics={}
    for arm in E.ARMS:
        diagnostics[arm]={}
        for obj in E.OBJECTIVES:
            rows=[]
            for d in E.TARGETS:
                rows.extend(E.F.read(E.OUT/'predictions'/arm/obj/'ensemble'/f'{d}.json'))
            rows=[r for r in rows if gold[r['district'],r['site_id']]['has_map_gold']]
            assert len(rows)==1761
            stages={}
            for name in ['raw_school_ids','predicted_school_ids','after']:
                correct=multi_correct=fp=fn=0
                for r in rows:
                    truth=set(gold[r['district'],r['site_id']]['gold'])
                    pred=set(r['after'] if name=='after' else r['decision'][name])
                    correct+=pred==truth;multi_correct+=(pred==truth and len(truth)>1)
                    fp+=len(pred-truth);fn+=len(truth-pred)
                stages[name]=dict(correct=correct,multi_correct=multi_correct,fp=fp,fn=fn)
            final=results[arm][obj]['ensemble']['全部']['after']
            assert stages['after']['correct']==final['correct']
            deltas=[results[arm][obj][str(s)]['全部']['before']['delta_pp'] for s in c['seeds']]
            diagnostics[arm][obj]=dict(stages=stages,seed_delta_pp_range=[min(deltas),max(deltas)],
                positive_seeds=sum(v>0 for v in deltas),negative_seeds=sum(v<0 for v in deltas))
    report=dict(models=30,steps=40,finite_losses=True,prediction_files=132,scored=1761,unscored=41,
                source_weights_and_labels_identical=True,target_no_observed_negatives=True,
                own_property_references_masked=True,gold_unchanged=True,
                raw_stage_note='raw_school_ids is threshold selection after university base restriction, before source positives and policy supplementation.',
                diagnostics=diagnostics,adopted=False,independent_validation=False,
                script_sha256=sha(__file__),results_sha256=sha(E.OUT/'results.json'))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
