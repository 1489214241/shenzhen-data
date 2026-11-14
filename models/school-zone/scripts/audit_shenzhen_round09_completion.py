"""Independently check round09 artifacts and raw/source multi-set diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
import experiment_shenzhen_schoolwise_mass as R
from shenzhen_exact_calibration import calibration_intervals


def main():
    out=R.E.ROOT/'round09_independent_completion_audit.json'
    if out.exists():raise FileExistsError('Preserve audit')
    c=R.verify();R.E.check(R.OUT,'probability_freeze.json');R.E.check(R.OUT,'prediction_freeze.json')
    results=R.F.read(R.OUT/'results.json')['reports'];history=R.F.read(R.OUT/'training_history.json')
    assert len(history)==30
    for arm in R.ARMS:
        for seed in c['seeds']:
            ck=R.OUT/'checkpoints'/f'{arm}_{seed}.pt';p=R.OUT/'probabilities'/arm/f'{seed}.npz'
            h=history[f'{arm}/{seed}'];assert R.F.sha(ck)==h['checkpoint_sha256'] and R.F.sha(p)==h['probabilities_sha256']
            saved=torch.load(ck,map_location='cpu',weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40
            assert all(np.isfinite(v) for r in saved['history'] for v in r.values())
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
            expected=R.E.OUT if arm=='property_mass' else R.OUT
            assert saved['protocol_hash']==R.F.sha(expected/'protocol.json')
            with np.load(p) as z:
                assert all(np.isfinite(z[k]).all() and ((z[k]>=0)&(z[k]<=1)).all() for k in z.files)
    for d,h in R.F.read(R.E.BASE/'results.json')['gold_sha256'].items():assert R.F.sha(R.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    xs,ys=R.load(c)
    for d,x in xs.items():
        np.testing.assert_array_equal(np.load(R.OUT/'weights'/f'{d}.npz')['weights'],
                                      np.load(R.E.OUT/'weights'/f'new_references_labels_{d}.npz')['weights'])
        np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        if d in R.TARGETS:assert not (ys[d]['labels']==0).any() and not len(ys[d]['valid_ids'])
    gold={(r['district'],r['site_id']):r for r in R.F.read(R.E.BASE/'scored_cases.json')}
    predictions=list((R.OUT/'predictions').rglob('*.json'));assert len(predictions)==132
    for p in predictions:
        rows=R.F.read(p);d=rows[0]['district'];expected={s for dd,s in gold if dd==d}
        assert len(rows)==len(expected) and {r['site_id'] for r in rows}==expected
        if p.relative_to(R.OUT/'predictions').parts[0]=='property_mass':
            rel=p.relative_to(R.OUT/'predictions/property_mass')
            assert rows==R.F.read(R.E.OUT/'predictions/new_references_labels'/rel)
    diagnostics={}
    for arm in R.ARMS:
        loaded=[dict(np.load(R.OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in c['sources']}
        intervals,sizes=calibration_intervals([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']])
        source=dict(n=len(sizes),multi=int((sizes>1).sum()),multi_ordering_capacity=sum(k>1 for lo,hi,k in intervals))
        diagnostics[arm]=dict(source=source,objectives={})
        cal=R.F.read(R.OUT/'calibration_summary.json')[arm]
        for obj in R.OBJECTIVES:
            rows=[]
            for d in R.TARGETS:rows+=R.F.read(R.OUT/'predictions'/arm/obj/'ensemble'/f'{d}.json')
            rows=[r for r in rows if gold[r['district'],r['site_id']]['has_map_gold']];assert len(rows)==1761
            stages={}
            for stage in ['raw_school_ids','predicted_school_ids','after']:
                correct=multi_correct=fp=fn=0
                for r in rows:
                    truth=set(gold[r['district'],r['site_id']]['gold'])
                    pred=set(r['after'] if stage=='after' else r['decision'][stage])
                    correct+=pred==truth;multi_correct+=(pred==truth and len(truth)>1)
                    fp+=len(pred-truth);fn+=len(truth-pred)
                stages[stage]=dict(correct=correct,multi_correct=multi_correct,fp=fp,fn=fn)
            assert stages['after']['correct']==results[arm][obj]['ensemble']['全部']['after']['correct']
            deltas=[results[arm][obj][str(s)]['全部']['before']['delta_pp'] for s in c['seeds']]
            diagnostics[arm]['objectives'][obj]=dict(stages=stages,source_calibration=cal[obj],
                delta_pp_range=[min(deltas),max(deltas)],positive_seeds=sum(v>0 for v in deltas),negative_seeds=sum(v<0 for v in deltas))
    report=dict(models=30,new_fits=20,reused_matched_controls=10,steps=40,prediction_files=132,
        scored=1761,unscored=41,gold_unchanged=True,controls_replayed_exactly=True,
        source_weights_identical=True,target_no_observed_negatives=True,own_property_references_masked=True,
        raw_stage_note='Threshold selection after university base restriction, before source positives/policy supplementation.',
        diagnostics=diagnostics,adopted=False,independent_validation=False,
        results_sha256=R.F.sha(R.OUT/'results.json'),script_sha256=R.F.sha(Path(__file__)))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
