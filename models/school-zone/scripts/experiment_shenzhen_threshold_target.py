"""Conditional round36 target evaluation; gates precede any target inference.

The source process may still be running while this driver is prepared. No
target artifacts are created until all predictive and convergence gates pass.
"""
import argparse
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch
import experiment_shenzhen_threshold_continuation as SOURCE
import experiment_shenzhen_adaptive_threshold as T
import experiment_shenzhen_explicit_negative_target as BASE
from shenzhen_adaptive_threshold import ThresholdHead,query_features
from shenzhen_reviewed_supplements import ReviewedSupplements
from experiment_shenzhen_resident_replay import decode
from experiment_shenzhen_relative_decode import relative

F,R,E=SOURCE.F,SOURCE.R,SOURCE.E
OUT=R.ROOT/'round36_adaptive_threshold_target'
ARMS=['original_relative','constant_head','adaptive_head']
BANKS=ARMS+['best_development']
PAIRS=[('adaptive_head','original_relative'),('adaptive_head','constant_head'),('adaptive_head','best_development')]
AUDIT=R.ROOT/'round36_independent_completion_audit.json'
FACT_AUDIT=R.ROOT/'reviewed_supplements_completion_20261113'


def require_gate(admission,report,audit,history,seeds,tolerance):
    checks=dict(admission=admission.get('selected_target_candidate')=='adaptive_head',
        convergence=admission.get('all_adaptive_converged') is True,
        prediction=report.get('selected_target_candidate')=='adaptive_head',
        metric_audit=audit.get('all_metrics_independently_recomputed') is True and audit.get('source_gate_passed') is True,
        per_seed=all(history.get(f'adaptive_head/{s}',{}).get('reason')=='gradient_converged' and
            history[f'adaptive_head/{s}']['final']['gradient_inf']<=tolerance for s in seeds))
    if not all(checks.values()):raise ValueError('Target gate failed: '+str(checks))
    return checks


def source_ready():
    c=SOURCE.verify()
    require_gate(F.read(SOURCE.OUT/'admission_gate.json'),F.read(SOURCE.OUT/'source_report.json'),
        F.read(AUDIT),F.read(SOURCE.OUT/'training_history.json'),c['seeds'],c['gradient_tolerance'])
    R.check(SOURCE.OUT,'prediction_freeze.json')
    replay=F.read(SOURCE.OUT/'independent_replay_audit.json')
    assert replay['all_head_outputs_bitwise_replayed']
    assert all(replay['heads'][f'adaptive_head/{s}']['converged'] for s in c['seeds'])
    assert F.read(SOURCE.OUT/'seed_paired_audit.json')['independently_recomputed']
    return c


def raw_target(xs,c,seed):
    basec=E.verify();model=BASE.checkpoint_model(xs,basec,'features_labels',seed);payload={}
    for d,x in xs.items():
        z,f,p=T.extract(model,x,np.arange(x.n),c['chunk'])
        previous=np.load(BASE.OUT/'probabilities/features_labels'/f'{seed}.npz')[d]
        np.testing.assert_array_equal(p,previous)
        payload[d+'_z']=z;payload[d+'_f']=f;payload[d+'_probability']=p
    return payload


def head_margins(raw,c,seed,arm):
    ck=SOURCE.OUT/'checkpoints'/f'{arm}_{seed}.pt'
    saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
    state=saved['model']
    head=ThresholdHead(state['mean'].cpu().numpy(),state['scale'].cpu().numpy(),arm=='adaptive_head',seed,c['width']).to(R.DEVICE).double()
    head.load_state_dict(state);head.eval();out={}
    with torch.no_grad():
        for d in c['targets']:
            z=torch.as_tensor(raw[d+'_z'],device=R.DEVICE).double();f=torch.as_tensor(raw[d+'_f'],device=R.DEVICE).double()
            th=head(z,f).cpu().numpy();out[d+'_threshold']=th
            out[d]=raw[d+'_z'].astype(float)-th[:,None]
    return out


def target_rows(values,d,arm,thresholds,objective,supplement):
    values=relative(values) if arm=='original_relative' else np.asarray(values,float)
    threshold=thresholds[objective]['threshold'] if arm=='original_relative' else np.nextafter(np.float64(0),np.inf)
    return supplement.apply(decode(values,d,threshold))


def prepare():
    F.guard();c=source_ready()
    R.check(FACT_AUDIT,'manifest.json');assert F.read(FACT_AUDIT/'audit.json')['changed_rows']==0
    R.check(BASE.LATEST,'prediction_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve target input freeze')
    thresholds=F.read(E.OUT/'source_report.json')['reports']['features_labels_relative']['selections']
    protocol=dict(seeds=c['seeds'],arms=ARMS,objectives=R.OBJECTIVES,targets=c['targets'],
        source=str(SOURCE.OUT),primary_arm='adaptive_head',primary_objective='balanced_exact',
        gates='Source predictive gates and all10gradient-convergence gates independently verified before inference.',
        model='Same frozen round32 features_labels school scorer and target features for every arm; only TH selection changes.',
        thresholds='Heads use fixed strict margin>0 for both objectives. Original-relative control retains both frozen source calibration objectives. No target calibration.',
        facts='Identical cumulative round31 supplements plus4reviewed round23administrative identities, no extra transitive expansion.',
        historical='round31 round07 predictions with identical4facts appended. Source-only79288-row audit found no set changes; best959comparator remains descriptive, not matched training.',
        source_training_steps=0,comparisons=PAIRS,scored_sites=1761,unscored_sites=41,
        target_gold_read=False,target_polygon_read=False,independent_validation=False,adopted=False)
    supplements={d:ReviewedSupplements(d) for d in c['targets']};links=dict(F.read(SOURCE.OUT/'input_links.json')['files'])
    for folder,manifest in [(SOURCE.OUT,'prediction_freeze.json'),(BASE.LATEST,'prediction_freeze.json'),(FACT_AUDIT,'manifest.json')]:
        links[str(folder/manifest)]=F.sha(folder/manifest)
        m=F.read(folder/manifest);links.update({str(folder/p):h for p,h in m['files'].items()});links.update(m.get('sources',{}))
    for p in [AUDIT,SOURCE.OUT/'admission_gate.json',SOURCE.OUT/'source_report.json',SOURCE.OUT/'independent_replay_audit.json',SOURCE.OUT/'training_history.json',SOURCE.OUT/'seed_paired_audit.json']:
        links[str(p)]=F.sha(p)
    for d,s in supplements.items():
        links.update(s.hashes)
        for p in [R.OLD/'inputs'/f'{d}.json',R.OUT/'policies'/f'{d}.json',R.OLD/'source_graphs'/f'{d}.json']:links[str(p)]=F.sha(p)
    for name in ['experiment_shenzhen_threshold_target.py','experiment_shenzhen_threshold_continuation.py','experiment_shenzhen_adaptive_threshold.py',
                 'shenzhen_adaptive_threshold.py','shenzhen_reviewed_supplements.py','shenzhen_current_supplements.py','experiment_shenzhen_resident_replay.py',
                 'shenzhen_partial_model.py','shenzhen_reference_retraining.py','shenzhen_scope_implication.py','experiment_shenzhen_explicit_negative_target.py']:
        p=F.REPO/'scripts'/name;links[str(p)]=F.sha(p)
    F.write(OUT/'protocol.json',protocol);F.write(OUT/'thresholds.json',thresholds);F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('TARGET_INPUTS_FROZEN',flush=True)


def verify_inputs():
    c=source_ready();R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return c,F.read(OUT/'protocol.json')


def infer():
    F.guard();c,cfg=verify_inputs()
    if (OUT/'prediction_freeze.json').exists():R.check(OUT,'prediction_freeze.json');return
    xs=BASE.target_inputs(E.verify(),'features_labels')
    supplements={d:ReviewedSupplements(d) for d in c['targets']};thresholds=F.read(OUT/'thresholds.json');banks={a:[] for a in ARMS}
    for seed in c['seeds']:
        raw=raw_target(xs,c,seed);p=OUT/'raw'/f'{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            old=dict(np.load(p))
            for k,v in raw.items():np.testing.assert_array_equal(v,old[k])
        else:np.savez_compressed(p,**raw)
        banks['original_relative'].append({d:raw[d+'_probability'] for d in c['targets']})
        for arm in T.ARMS:
            values=head_margins(raw,c,seed,arm);banks[arm].append({d:values[d] for d in c['targets']})
            p=OUT/'margins'/arm/f'{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                old=dict(np.load(p))
                for k,v in values.items():np.testing.assert_array_equal(v,old[k])
            else:np.savez_compressed(p,**values)
        print('TARGET_INFERRED',seed,flush=True)
    for arm in ARMS:
        stored=banks[arm];mean={d:np.stack([v[d] for v in stored]).mean(0) for d in c['targets']}
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],stored)]:
            for obj in cfg['objectives']:
                for d in c['targets']:
                    rows=target_rows(values[d],d,arm,thresholds,obj,supplements[d])
                    if arm=='original_relative':
                        old=F.read(BASE.OUT/'predictions/features_labels'/obj/label/f'{d}.json')
                        aligned=supplements[d].augment_historical(old)
                        assert [(r['site_id'],r['after']) for r in rows]==[(r['site_id'],r['after']) for r in aligned]
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
    for label in ['ensemble']+list(map(str,c['seeds'])):
        for obj in cfg['objectives']:
            for d in c['targets']:
                old=F.read(BASE.LATEST/'predictions/round07'/obj/label/f'{d}.json')
                aligned=supplements[d].augment_historical(old)
                assert [r['after'] for r in old]==[r['after'] for r in aligned]
                F.write(OUT/'predictions/best_development'/obj/label/f'{d}.json',aligned)
    verify_inputs()
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['raw','margins','predictions'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    print('TARGET_PREDICTIONS_FROZEN',flush=True)


def verify():
    c,cfg=verify_inputs();R.check(OUT,'prediction_freeze.json');return c,cfg


def replay():
    F.guard();c,cfg=verify();dest=OUT/'prediction_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve target replay')
    xs=BASE.target_inputs(E.verify(),'features_labels');supplements={d:ReviewedSupplements(d) for d in c['targets']}
    thresholds=F.read(OUT/'thresholds.json');banks={a:[] for a in ARMS};rows_checked=0
    for seed in c['seeds']:
        raw=dict(np.load(OUT/'raw'/f'{seed}.npz'));actual=raw_target(xs,c,seed)
        for k,v in actual.items():np.testing.assert_array_equal(v,raw[k])
        banks['original_relative'].append({d:raw[d+'_probability'] for d in c['targets']})
        for arm in T.ARMS:
            actual=head_margins(raw,c,seed,arm);saved=dict(np.load(OUT/'margins'/arm/f'{seed}.npz'))
            for k,v in actual.items():np.testing.assert_array_equal(v,saved[k])
            banks[arm].append({d:saved[d] for d in c['targets']})
        print('TARGET_MODEL_REPLAYED',seed,flush=True)
    for arm in ARMS:
        stored=banks[arm];mean={d:np.stack([p[d] for p in stored]).mean(0) for d in c['targets']}
        for label,p in [('ensemble',mean)]+[(str(s),p) for s,p in zip(c['seeds'],stored)]:
            for obj in cfg['objectives']:
                for d in c['targets']:
                    actual=target_rows(p[d],d,arm,thresholds,obj,supplements[d])
                    assert actual==F.read(OUT/'predictions'/arm/obj/label/f'{d}.json');rows_checked+=len(actual)
    assert rows_checked==118932
    F.write(dest,dict(base_models=10,heads=20,prediction_rows=rows_checked,all_probabilities_features_margins_and_decisions_reproduced=True,
        target_gold_read=False,target_polygon_read=False))


def evaluation_rows(obj,label):
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};rows=[];outside=[]
    for d in R.TARGETS:
        with (E.B.OUT/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy=dict(zip(dist.site_ids,dist.coords));banks={}
        for arm in BANKS:
            values=F.read(OUT/'predictions'/arm/obj/label/f'{d}.json')
            assert len(values)==len({r['site_id'] for r in values});banks[arm]={r['site_id']:r for r in values}
        assert all(set(b)==set(banks['adaptive_head']) for b in banks.values())
        for site,r in banks['adaptive_head'].items():
            g=gold[d,site];row=dict(d=d,district=d,site_id=site,building_name=r['building_name'],**{a:b[site]['after'] for a,b in banks.items()})
            if not g['has_map_gold']:outside.append(row);continue
            row.update(gold=set(g['gold']),block=str(tuple(int(v//1000) for v in xy[site])),semantic_pending=bool(g['semantic_pending']));rows.append(row)
    assert len(rows)==1761 and len(outside)==41
    return rows,outside


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c,cfg=verify();hashes=BASE.check_gold()
    assert F.read(OUT/'prediction_replay_audit.json')['all_probabilities_features_margins_and_decisions_reproduced']
    if (OUT/'results.json').exists():raise FileExistsError('Preserve target score')
    reports={}
    for obj in cfg['objectives']:
        reports[obj]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rows,outside=evaluation_rows(obj,label);groups={}
            for name,rs in BASE.groups(rows):
                effects={}
                for a,b in PAIRS:
                    delta=[int(set(r[a])==r['gold'])-int(set(r[b])==r['gold']) for r in rs]
                    v=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100))
                    if label=='ensemble':v['delta_95_pp']=bootstrap(rs,delta,seed=20261104,draws=3000)
                    effects[a+'_vs_'+b]=v
                groups[name]=dict(metrics={a:metrics(rs,a) for a in BANKS},comparisons=effects)
            reports[obj][label]=groups
            if label=='ensemble':
                F.write(OUT/'diagnostics_not_training'/f'{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
                F.write(OUT/'unscored'/f'{obj}.json',outside);print('TARGET_RESULT',obj,groups['all'],flush=True)
    F.write(OUT/'results.json',dict(reports=reports,gold_sha256=hashes,scored=1761,unscored=41,source_and_convergence_gates_passed=True,
        predictions_frozen_before_scoring=True,independent_validation=False,adopted=False))


def audit():
    from zoning.evaluate import bootstrap
    c,cfg=verify();result=F.read(OUT/'results.json');assert result['gold_sha256']==BASE.check_gold()
    dest=R.ROOT/'round36_target_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve target completion audit')
    count=0
    for obj in cfg['objectives']:
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rows,outside=evaluation_rows(obj,label)
            for name,rs in BASE.groups(rows):
                claim=result['reports'][obj][label][name]
                for a in BANKS:
                    v=dict(n=len(rs),correct=sum(set(r[a])==r['gold'] for r in rs),fp=sum(len(set(r[a])-r['gold']) for r in rs),fn=sum(len(r['gold']-set(r[a])) for r in rs))
                    assert all(claim['metrics'][a][k]==value for k,value in v.items())
                    assert claim['metrics'][a]['exact_accuracy']==v['correct']/v['n'];count+=1
                for a,b in PAIRS:
                    delta=[int(set(r[a])==r['gold'])-int(set(r[b])==r['gold']) for r in rs]
                    v=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100))
                    if label=='ensemble':v['delta_95_pp']=bootstrap(rs,delta,seed=20261104,draws=3000)
                    assert v==claim['comparisons'][a+'_vs_'+b];count+=1
    assert count==2*11*7*7
    F.write(dest,dict(metric_groups=count,gold_unchanged=True,source_and_convergence_gates_verified=True,
        all_seeds_districts_single_multi_fp_fn_harms_and_intervals_recomputed=True,scored=1761,unscored=41,
        target_adopted=False,independent_validation=False,audit_code_sha256=F.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','replay','score','audit']);globals()[p.parse_args().phase]()
