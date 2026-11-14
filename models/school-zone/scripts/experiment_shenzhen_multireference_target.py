"""Conditional target test for the source-gated round16 reference intervention."""
from pathlib import Path
from collections import defaultdict
import argparse
import pickle
import shutil
import numpy as np
import torch
import experiment_shenzhen_source_multireferences as X
from experiment_shenzhen_resident_replay import decode
from audit_shenzhen_round16 import select

F=X.F;R=X.R
OUT=R.ROOT/'round16_multireference_target'
ROAD=R.ROOT/'road_corridor_scope_estimates'
PREVIOUS=R.ROOT/'round13_corridor_supplement'
DECODER_EXPERIMENT=None


def decoded(p,d,t,mode='absolute'):
    if mode=='relative':
        from experiment_shenzhen_relative_decode import relative
        p=relative(p)
    meta=F.read(R.OLD/'inputs'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
    facts=defaultdict(set)
    for r in F.read(ROAD/d/'estimates.json')['facts']:facts[r['site_id']].add(r['school_id'])
    rows=decode(p,d,t);raw=select(p,t)
    for i,r in enumerate(rows):
        r['raw_model_set']=[ids[j] for j in np.flatnonzero(raw[i])]
        r['before_corridor']=r['after'];r['corridor_added']=sorted(facts[r['site_id']]-set(r['after']))
        r['after']=sorted(set(r['after'])|facts[r['site_id']])
    return rows


def prepare():
    F.guard();c=X.verify()
    source_out=X.OUT if DECODER_EXPERIMENT is None else DECODER_EXPERIMENT.OUT
    if DECODER_EXPERIMENT is not None:DECODER_EXPERIMENT.verify()
    report=F.read(source_out/'source_report.json')
    audit_path=R.ROOT/('round16_independent_completion_audit.json' if DECODER_EXPERIMENT is None else 'round17_independent_completion_audit.json')
    audit=F.read(audit_path);candidate=report['selected_target_candidate']
    assert candidate is not None and candidate==audit['selected_target_candidate'],'No source-gate passing candidate'
    assert report['reports'][candidate]['source_gate_passed']
    if OUT.exists():raise FileExistsError('Preserve target experiment')
    links={}
    for folder,manifest in [(X.OUT,'input_freeze.json'),(X.OUT,'probability_freeze.json'),(X.TARGET,'manifest.json'),
                            (ROAD,'manifest.json'),(PREVIOUS,'prediction_freeze.json'),(R.OUT,'input_freeze.json')]:
        R.check(folder,manifest);links[str(folder/manifest)]=F.sha(folder/manifest)
        links.update({str(folder/p):h for p,h in F.read(folder/manifest)['files'].items()})
    links.update(F.read(X.OUT/'input_links.json')['files'])
    for p,h in F.read(ROAD/'manifest.json')['sources'].items():assert F.sha(p)==h;links[p]=h
    for p in [source_out/'source_report.json',audit_path]:links[str(p)]=F.sha(p)
    if DECODER_EXPERIMENT is not None:
        links.update(F.read(source_out/'input_links.json')['files']);links[str(source_out/'input_freeze.json')]=F.sha(source_out/'input_freeze.json')
        bank=candidate.removesuffix('_relative');arms=['control_absolute',bank+'_absolute',candidate]
        model_banks={arms[0]:'control',arms[1]:bank,arms[2]:bank};decoders={a:('relative' if a==candidate else 'absolute') for a in arms}
    else:
        arms=['control',candidate];model_banks={a:a for a in arms};decoders={a:'absolute' for a in arms}
    F.write(OUT/'protocol.json',dict(candidate=candidate,control=arms[0],arms=arms,model_banks=model_banks,decoders=decoders,seeds=c['seeds'],objectives=R.OBJECTIVES,
        primary='balanced_exact',source_gate_passed=True,source_input_intervention_only=DECODER_EXPERIMENT is None,
        target_inputs=str(X.TARGET),same_parent_steps_seeds_losses_and_target_inputs=True,
        policy='Same original-area university restriction, direct partial positives, scope implications, round07 community facts, round13 road estimates.',
        comparison='Absolute candidate versus absolute control isolates references; relative versus same-bank absolute isolates decoding. Combined candidate versus control changes both when relative is enabled. Previous native/latest and best-development are descriptive comparisons.',
        seed_reporting='Per-seed district, single/multi, exact/FP/FN and corrected/harmed; spatial confidence intervals for ensemble.',
        target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,scored=1761,unscored=41,independent_validation=False,adopted=False))
    thresholds={a:report['reports'][a]['selections'] for a in arms};F.write(OUT/'thresholds.json',thresholds)
    xs={}
    for d in R.TARGETS:
        with (X.TARGET/d/'inputs.pkl').open('rb') as h:xs[d]=pickle.load(h)
        x=xs[d];np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        assert not len(x.supervised) and (x.y==-1).all()
    start=R.M.LinearRetrieval(next(iter(xs.values())))
    for arm in arms:
        stored=[]
        for seed in c['seeds']:
            ck=torch.load(X.OUT/'checkpoints'/f'{model_banks[arm]}_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(ck['model']);model.eval()
            probs={d:R.probabilities(model,x,np.arange(x.n),c['chunk']) for d,x in xs.items()}
            assert all(np.isfinite(v).all() for v in probs.values());stored.append(probs)
            dest=OUT/'probabilities'/arm/f'{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**probs)
            print('TARGET_PROBABILITIES',arm,seed,flush=True)
        mean={d:np.stack([p[d] for p in stored]).mean(0) for d in R.TARGETS}
        for label,ps in [('ensemble',mean)]+[(str(s),p) for s,p in zip(c['seeds'],stored)]:
            for obj in R.OBJECTIVES:
                for d in R.TARGETS:
                    rows=decoded(ps[d],d,thresholds[arm][obj]['threshold'],decoders[arm])
                    assert len(rows)==xs[d].n and len({r['site_id'] for r in rows})==xs[d].n
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
    for p in [Path(__file__),F.REPO/'scripts/experiment_shenzhen_resident_replay.py',F.REPO/'scripts/shenzhen_partial_model.py',F.REPO/'scripts/audit_shenzhen_round16.py',F.REPO/'scripts/shenzhen_reference_retraining.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    if DECODER_EXPERIMENT is not None:
        for p in [F.REPO/'scripts/experiment_shenzhen_relative_multireference_target.py',F.REPO/'scripts/experiment_shenzhen_relative_decode.py']:
            dest=OUT/'code'/p.name;shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('TARGET_FROZEN',flush=True)


def verify():
    R.check(OUT,'prediction_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def replay():
    F.guard();c=verify();source=X.verify();thresholds=F.read(OUT/'thresholds.json');xs={};n=0
    if (OUT/'prediction_replay_audit.json').exists():raise FileExistsError('Preserve replay')
    for d in R.TARGETS:
        with (X.TARGET/d/'inputs.pkl').open('rb') as h:xs[d]=pickle.load(h)
    start=R.M.LinearRetrieval(next(iter(xs.values())))
    for arm in c['arms']:
        allp=[]
        for seed in c['seeds']:
            model=R.BernoulliRetrieval(41,27,start,seed,source['width']).to(R.DEVICE)
            model.load_state_dict(torch.load(X.OUT/'checkpoints'/f"{c['model_banks'][arm]}_{seed}.pt",map_location=R.DEVICE,weights_only=False)['model']);model.eval()
            ps=dict(np.load(OUT/'probabilities'/arm/f'{seed}.npz'));allp.append(ps)
            for d,x in xs.items():np.testing.assert_array_equal(R.probabilities(model,x,np.arange(x.n),source['chunk']),ps[d])
        mean={d:np.stack([p[d] for p in allp]).mean(0) for d in R.TARGETS}
        for label,ps in [('ensemble',mean)]+[(str(s),p) for s,p in zip(c['seeds'],allp)]:
            for obj in c['objectives']:
                for d in R.TARGETS:
                    rows=decoded(ps[d],d,thresholds[arm][obj]['threshold'],c['decoders'][arm])
                    assert rows==F.read(OUT/'predictions'/arm/obj/label/f'{d}.json');n+=len(rows)
        print('TARGET_REPLAYED',arm,flush=True)
    assert n==1802*len(c['arms'])*2*11
    F.write(OUT/'prediction_replay_audit.json',dict(prediction_rows=n,checkpoint_replays=len(c['arms'])*10,all_probabilities_bitwise_reproduced=True,
        all_decoded_predictions_reproduced=True,target_gold_read=False))


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c=verify();assert F.read(OUT/'prediction_replay_audit.json')['all_decoded_predictions_reproduced']
    if (OUT/'results.json').exists():raise FileExistsError('Preserve target results')
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};candidate=c['candidate']
    for d in R.TARGETS:
        with (X.TARGET/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={}
    for obj in c['objectives']:
        results[obj]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rows=[];unscored=[]
            for d in R.TARGETS:
                paths={a:OUT/'predictions'/a/obj/label/f'{d}.json' for a in c['arms']}
                paths.update(native_latest=PREVIOUS/'predictions/round12'/obj/label/f'{d}.json',best_development=PREVIOUS/'predictions/round07'/obj/label/f'{d}.json')
                data={a:{r['site_id']:r for r in F.read(p)} for a,p in paths.items()};assert all(set(v)==set(data[candidate]) for v in data.values())
                for sid,r in data[candidate].items():
                    g=gold[d,sid];row=dict(d=d,district=d,site_id=sid,building_name=r['building_name'],**{a:v[sid]['after'] for a,v in data.items()})
                    if not g['has_map_gold']:unscored.append(row);continue
                    row.update(gold=set(g['gold']),block=str(tuple(int(z//1000) for z in xy[d,sid])),semantic_pending=g['semantic_pending']);rows.append(row)
            assert len(rows)==1761 and len(unscored)==41
            groups=[('all',rows),('single',[r for r in rows if len(r['gold'])==1]),('multi',[r for r in rows if len(r['gold'])>1])]
            groups.extend((d,[r for r in rows if r['d']==d]) for d in R.TARGETS)
            groups.extend((k,[r for r in rows if bool(r['semantic_pending'])==flag]) for k,flag in [('semantic_pending',True),('not_flagged',False)])
            summaries={}
            for group,rr in groups:
                comparisons={}
                for before in [a for a in c['arms'] if a!=candidate]+['native_latest','best_development']:
                    delta=[int(set(r[candidate])==r['gold'])-int(set(r[before])==r['gold']) for r in rr]
                    comparisons[before]=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100))
                    if label=='ensemble':comparisons[before]['delta_95_pp']=bootstrap(rr,delta,seed=20261104,draws=3000)
                summaries[group]=dict(metrics={a:metrics(rr,a) for a in c['arms']+['native_latest','best_development']},comparisons=comparisons)
            results[obj][label]=summaries
            if label=='ensemble':
                F.write(OUT/'diagnostics_not_training'/f'{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows]);F.write(OUT/'unscored'/f'{obj}.json',unscored)
                print('TARGET_RESULT',obj,summaries['all'],flush=True)
    F.write(OUT/'results.json',dict(reports=results,candidate=candidate,scored=1761,unscored=41,gold_hashes_verified=True,
        predictions_frozen_before_scoring=True,source_gate_passed=True,independent_validation=False,adopted=False))


def score_audit():
    c=verify();result=F.read(OUT/'results.json');candidate=c['candidate']
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};checked=0
    for obj in c['objectives']:
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rows=[];outside=0
            for d in R.TARGETS:
                paths={a:OUT/'predictions'/a/obj/label/f'{d}.json' for a in c['arms']}
                paths.update(native_latest=PREVIOUS/'predictions/round12'/obj/label/f'{d}.json',best_development=PREVIOUS/'predictions/round07'/obj/label/f'{d}.json')
                banks={a:F.read(p) for a,p in paths.items()}
                for a,rr in banks.items():assert len(rr)==len({r['site_id'] for r in rr})
                banks={a:{r['site_id']:r for r in rr} for a,rr in banks.items()}
                assert all(set(v)==set(banks[candidate]) for v in banks.values())
                for sid in banks[candidate]:
                    g=gold[d,sid]
                    if not g['has_map_gold']:outside+=1;continue
                    rows.append(dict(d=d,truth=set(g['gold']),pending=bool(g['semantic_pending']),sets={a:set(v[sid]['after']) for a,v in banks.items()}))
            assert len(rows)==1761 and outside==41
            groups=[('all',rows),('single',[r for r in rows if len(r['truth'])==1]),('multi',[r for r in rows if len(r['truth'])>1])]
            groups.extend((d,[r for r in rows if r['d']==d]) for d in R.TARGETS)
            groups.extend((k,[r for r in rows if r['pending']==flag]) for k,flag in [('semantic_pending',True),('not_flagged',False)])
            for name,rr in groups:
                claim=result['reports'][obj][label][name]
                for arm in c['arms']+['native_latest','best_development']:
                    measured=dict(n=len(rr),correct=sum(r['sets'][arm]==r['truth'] for r in rr),fp=sum(len(r['sets'][arm]-r['truth']) for r in rr),fn=sum(len(r['truth']-r['sets'][arm]) for r in rr))
                    assert all(claim['metrics'][arm][k]==v for k,v in measured.items());checked+=1
                for before in [a for a in c['arms'] if a!=candidate]+['native_latest','best_development']:
                    delta=[int(r['sets'][candidate]==r['truth'])-int(r['sets'][before]==r['truth']) for r in rr]
                    assert claim['comparisons'][before]['corrected']==delta.count(1) and claim['comparisons'][before]['harmed']==delta.count(-1)
    audit_name='round16_target_completion_audit.json' if DECODER_EXPERIMENT is None else 'round17_target_completion_audit.json'
    F.write(R.ROOT/audit_name,dict(checked_metric_groups=checked,source_gate_verified=True,
        original_gold_hashes_unchanged=True,predictions_replayed_before_scoring=True,all_seeds_districts_single_multi_fp_fn_corrected_harmed_recounted=True,
        scored=1761,unscored=41,adopted=False,independent_validation=False))
    print('TARGET_SCORE_AUDIT_PASSED',checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','replay','score','score_audit']);globals()[p.parse_args().phase]()
