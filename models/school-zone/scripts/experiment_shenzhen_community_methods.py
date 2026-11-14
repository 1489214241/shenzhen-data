"""Frozen-model Shenzhen comparison of two existing community estimators.

prepare/infer/replay never read target labels; score runs only after freezing all
five arms. This is a deployment-input comparison, not a retraining experiment.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import torch

import shenzhen_full_inputs as F
from zoning import evidence as E, model as M, data, DEVICE
from shenzhen_partial_model import BernoulliRetrieval, predict_sets
from experiment_shenzhen_partial_labels import probabilities
from shenzhen_reviewed_supplements import ReviewedSupplements
from shenzhen_community_comparison import nearest_separated, identity, community_features, direct_estimates

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'community_methods_20261113'
RAW=F.REPO/'data/深圳市/修复实验输入_20261112'
OLD=ROOT/'round03_partial_labels'
PARENT=ROOT/'round06_balanced_ranking'
BEST=ROOT/'round31_explicit_scope/combined'
ARMS=['current','changsha_E','hangzhou_E','changsha_ER','hangzhou_ER']
TARGETS=['龙华区','龙岗区']
SEEDS=list(range(20261131,20261141))
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False


def guard():
    F.guard()
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/')
            if any(t in p for t in ['/diagnostics_not_training/','/scored_cases.json','/results.json','/eval.json']):
                raise PermissionError('Community comparison input/inference cannot read scored artifacts: '+p)
    sys.addaudithook(hook)


def pkl(path):
    with Path(path).open('rb') as f:return pickle.load(f)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:pickle.dump(value,f,protocol=5)


def hz_reproduction():
    from pyproj import Transformer
    transform=Transformer.from_crs(4326,32651,always_xy=True)
    result={};paths=[]
    for d in ['拱墅区','萧山区','富阳区','临安区']:
        source=F.ROOT/'科研推进/社区坐标新用法_20261104/raw/mca_hz'/f'{d}.json'
        a=F.REPO/'data'/d/'arrays.npz';b=F.REPO/'data'/d/'community_mca.npz'
        paths.extend([source,a,b]);units=defaultdict(list)
        for r in F.read(source):
            if r.get('place_type') not in ('社区','行政村'):continue
            g=r.get('gdm') or {}
            if g.get('type') not in ('point','multipoint'):continue
            c=g['coordinates'][0] if g['type']=='multipoint' else g['coordinates']
            units[r['standard_name']].append(transform.transform(*c))
        coords=np.load(a)['coords'];saved=np.load(b)
        points=np.array([p for ps in units.values() for p in ps]);names=[name for name,ps in units.items() for _ in ps]
        est=nearest_separated(coords,points,names)
        actual=np.where(est['strict']|est['loose'],np.array(names)[est['point']],'')
        np.testing.assert_array_equal(est['strict'],saved['community_strict'])
        np.testing.assert_array_equal(actual,saved['community_name'])
        assert not np.any((est['distance']==0)&(est['second_distance']==0))
        result[d]=dict(n=len(coords),strict=int(est['strict'].sum()),loose=int(est['loose'].sum()),all_identity_and_confidence_rows_identical=True)
    return result,paths


def patch_features(x,names,strict,cs):
    out=deepcopy(x)
    both=strict[:,None]&strict[x.ref_sites][None]
    same=(names[:,None]==names[x.ref_sites][None])&(names[:,None]!='')
    out.pair[:,:,out.pair_blocks['same_community']]=np.stack([same&both,(~same)&both],-1).astype(np.float32)
    out.anchor[:,:,out.anchor_blocks['community_clause']]=cs
    for k,v in x.pair_blocks.items():
        if k!='same_community':np.testing.assert_array_equal(out.pair[:,:,v],x.pair[:,:,v])
    for k,v in x.anchor_blocks.items():
        if k!='community_clause':np.testing.assert_array_equal(out.anchor[:,:,v],x.anchor[:,:,v])
    return out


def prepare():
    guard()
    if OUT.exists():raise FileExistsError('Keep the existing frozen community comparison')
    protocol=dict(date='2026-11-13',arms=ARMS,targets=TARGETS,seeds=SEEDS,
        parent='round06 balanced_rank; current final output must reproduce round31 round07',
        steps=0,weights='Same ten frozen checkpoints; no retraining or calibration',chunk=64,
        threshold='Unchanged source-only balanced_exact threshold from round06',
        primary_comparisons=[['hangzhou_E','changsha_E'],['hangzhou_ER','changsha_ER']],
        control='current retains all historical features and reference estimates, for exact replay',
        E='Same original reference bank; shared positive-scope binding and exact unit identities; only community feature columns change',
        ER='E plus Hangzhou-style singleton whole-community estimates for previously unknown sites; union into original references and final partial positives; all other target labels remain unknown',
        shared_identity='Exact existing policy unit identity; role-stripped POI fallback, no fuzzy aliasing or parent/subcommittee merge',
        shared_scope='Positive membership nodes only; AND/MINUS partial; exclusions never positive; unknown/conditional/ref not inferred',
        carryover='All arms retain original Changsha-derived rule references; this estimates incremental replacement on the current pipeline, not a clean-room no-reference factorial',
        sources='Same frozen Baidu inventory and metric coordinates for both estimators; no claim these are MCA-quality units',
        inference_policy='Same university/shared rules, refreshed graph and cumulative reviewed supplements in all arms',
        scored_population=1761,unscored_population=41,adoption_rule={
            'candidate_family':ARMS[1:], 'primary_metric':'exact final set',
            'simultaneous_interval':'1km district-stratified paired bootstrap, 3000 draws; Bonferroni 98.75 percent interval versus current',
            'eligibility':'positive lower bound; neither district nor single/multi exact counts decrease; at least8of10seed deltas nonnegative',
            'selection':'highest ensemble exact among eligible, ties prefer E then changsha; otherwise retain current',
            'status':'user-authorized local development selection, not independent generalization proof'},
        target_gold_read=False,target_polygons_read=False,independent_validation=False)
    F.write(OUT/'protocol.json',protocol)
    reproduced,hzpaths=hz_reproduction();F.write(OUT/'hangzhou_reproduction.json',reproduced)
    cal=PARENT/'source_calibration/balanced_rank.json'
    threshold=F.read(cal)['selections']['balanced_exact']['threshold']
    F.write(OUT/'threshold.json',dict(value=threshold))
    paths=[cal,PARENT/'protocol.json',OLD/'protocol.json']+hzpaths
    paths += [PARENT/'checkpoints'/f'balanced_rank_{s}.pt' for s in SEEDS]
    paths += [PARENT/'probabilities/balanced_rank'/f'{s}.npz' for s in SEEDS]
    audit={}
    for d in TARGETS:
        dist=pkl(RAW/d/'district.pkl');rr=pkl(RAW/d/'rules.pkl');x=pkl(OLD/'inputs'/f'{d}.pkl')
        assert dist.answers=={} and dist.train_answers=={}
        meta=F.read(OLD/'inputs'/f'{d}.json')
        assert {str(q):v for q,v in rr.known.items()}==meta['known']
        np.testing.assert_array_equal(x.groups,dist.groups)
        points=F.read(RAW/d/'community_points.json')['points']
        assigned=F.read(RAW/d/'community_assignments.json')
        observations=F.read(RAW/d/'policy_observations.json')
        cs_est=dict(point=np.array([r['point'] if r['point'] is not None else -1 for r in assigned]),
                    strict=np.array([r['strict'] for r in assigned]),
                    loose=np.array([r['loose'] and not r['strict'] for r in assigned]))
        hz_est=nearest_separated(dist.coords,F.project([p['coordinates_wgs84'] for p in points]),[identity(p) for p in points])
        assert not np.any((hz_est['distance']==0)&(hz_est['second_distance']==0)), 'Ambiguous coincident different communities'
        supplement=ReviewedSupplements(d);paths.extend(map(Path,supplement.hashes))
        negative=rr.negative.copy()
        for q,site in enumerate(meta['sites']):
            for sid in supplement.negative.get(site['coordinate_group_id'],[]):negative[q,dist.school_ids.index(sid)]=True
        F.write(OUT/'metadata'/f'{d}.json',meta)
        dump(OUT/'inputs/current'/f'{d}.pkl',x)
        F.write(OUT/'additions/current'/f'{d}.json',{})
        audit[d]={}
        for method,est in [('changsha',cs_est),('hangzhou',hz_est)]:
            names,strict,cs=community_features(points,est,observations)
            e=patch_features(x,names,strict,cs)
            additions=direct_estimates(cs,strict,rr.known,negative)
            dump(OUT/'inputs'/f'{method}_E'/f'{d}.pkl',e)
            F.write(OUT/'additions'/f'{method}_E'/f'{d}.json',{})
            newdist=deepcopy(dist);newdist.community_name=names;newdist.community_strict=strict;newdist.community_school=cs
            newrr=deepcopy(rr)
            for q,ss in additions.items():newrr.known[q]=ss;newrr.kind[q]='community'
            original=E.learned_boundary
            def fixed_boundary(ls,coords,groups,known,ref_sites,**kwargs):
                return original(ls,coords,groups,rr.known,ref_sites,**kwargs)
            try:
                E.learned_boundary=fixed_boundary
                er=E.build(newdist,newrr,F.read(F.REPO/'configs/default.json')['evidence'])
            finally:E.learned_boundary=original
            # Avoid indirect changes to name features and learned negative labels.
            er.anchor=e.anchor.copy();er.y=np.full(er.n,-1,int);er.supervised=np.empty(0,int)
            original_columns={(int(q),int(s)):k for k,(q,s) in enumerate(zip(x.ref_sites,x.ref_schools))}
            for k,(q,s) in enumerate(zip(er.ref_sites,er.ref_schools)):
                if (int(q),int(s)) in original_columns:
                    oldk=original_columns[int(q),int(s)]
                    for block,sl in x.pair_blocks.items():
                        np.testing.assert_array_equal(er.pair[:,k,sl],e.pair[:,oldk,sl],err_msg=block)
                else:
                    assert q in additions and s in additions[q]
                    er.pair[:,k,er.pair_blocks['named_reference']]=0
            np.testing.assert_array_equal(er.self_mask,er.groups[:,None]==er.groups[er.ref_sites][None])
            dump(OUT/'inputs'/f'{method}_ER'/f'{d}.pkl',er)
            F.write(OUT/'additions'/f'{method}_ER'/f'{d}.json',{str(q):ss for q,ss in additions.items()})
            F.write(OUT/'estimates'/method/f'{d}.json',[
                dict(site_id=dist.site_ids[q],unit=names[q],strict=bool(strict[q]),loose=bool(est['loose'][q]),
                     school_feature_whole=np.flatnonzero(cs[q,:,0:2].any(1)).tolist(),
                     direct_school_estimates=additions.get(q,[])) for q in range(dist.n)])
            audit[d][method]=dict(strict=int(strict.sum()),loose=int(est['loose'].sum()),
                feature_sites=int((cs!=0).any((1,2)).sum()),direct_estimated_sites=len(additions),
                total_reference_schools=len(set(er.ref_schools)),original_reference_schools=len(set(x.ref_schools)),
                pair_community_changed_cells=int(np.count_nonzero(e.pair!=x.pair)),
                school_community_changed_cells=int(np.count_nonzero(e.anchor!=x.anchor)),
                ambiguous_multi_identity_pois=sum(len(p['policy_names'])>1 for p in points))
            print('COMMUNITY_INPUT',d,method,audit[d][method],flush=True)
            del er,e
        paths += [RAW/d/f for f in ['district.pkl','rules.pkl','community_points.json','community_assignments.json','policy_observations.json']]
        paths += [OLD/'inputs'/f'{d}.pkl',OLD/'inputs'/f'{d}.json',ROOT/'round08_reference_retraining/policies'/f'{d}.json',OLD/'source_graphs'/f'{d}.json']
        paths += [BEST/'predictions/round07/balanced_exact'/label/f'{d}.json' for label in ['ensemble']+list(map(str,SEEDS))]
        del x,dist
    F.write(OUT/'input_audit.json',audit)
    code=[Path(__file__),F.REPO/'scripts/shenzhen_community_comparison.py',F.REPO/'scripts/shenzhen_partial_model.py',F.REPO/'scripts/shenzhen_current_supplements.py',F.REPO/'scripts/shenzhen_reviewed_supplements.py']+list((F.REPO/'zoning').glob('*.py'))
    for p in code:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    paths+=code
    F.write(OUT/'input_links.json',dict(files={str(p):F.sha(p) for p in sorted(set(paths))}))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('COMMUNITY_INPUTS_FROZEN',flush=True)


def verify(manifest='input_freeze.json'):
    for rel,h in F.read(OUT/manifest)['files'].items():assert F.sha(OUT/rel)==h,rel
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def model(x,seed):
    net=BernoulliRetrieval(41,27,M.LinearRetrieval(x),seed,32).to(DEVICE)
    ck=torch.load(PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=DEVICE,weights_only=False)
    net.load_state_dict(ck['model']);net.eval();return net


def decoded(ps,d,arm):
    meta=F.read(OUT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
    additions=F.read(OUT/'additions'/arm/f'{d}.json')
    policy=F.read(ROOT/'round08_reference_retraining/policies'/f'{d}.json')
    graph=F.read(OLD/'source_graphs'/f'{d}.json')
    threshold=F.read(OUT/'threshold.json')['value'];rows=[]
    for q,site in enumerate(meta['sites']):
        ss=set(meta['known'].get(str(q),[]))|set(additions.get(str(q),[]))
        decision=predict_sets(np.asarray(ps[q],np.float64),threshold,ids,policy,graph,site,[ids[j] for j in sorted(ss)])
        rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],after=decision['predicted_school_ids'],decision=decision,
                         community_estimate_ids=[ids[j] for j in additions.get(str(q),[])]))
    return ReviewedSupplements(d).apply(rows)


def infer():
    guard();c=verify()
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve frozen predictions')
    for arm in ARMS:
        for d in TARGETS:
            x=pkl(OUT/'inputs'/arm/f'{d}.pkl');values=[]
            for seed in SEEDS:
                p=probabilities(model(x,seed),x,np.arange(x.n),c['chunk']);values.append(p)
                if arm=='current':np.testing.assert_array_equal(p,np.load(PARENT/'probabilities/balanced_rank'/f'{seed}.npz')[d])
                dest=OUT/'probabilities'/arm/d/f'{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,probabilities=p)
                print('INFERRED',arm,d,seed,flush=True)
            for label,ps in [('ensemble',np.stack(values).mean(0))]+list(zip(map(str,SEEDS),values)):
                rows=decoded(ps,d,arm)
                if arm=='current':
                    expected=F.read(BEST/'predictions/round07/balanced_exact'/label/f'{d}.json')
                    assert [(r['site_id'],r['after']) for r in rows]==[(r['site_id'],r['after']) for r in expected],(arm,d,label)
                F.write(OUT/'predictions'/arm/label/f'{d}.json',rows)
            del x,values
    verify()
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','predictions'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    print('ALL FIVE ARMS FROZEN BEFORE SCORING',flush=True)


def replay():
    guard();c=verify();verify('prediction_freeze.json');counts=0
    for arm in ARMS:
        for d in TARGETS:
            x=pkl(OUT/'inputs'/arm/f'{d}.pkl');values=[]
            for seed in SEEDS:
                saved=np.load(OUT/'probabilities'/arm/d/f'{seed}.npz')['probabilities']
                actual=probabilities(model(x,seed),x,np.arange(x.n),c['chunk'])
                np.testing.assert_array_equal(actual,saved);values.append(saved)
            for label,p in [('ensemble',np.stack(values).mean(0))]+list(zip(map(str,SEEDS),values)):
                rows=decoded(p,d,arm);assert rows==F.read(OUT/'predictions'/arm/label/f'{d}.json');counts+=len(rows)
            print('REPLAYED',arm,d,flush=True)
    F.write(OUT/'replay_audit.json',dict(model_runs=100,prediction_rows=counts,all_probabilities_bitwise_equal=True,all_decisions_equal=True,target_gold_read=False))


def metrics(rows):
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),
                exact_accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
                fp=sum(r['fp'] for r in rows),fn=sum(r['fn'] for r in rows))


def score():
    verify();verify('prediction_freeze.json')
    assert F.read(OUT/'replay_audit.json')['all_decisions_equal']
    if (OUT/'results.json').exists():raise FileExistsError('Preserve score')
    # Only this phase reads preserved evaluation labels. No geometry is opened.
    frozen=F.read(ROOT.parent/'shenzhen_aligned_model_20261111/results.json')['gold_sha256']
    for d,h in frozen.items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(ROOT.parent/'shenzhen_aligned_model_20261111/scored_cases.json')}
    xy={}
    for d in TARGETS:
        dist=pkl(RAW/d/'district.pkl');xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    banks={};unscored={}
    for arm in ARMS:
        banks[arm]={};unscored[arm]=[]
        for label in ['ensemble']+list(map(str,SEEDS)):
            rows=[]
            for d in TARGETS:
                pred=F.read(OUT/'predictions'/arm/label/f'{d}.json');assert len(pred)==sum(k[0]==d for k in gold)
                for r in pred:
                    key=d,r['site_id'];g=gold[key]
                    if not g['has_map_gold']:
                        if label=='ensemble':unscored[arm].append(r)
                        continue
                    truth=set(g['gold']);ps=set(r['after']);coord=xy[key]
                    rows.append(dict(d=d,site_id=r['site_id'],name=r['building_name'],gold=sorted(truth),pred=sorted(ps),
                                     correct=ps==truth,fp=len(ps-truth),fn=len(truth-ps),size=len(truth),block=str((int(coord[0]//1000),int(coord[1]//1000)))))
            assert len(rows)==1761;banks[arm][label]=rows
        assert len(unscored[arm])==41
    def groups(rows):
        return {'all':rows,**{d:[r for r in rows if r['d']==d] for d in TARGETS},
                'single':[r for r in rows if r['size']==1],'multi':[r for r in rows if r['size']>1]}
    summary={arm:{label:{g:metrics(rr) for g,rr in groups(rows).items()} for label,rows in v.items()} for arm,v in banks.items()}
    assert summary['current']['ensemble']['all']['correct']==959
    pairs=[(a,'current') for a in ARMS[1:]]+[('hangzhou_E','changsha_E'),('hangzhou_ER','changsha_ER'),('changsha_ER','changsha_E'),('hangzhou_ER','hangzhou_E')]
    comparison={}
    for a,b in pairs:
        detail={}
        for group,rows in groups(banks[a]['ensemble']).items():
            base={(r['d'],r['site_id']):r for r in banks[b]['ensemble']}
            delta=np.array([int(r['correct'])-int(base[r['d'],r['site_id']]['correct']) for r in rows])
            bins=defaultdict(lambda:defaultdict(lambda:[0,0]))
            for r,v in zip(rows,delta):bins[r['d']][r['block']][0]+=int(v);bins[r['d']][r['block']][1]+=1
            rng=np.random.default_rng(20261104);sums=np.zeros(3000);ns=np.zeros(3000)
            for blocks in bins.values():
                vals=np.array(list(blocks.values()));ix=rng.integers(0,len(vals),(3000,len(vals)))
                sums+=vals[ix,0].sum(1);ns+=vals[ix,1].sum(1)
            boot=100*sums/ns
            detail[group]=dict(corrected=int((delta==1).sum()),harmed=int((delta==-1).sum()),delta_pp=float(100*delta.mean()),
                               delta_95_pp=np.percentile(boot,[2.5,97.5]).tolist(),delta_family_98_75_pp=np.percentile(boot,[.625,99.375]).tolist())
        detail['seed_delta']={str(s):summary[a][str(s)]['all']['correct']-summary[b][str(s)]['all']['correct'] for s in SEEDS}
        comparison[a+'_vs_'+b]=detail
    eligible=[];decisions={}
    for a in ARMS[1:]:
        pair=comparison[a+'_vs_current'];checks=dict(positive_family_interval=pair['all']['delta_family_98_75_pp'][0]>0,
            districts_nonnegative=all(pair[d]['delta_pp']>=0 for d in TARGETS),
            single_multi_nonnegative=all(pair[g]['delta_pp']>=0 for g in ['single','multi']),
            seeds_nonnegative=sum(v>=0 for v in pair['seed_delta'].values())>=8)
        decisions[a]=checks
        if all(checks.values()):eligible.append(a)
    selected=max(eligible,key=lambda a:(summary[a]['ensemble']['all']['correct'],a.endswith('_E'),a.startswith('changsha'))) if eligible else 'current'
    result=dict(reports=summary,comparisons=comparison,selection_checks=decisions,selected_method=selected,
                selected_status='local_development_choice_only',independent_validation=False,gold_sha256=frozen,
                original_54_46_reproduced=True,all_arms_scored=True)
    F.write(OUT/'results.json',result)
    diag=OUT/'diagnostics_not_training';F.write(diag/'scored_rows.json',banks);F.write(diag/'unscored_rows.json',unscored)
    # This consumed pointer is used by the replay-selected phase, not an unused claim of deployment.
    F.write(OUT/'selected_method.json',dict(method=selected,experiment=str(OUT.relative_to(F.REPO)),
        checkpoint_parent=str(PARENT.relative_to(F.REPO)),threshold=F.read(OUT/'threshold.json')['value'],
        scope='Shenzhen current frozen model only',independent_validation=False))
    for a in ARMS:print('RESULT',a,summary[a]['ensemble'],flush=True)
    print('SELECTED',selected,decisions,flush=True)


def selected_replay():
    """Runs the selected input strategy with actual checkpoints and checks saved output."""
    choice=F.read(OUT/'selected_method.json');arm=choice['method'];guard();verify()
    counts=0
    for d in TARGETS:
        x=pkl(OUT/'inputs'/arm/f'{d}.pkl')
        values=[probabilities(model(x,s),x,np.arange(x.n),64) for s in SEEDS]
        rows=decoded(np.stack(values).mean(0),d,arm)
        assert rows==F.read(OUT/'predictions'/arm/'ensemble'/f'{d}.json')
        F.write(OUT/'selected_predictions'/f'{d}.json',rows);counts+=len(rows)
    F.write(OUT/'selected_replay_audit.json',dict(method=arm,rows=counts,selected_pipeline_executed=True,all_outputs_equal=True))
    print('SELECTED_PIPELINE_VERIFIED',arm,counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','replay','score','selected']);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'replay':replay,'score':score,'selected':selected_replay}[a.phase]()
