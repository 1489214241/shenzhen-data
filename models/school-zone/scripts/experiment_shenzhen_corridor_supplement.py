"""Fixed independent-road estimate supplement, without retraining or threshold tuning."""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment_shenzhen_reference_retraining as R

F=R.F
ROOT=R.ROOT
SOURCE=ROOT/'road_corridor_scope_estimates'
OUT=ROOT/'round13_corridor_supplement'
PARENTS={'round07':ROOT/'round07_community_facts','round12':ROOT/'round12_conjunction_replay'}


def parent_predictions(arm,obj,label,d):
    path=PARENTS[arm]/'predictions'
    if arm=='round07':path=path/'balanced_rank'
    return path/obj/label/f'{d}.json'


def verify():
    R.check(OUT,'prediction_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve predictions')
    links={}
    for folder,manifest in [(SOURCE,'manifest.json')]+[(p,'prediction_freeze.json') for p in PARENTS.values()]:
        R.check(folder,manifest);links[str(folder/manifest)]=F.sha(folder/manifest)
        for rel,h in F.read(folder/manifest)['files'].items():links[str(folder/rel)]=h
    for p,h in F.read(SOURCE/'manifest.json')['sources'].items():assert F.sha(p)==h,p;links[p]=h
    cfg=F.read(PARENTS['round12']/'protocol.json');labels=['ensemble']+[str(s) for s in cfg['seeds']]
    F.write(OUT/'protocol.json',dict(arms=list(PARENTS),objectives=R.OBJECTIVES,labels=labels,
        intervention='Union only11independent road-corridor positive estimates, without transitive policy expansion.',
        radii_m=[10,25,50],boundary_margin_m=50,source_estimates_are_not_confirmed_enrollment=True,
        training_steps=0,changed_model_inputs=False,thresholds_unchanged=True,target_gold_read=False,
        target_polygon_read=False,independent_validation=False,adopted=False,
        comparators='Frozen best-development round07 and latest cumulative input repair round12; both declared before scoring.'))
    changes={}
    for d in R.TARGETS:
        source=F.read(SOURCE/d/'estimates.json');facts=defaultdict(set)
        for r in source['facts']:facts[r['site_id']].add(r['school_id'])
        F.write(OUT/'estimates'/f'{d}.json',source)
        for arm in PARENTS:
            for obj in R.OBJECTIVES:
                for label in labels:
                    before=F.read(parent_predictions(arm,obj,label,d));rows=[]
                    assert set(facts)<={r['site_id'] for r in before}
                    for r in before:
                        old=set(r['after']);extra=facts[r['site_id']]-old
                        rows.append(dict(district=d,site_id=r['site_id'],building_name=r['building_name'],
                            before=sorted(old),after=sorted(old|extra),added_school_ids=sorted(extra),
                            complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    if label=='ensemble':changes[f'{arm}/{obj}/{d}']=dict(changed_sites=sum(bool(r['added_school_ids']) for r in rows),added_pairs=sum(len(r['added_school_ids']) for r in rows))
    for p in [Path(__file__),F.REPO/'scripts/build_shenzhen_corridor_scope_estimates.py',F.REPO/'tests/test_shenzhen_corridor_estimates.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links));F.write(OUT/'source_only_changes.json',changes)
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',changes,flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scored experiment')
    verify();cfg=F.read(OUT/'protocol.json')
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as handle:dist=pickle.load(handle)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={}
    for arm in cfg['arms']:
        results[arm]={}
        for obj in cfg['objectives']:
            results[arm][obj]={}
            for label in cfg['labels']:
                rows=[];unscored=[]
                for d in R.TARGETS:
                    parent={r['site_id']:r for r in F.read(parent_predictions(arm,obj,label,d))}
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        assert r['before']==sorted(parent[r['site_id']]['after'])
                        key=(d,r['site_id']);g=gold[key]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r['gold']=set(g['gold']);r['d']=d;r['block']=str(tuple(int(v//1000) for v in xy[key]));rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                report={}
                for group,rs in [('all',rows)]+[(d,[r for r in rows if r['district']==d]) for d in R.TARGETS]+[
                    ('single',[r for r in rows if len(r['gold'])==1]),('multi',[r for r in rows if len(r['gold'])>1])]:
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                    report[group]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),corrected=delta.count(1),harmed=delta.count(-1),
                        delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
                results[arm][obj][label]=report
                if label=='ensemble':
                    F.write(OUT/'diagnostics_not_training'/f'{arm}_{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows if r['added_school_ids']])
                    F.write(OUT/'unscored'/f'{arm}_{obj}.json',unscored)
            print(arm,obj,results[arm][obj]['ensemble']['all'],flush=True)
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    F.write(OUT/'completion_integrity.json',dict(prediction_files=88,all_sites=1802,scored=1761,unscored=41,
        gold_hashes_verified=True,control_predictions_verified=True,source_estimate_pairs=11,models_retrained=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
