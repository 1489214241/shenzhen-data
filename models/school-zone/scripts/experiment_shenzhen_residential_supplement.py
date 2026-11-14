"""Round20: fixed-prediction intervention from new official community facts.

Pure street-wide recruitment remains a separate eligibility clue. Only a proof
with an explicit community-membership witness is a geographical supplement.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment_shenzhen_reference_retraining as R

F=R.F;ROOT=R.ROOT
SOURCE=ROOT/'municipal_residential_membership_sources/review_v1'
PARENT=ROOT/'round13_corridor_supplement'
OUT=ROOT/'round20_residential_membership'
ARMS=['round07','round12']


def parent_path(arm,obj,label,d):
    return PARENT/'predictions'/arm/obj/label/f'{d}.json'


def verify():
    R.check(OUT,'prediction_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen intervention')
    R.check(SOURCE,'manifest.json');R.check(PARENT,'prediction_freeze.json')
    links={}
    for folder,manifest in [(SOURCE,'manifest.json'),(PARENT,'prediction_freeze.json')]:
        links[str(folder/manifest)]=F.sha(folder/manifest)
        links.update({str(folder/rel):h for rel,h in F.read(folder/manifest)['files'].items()})
    for p,h in F.read(SOURCE/'manifest.json')['sources'].items():assert F.sha(p)==h,p;links[p]=h
    c=F.read(PARENT/'protocol.json');labels=c['labels']
    F.write(OUT/'protocol.json',dict(round=20,arms=ARMS,objectives=R.OBJECTIVES,labels=labels,
        hypothesis='New exact residential-community identities fill missing community-conditioned scope facts.',
        intervention='Union only explicit community-witness relations; pure street-wide recruitment retained as pending eligibility, not original catchment.',
        inputs='14 new dated memberships from two official residential notices; exact entity names only.',
        training_steps=0,changed_model_probabilities=False,changed_model_references=False,thresholds_unchanged=True,
        target_gold_read=False,target_polygons_read=False,temporal_cutoff='2026-06-01',source_age_limitation=True,
        complete_school_set=False,enrollment_eligibility=False,unobserved_relations_are_negative=False,
        primary_comparator='round07+round13 road supplement, prior best development candidate; fixed before scoring.',
        secondary_comparator='round12+round13 cumulative parser/road repair; fixed before scoring.',
        independent_validation=False,adopted=False))
    coverage={};changes={}
    for d in R.TARGETS:
        source=F.read(SOURCE/'school_facts'/f'{d}.json');accepted=[];pending=[]
        for fact in source['facts']:
            if any(w['kind']=='official_community_membership' for w in fact['proof']):accepted.append(fact)
            else:pending.append(dict(**fact,reason='Street-wide application scope does not independently establish original geographical catchment.'))
        bysite=defaultdict(set)
        for f in accepted:bysite[f['site_id']].add(f['school_id'])
        assert sum(map(len,bysite.values()))==len(accepted)
        F.write(OUT/'facts'/f'{d}.json',dict(facts=accepted,pending_eligibility_only=pending,target_gold_read=False))
        coverage[d]=dict(accepted_school_relations=len(accepted),applicable_sites=len(bysite),held_street_only=len(pending))
        for arm in ARMS:
            for obj in R.OBJECTIVES:
                for label in labels:
                    prior=F.read(parent_path(arm,obj,label,d));rows=[]
                    assert set(bysite)<={r['site_id'] for r in prior}
                    for r in prior:
                        old=set(r['after']);extra=bysite[r['site_id']]-old
                        rows.append(dict(district=d,site_id=r['site_id'],building_name=r['building_name'],
                            before=sorted(old),after=sorted(old|extra),added_school_ids=sorted(extra),
                            complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    if label=='ensemble':changes[f'{arm}/{obj}/{d}']=dict(changed_sites=sum(bool(r['added_school_ids']) for r in rows),added_pairs=sum(len(r['added_school_ids']) for r in rows))
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_community_facts.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'source_only_changes.json',dict(coverage=coverage,changes=changes))
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',coverage,changes,flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scored experiment')
    verify();cfg=F.read(OUT/'protocol.json')
    gold_hashes=F.read(R.BASE/'results.json')['gold_sha256']
    for d,h in gold_hashes.items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={}
    for arm in cfg['arms']:
        results[arm]={}
        for obj in cfg['objectives']:
            results[arm][obj]={}
            for label in cfg['labels']:
                rows=[];unscored=[]
                for d in R.TARGETS:
                    prior={r['site_id']:r for r in F.read(parent_path(arm,obj,label,d))}
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        assert r['before']==sorted(prior[r['site_id']]['after'])
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
    F.write(OUT/'results.json',dict(reports=results,gold_sha256=gold_hashes,adopted=False,independent_validation=False))
    print('COMPLETE no automatic adoption',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
