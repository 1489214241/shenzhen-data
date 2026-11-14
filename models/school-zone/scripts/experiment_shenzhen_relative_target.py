"""Frozen target test after the round15 relative decoder passed its source gate."""
import os
os.environ['OMP_NUM_THREADS']='6'
import argparse
from pathlib import Path
from collections import defaultdict
import pickle
import shutil
import numpy as np
import experiment_shenzhen_relative_decode as E
import experiment_shenzhen_conjunction_replay as C
from shenzhen_relation_probe import school_vectors
from experiment_shenzhen_resident_replay import decode

F=E.F;R=E.R;P=E.P
OUT=R.ROOT/'round15_relative_target'
ROAD=R.ROOT/'road_corridor_scope_estimates'
PREVIOUS=R.ROOT/'round13_corridor_supplement'


def verify():
    R.check(OUT,'prediction_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p)==h,p


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen target predictions')
    cfg=E.verify();source=F.read(E.OUT/'source_report.json')
    assert source['reports']['relative']['source_gate_passed']
    links={}
    for folder,manifest in [(P.OUT,'probability_freeze.json'),(C.NEW,'manifest.json'),(ROAD,'manifest.json'),
                            (PREVIOUS,'prediction_freeze.json'),(R.OUT,'input_freeze.json')]:
        R.check(folder,manifest);links[str(folder/manifest)]=F.sha(folder/manifest)
        for rel,h in F.read(folder/manifest)['files'].items(): links[str(folder/rel)]=h
    links.update(F.read(E.OUT/'input_links.json')['files'])
    for path in [E.OUT/'source_report.json',E.OUT/'protocol.json']:
        links[str(path)]=F.sha(path)
    for p,h in F.read(ROAD/'manifest.json')['sources'].items(): assert F.sha(p)==h,p;links[p]=h
    F.write(OUT/'protocol.json',dict(arms=['absolute_control','relative_candidate'],objectives=R.OBJECTIVES,
        source_gate_passed=True,scorer='Frozen round11 pooled_relations, seed20261131, no retraining',
        target_inputs=str(C.NEW),absolute_control='Diagnostic matched decoder control, not a gate-passing candidate.',
        relative_candidate='Only candidate passing unchanged round11 source gate; no Shenzhen outcome selected its threshold.',
        policy='Same original-area university restriction, direct partial positives, scope implications and round07 community facts.',
        corridor='Same11independent road positive estimates for both arms and existing round13 comparators; no transitive expansion.',
        comparators=['matched_absolute','native_latest_round13_round12','best_development_round13_round07'],
        comparison_limit='Only relative versus absolute isolates decoding. Tree versus native changes architecture and training objective; one tree seed versus10seed native ensemble.',
        primary='balanced_exact',target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,
        independent_validation=False,adopted=False,scored_sites=1761,unscored_sites=41))
    with (P.OUT/'models/pooled_relations.pkl').open('rb') as handle:model=pickle.load(handle)
    # Verify copied source scores against the exact classifier before target inference.
    src=dict(np.load(P.OUT/'probabilities/pooled_relations.npz'))
    for d in cfg['sources']:
        a=np.load(P.OUT/'valid'/f'{d}.npz')['features']
        np.testing.assert_array_equal(model.predict_proba(a.reshape(-1,159))[:,1].reshape(a.shape[:2]),src[d+'_valid'])
    source_checks={};all_sites=0
    for d in R.TARGETS:
        with (C.NEW/d/'inputs.pkl').open('rb') as handle:x=pickle.load(handle)
        assert np.array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        assert not len(x.supervised) and (x.y==-1).all()
        meta=F.read(R.OLD/'inputs'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
        assert len(ids)==x.school_n and len(meta['sites'])==x.n
        vectors=school_vectors(x,np.arange(x.n));values=model.predict_proba(vectors.reshape(-1,159))[:,1].reshape(vectors.shape[:2])
        dest=OUT/'probabilities'/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,absolute=values,relative=E.relative(values))
        facts=defaultdict(set)
        for r in F.read(ROAD/d/'estimates.json')['facts']:facts[r['site_id']].add(r['school_id'])
        for arm,key in [('absolute_control','absolute'),('relative_candidate','relative')]:
            p=values if key=='absolute' else E.relative(values)
            for obj in R.OBJECTIVES:
                threshold=source['reports'][key]['selections'][obj]['threshold']
                rows=decode(p,d,threshold);raw=E.select(p,threshold)
                for i,row in enumerate(rows):
                    row['raw_model_set']=[ids[j] for j in np.flatnonzero(raw[i])]
                    row['before_corridor']=row['after']
                    row['corridor_added']=sorted(facts[row['site_id']]-set(row['after']))
                    row['after']=sorted(set(row['after'])|facts[row['site_id']])
                assert len({r['site_id'] for r in rows})==x.n
                F.write(OUT/'predictions'/arm/obj/f'{d}.json',rows)
        source_checks[d]=dict(sites=x.n,schools=x.school_n,query_self_mask_verified=True,no_target_complete_labels=True,
            source_model_replayed_exactly=True,road_estimate_pairs=sum(map(len,facts.values())))
        all_sites+=x.n;print('TARGET_FROZEN',d,x.n,flush=True)
    assert all_sites==1802
    for path in [Path(__file__),F.REPO/'scripts/experiment_shenzhen_resident_replay.py',
                 F.REPO/'scripts/shenzhen_relation_probe.py',F.REPO/'scripts/shenzhen_partial_model.py',
                 F.REPO/'scripts/shenzhen_reference_retraining.py']:
        dest=OUT/'code'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);links[str(path)]=F.sha(path)
    F.write(OUT/'source_and_input_audit.json',source_checks)
    F.write(OUT/'thresholds.json',{arm:source['reports'][arm]['selections'] for arm in ['absolute','relative']})
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('ALL_PREDICTIONS_FROZEN',flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    if (OUT/'results.json').exists():raise FileExistsError('Preserve target outcomes')
    verify();cfg=F.read(OUT/'protocol.json')
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={}
    for d in R.TARGETS:
        with (C.NEW/d/'district.pkl').open('rb') as handle:dist=pickle.load(handle)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={}
    for obj in cfg['objectives']:
        rows=[];unscored=[]
        for d in R.TARGETS:
            paths={'absolute_control':OUT/'predictions/absolute_control'/obj/f'{d}.json',
                   'relative_candidate':OUT/'predictions/relative_candidate'/obj/f'{d}.json',
                   'native_latest':PREVIOUS/'predictions/round12'/obj/'ensemble'/f'{d}.json',
                   'best_development':PREVIOUS/'predictions/round07'/obj/'ensemble'/f'{d}.json'}
            data={k:{r['site_id']:r for r in F.read(p)} for k,p in paths.items()}
            assert all(set(a)==set(data['relative_candidate']) for a in data.values())
            for sid,r in data['relative_candidate'].items():
                key=(d,sid);g=gold[key]
                row=dict(d=d,district=d,site_id=sid,building_name=r['building_name'],
                    **{k:v[sid]['after'] for k,v in data.items()},raw_model_set=r['raw_model_set'],before_corridor=r['before_corridor'])
                if not g['has_map_gold']:unscored.append(row);continue
                row['gold']=set(g['gold']);row['block']=str(tuple(int(v//1000) for v in xy[key]));rows.append(row)
        assert len(rows)==1761 and len(unscored)==41
        summaries={}
        groups=[('all',rows)]+[(d,[r for r in rows if r['d']==d]) for d in R.TARGETS]+[
            ('single',[r for r in rows if len(r['gold'])==1]),('multi',[r for r in rows if len(r['gold'])>1])]
        for label,rr in groups:
            comparison={}
            for control in ['absolute_control','native_latest','best_development']:
                delta=[int(set(r['relative_candidate'])==r['gold'])-int(set(r[control])==r['gold']) for r in rr]
                comparison[control]=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
                    delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
            summaries[label]=dict(metrics={k:metrics(rr,k) for k in ['absolute_control','relative_candidate','native_latest','best_development','raw_model_set','before_corridor']},comparisons=comparison)
        results[obj]=summaries
        F.write(OUT/'diagnostics_not_training'/f'{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
        F.write(OUT/'unscored'/f'{obj}.json',unscored)
        print(obj,summaries['all'],flush=True)
    F.write(OUT/'results.json',dict(reports=results,independent_validation=False,adopted=False,scored=1761,unscored=41,
        gold_hashes_verified=True,predictions_frozen_before_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
