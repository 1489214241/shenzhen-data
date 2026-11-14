"""Independent replay and diagnostic-only ranking/rule-stage audit for round15."""
import os
os.environ['OMP_NUM_THREADS']='6'
import pickle
from collections import defaultdict
import numpy as np
import experiment_shenzhen_relative_target as T
from shenzhen_relation_probe import school_vectors
from experiment_shenzhen_resident_replay import decode

F=T.F;R=T.R;E=T.E;P=T.P


def count(rows,key):
    return dict(n=len(rows),correct=sum(set(r[key])==set(r['gold']) for r in rows),
        fp=sum(len(set(r[key])-set(r['gold'])) for r in rows),
        fn=sum(len(set(r['gold'])-set(r[key])) for r in rows))


def transitions(rows,a,b):
    return dict(corrected=sum(set(r[a])!=set(r['gold']) and set(r[b])==set(r['gold']) for r in rows),
        harmed=sum(set(r[a])==set(r['gold']) and set(r[b])!=set(r['gold']) for r in rows),
        added_relations=sum(len(set(r[b])-set(r[a])) for r in rows),
        removed_relations=sum(len(set(r[a])-set(r[b])) for r in rows))


def input_replay():
    # Separate read-guarded process: this phase never reads target labels.
    F.guard();T.verify();E.verify()
    with (P.OUT/'models/pooled_relations.pkl').open('rb') as handle:model=pickle.load(handle)
    report=F.read(E.OUT/'source_report.json')['reports'];n=0
    for d in R.TARGETS:
        with (T.C.NEW/d/'inputs.pkl').open('rb') as handle:x=pickle.load(handle)
        a=school_vectors(x,np.arange(x.n));p=model.predict_proba(a.reshape(-1,159))[:,1].reshape(a.shape[:2])
        stored=dict(np.load(T.OUT/'probabilities'/f'{d}.npz'))
        np.testing.assert_array_equal(p,stored['absolute']);np.testing.assert_array_equal(E.relative(p),stored['relative'])
        meta=F.read(R.OLD/'inputs'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
        facts=defaultdict(set)
        for f in F.read(T.ROAD/d/'estimates.json')['facts']:facts[f['site_id']].add(f['school_id'])
        for arm,key in [('absolute_control','absolute'),('relative_candidate','relative')]:
            for obj in R.OBJECTIVES:
                values=stored[key];threshold=report[key]['selections'][obj]['threshold']
                expected=F.read(T.OUT/'predictions'/arm/obj/f'{d}.json');actual=decode(values,d,threshold)
                raw=E.select(values,threshold)
                for i,(old,new) in enumerate(zip(expected,actual)):
                    assert old['site_id']==new['site_id'] and old['decision']==new['decision']
                    assert old['before_corridor']==new['after']
                    assert set(old['after'])==set(new['after'])|facts[old['site_id']]
                    assert old['raw_model_set']==[ids[j] for j in np.flatnonzero(raw[i])]
                assert len(expected)==len(actual)==x.n;n+=len(actual)
    assert n==7208
    F.write(T.OUT/'prediction_replay_audit.json',dict(prediction_rows=n,prediction_files=8,
        probabilities_bitwise_reproduced=True,all_policy_and_corridor_outputs_reproduced=True,
        target_labels_read=False,model_and_dependency_hashes_verified=True))
    print('INPUT_REPLAY_PASSED',n,flush=True)


def score_audit():
    T.verify()
    assert F.read(T.OUT/'prediction_replay_audit.json')['target_labels_read'] is False
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')}
    results=F.read(T.OUT/'results.json')['reports'];diagnostics={};stage_rows=[]
    for obj in R.OBJECTIVES:
        rows=F.read(T.OUT/'diagnostics_not_training'/f'{obj}.json')
        assert len(rows)==1761 and len(F.read(T.OUT/'unscored'/f'{obj}.json'))==41
        assert len({(r['district'],r['site_id']) for r in rows})==1761
        for r in rows:assert r['gold']==sorted(gold[(r['district'],r['site_id'])]['gold'])
        for label,rr in [('all',rows),('single',[r for r in rows if len(r['gold'])==1]),('multi',[r for r in rows if len(r['gold'])>1])]+[(d,[r for r in rows if r['district']==d]) for d in R.TARGETS]:
            for key,claimed in results[obj][label]['metrics'].items():
                actual=count(rr,key)
                assert all(claimed[k]==v for k,v in actual.items()),(obj,label,key)
            for control,claimed in results[obj][label]['comparisons'].items():
                actual=transitions(rr,control,'relative_candidate')
                assert actual['corrected']==claimed['corrected'] and actual['harmed']==claimed['harmed']
    cfg=F.read(R.OUT/'protocol.json');native=[dict(np.load(T.C.OUT/'probabilities'/f'{s}.npz')) for s in cfg['seeds']]
    for d in R.TARGETS:
        meta=F.read(R.OLD/'inputs'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
        probs={'tree':np.load(T.OUT/'probabilities'/f'{d}.npz')['absolute'],
               'native':np.stack([p[d] for p in native]).mean(0)}
        pred=F.read(T.OUT/'predictions/relative_candidate/balanced_exact'/f'{d}.json')
        for j,r in enumerate(pred):
            g=gold[(d,r['site_id'])]
            if not g['has_map_gold']:continue
            truth=set(g['gold']);mask=np.array([s in truth for s in ids]);info={}
            for family,p in probs.items():
                v=p[j];other=float(v[~mask].max()) if (~mask).any() else -1.
                info[family]=dict(rank_separable=bool(v[mask].min()>other),top1_correct=bool(mask[v.argmax()]),
                    min_gold_score=float(v[mask].min()),max_other_score=other)
            stage_rows.append(dict(d=d,site_id=r['site_id'],building_name=r['building_name'],gold=sorted(truth),
                raw=r['raw_model_set'],base=r['decision']['raw_school_ids'],
                policy=r['decision']['predicted_school_ids'],community=r['before_corridor'],final=r['after'],ranking=info))
    for group,rr in [('all',stage_rows),('single',[r for r in stage_rows if len(r['gold'])==1]),('multi',[r for r in stage_rows if len(r['gold'])>1])]+[(d,[r for r in stage_rows if r['d']==d]) for d in R.TARGETS]:
        diagnostics[group]=dict(n=len(rr),stages={k:count(rr,k) for k in ['raw','base','policy','community','final']},
            transitions={a+'->'+b:transitions(rr,a,b) for a,b in zip(['raw','base','policy','community'],['base','policy','community','final'])},
            ranking={family:dict(rank_separable=sum(r['ranking'][family]['rank_separable'] for r in rr),
                top1_correct=sum(r['ranking'][family]['top1_correct'] for r in rr)) for family in ['native','tree']})
    F.write(T.OUT/'diagnostics_not_training/ranking_and_stage_rows.json',stage_rows)
    F.write(T.OUT/'diagnostics_not_training/ranking_and_stages.json',diagnostics)
    F.write(R.ROOT/'round15_independent_completion_audit.json',dict(prediction_replay=True,gold_hashes_unchanged=True,
        all_scoring_groups_recomputed=True,source_gate_passed=True,scored=1761,unscored=41,
        target_tuning=False,adopted=False,independent_validation=False,diagnostics=diagnostics,
        limitation='Rank-separability is an answer-dependent diagnostic upper bound, never a deployable threshold or training signal. Stages are descriptive, not new candidate rule removals.'))
    print('SCORE_AUDIT_PASSED',diagnostics['all'],diagnostics['multi'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['input_replay','score_audit']);globals()[p.parse_args().phase]()
