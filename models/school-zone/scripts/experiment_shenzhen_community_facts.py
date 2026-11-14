"""Round07: fixed-output intervention from independently reviewed membership.

This first isolates whether source membership contributes correct missing
school relations; it does NOT claim a retrained model or changed references.
Every round06 arm/calibration/seed is included, with fixed original scoring.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_community_facts import MembershipScopeProof

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCES=ROOT/'community_membership_sources'
REVIEW=SOURCES/'review_v1'
PARENT=ROOT/'round06_balanced_ranking'
INPUT=F.REPO/'data/深圳市/局部条件修复输入_20261112'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
OUT=ROOT/'round07_community_facts'
ARMS=['continued_bce','balanced_bce','balanced_rank']
OBJECTIVES=['micro_exact','balanced_exact']
QUEUE_SHA='d2aade7b0f247fbaedd79e302087ea3cf00b4e86b82f5b7ee686d206c8bb3726'


def check(folder,filename):
    freeze=F.read(folder/filename)
    for rel,h in freeze['files'].items():assert F.sha(folder/rel)==h,(folder,rel)


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    assert F.sha(SOURCES/'literal_hit_review_queue.json')==QUEUE_SHA
    check(SOURCES,'inventory_manifest.json');check(REVIEW,'manifest.json')
    assert F.sha(F.REPO/'scripts/review_shenzhen_community_membership.py')==F.read(REVIEW/'manifest.json')['code_sha256']
    check(INPUT,'manifest.json');check(PARENT,'prediction_freeze.json')
    profiles=F.read(SOURCES/'profile_inventory.json')['profiles']
    members=F.read(REVIEW/'membership_facts.json')
    c=F.read(PARENT/'protocol.json');labels=['ensemble']+[str(s) for s in c['seeds']]
    protocol=dict(round=7,date='2026-11-12',seeds=c['seeds'],labels=labels,arms=ARMS,objectives=OBJECTIVES,
        hypothesis='Independent official residential membership can supply missing applicability facts for community-based school scopes.',
        intervention='Union source-proved positive school relations with each unchanged round06 final prediction; no transitive relation expansion.',
        training=False,recomputed_model_probabilities=False,changed_model_references=False,
        temporal_cutoff='2026-06-01',source_age_limitation='Archived official profile publication predates cutoff; unchanged2026boundary not independently guaranteed.',
        identity='Every coordinate-group name supported by reviewed clause; homonyms across communities held; no fuzzy aliases or subcommittee collapse.',
        proof='3-valued AST: exact community/street identity and explicit source range references; unresolved roads, exclusions, household conditions remain unknown.',
        all_schools=202,target_gold_read=False,target_polygon_read=False,independent_validation=False,adopted=False,
        target='Geographical scope membership, not admission eligibility; listed positives do not constitute a complete answer set.',
        comparison='All3arms x2source-calibration objectives x10seeds plus ensemble; same1761scored and41unscored.',
        metrics='Exact-set,FP,FN,corrected,harmed,1km paired district-stratified bootstrap3000 draws; both districts,single,multi and semantic-pending.',
        adoption='No automatic adoption or target-score selection; independent validation and full input/training experiment still required.')
    F.write(OUT/'protocol.json',protocol)
    changes={};coverage={}
    for d in F.SZ.DISTRICTS:
        policy=F.read(INPUT/d/'reviewed_policy.json');inv=policy['school_scope_inventory']
        parser=QualifiedParser(inv,parse_scope);asts=[parser.parse(r['base_scope_text']) for r in inv]
        local=[f for f in members['facts'] if f['district']==d]
        proof=MembershipScopeProof(inv,asts,profiles,d)
        facts=proof.derive(local)
        assert len({(f['site_id'],f['school_id']) for f in facts})==len(facts)
        F.write(OUT/'facts'/f'{d}.json',dict(facts=facts,target_gold_read=False,target_polygon_read=False,
            all_school_ast_audit=[dict(school_id=r['school_id'],school_name=r['school_name'],ast=a,
               source_text=r['base_scope_text'],evidence=r['evidence']) for r,a in zip(inv,asts)]))
        bysite=defaultdict(set)
        for f in facts:bysite[f['site_id']].add(f['school_id'])
        coverage[d]=dict(reviewed_membership_sites=len(local),school_facts=len(facts),
            applicable_sites=len(bysite),supported_schools=len({f['school_id'] for f in facts}),
            reviewed_schools=len(inv),membership_sites_without_proof=len(local)-len(bysite))
        meta=F.read(INPUT/d/'model_metadata.json');school_ids={s['school_id'] for s in meta['schools']}
        site_ids=[s['coordinate_group_id'] for s in meta['sites']]
        assert set(bysite)<=set(site_ids)
        assert all(ss<=school_ids for ss in bysite.values())
        for arm in ARMS:
            for obj in OBJECTIVES:
                for label in labels:
                    prior=F.read(PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    assert [r['site_id'] for r in prior]==site_ids
                    rows=[]
                    for r in prior:
                        before=set(r['after']);added=bysite[r['site_id']]-before
                        rows.append(dict(district=d,site_id=r['site_id'],building_name=r['building_name'],
                            before=sorted(before),after=sorted(before|added),added_school_ids=sorted(added),
                            changed=bool(added),complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    if label=='ensemble':changes[f'{arm}/{obj}/{d}']=dict(changed_sites=sum(r['changed'] for r in rows),
                        added_pairs=sum(len(r['added_school_ids']) for r in rows))
    F.write(OUT/'source_coverage.json',dict(coverage=coverage,changes=changes,source_membership_summary=members['summary']))
    sources=[Path(__file__),F.REPO/'scripts/shenzhen_community_facts.py',F.REPO/'scripts/review_shenzhen_community_membership.py',
        F.REPO/'scripts/shenzhen_qualified_scope.py',F.REPO/'scripts/shenzhen_aligned_policy.py',
        F.REPO/'scripts/experiment_shenzhen_set_decode.py',F.REPO/'zoning/evaluate.py',
        F.REPO/'tests/test_shenzhen_community_facts.py']
    for p in sources:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'source_hashes.json',dict(review_manifest=F.sha(REVIEW/'manifest.json'),
        inventory_manifest=F.sha(SOURCES/'inventory_manifest.json'),
        parent_predictions=F.sha(PARENT/'prediction_freeze.json'),
        policy_inputs=F.sha(INPUT/'manifest.json')))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        target_gold_read=False,target_polygon_read=False))
    print('FROZEN',coverage,changes,flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    check(OUT,'prediction_freeze.json')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scored result')
    for d,h in F.read(BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')}
    xy={}
    for d in F.SZ.DISTRICTS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    reports={};lines=['第七轮：官方社区归属事实，固定模型输出补充实验',
        '同一1761个原图答案、41个图外点不评分；不改模型、参照、概率、校准或答案。',
        '来源213条社区归属关系；后期资料及社区同名冲突暂缓。旧资料不能保证2026边界未变。',
        '仅为开发证据；未采用，不能称重训成功或已解决。','']
    for arm in ARMS:
        reports[arm]={}
        for obj in OBJECTIVES:
            reports[arm][obj]={}
            for label in F.read(OUT/'protocol.json')['labels']:
                rows=[];unscored=[]
                for d in F.SZ.DISTRICTS:
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r['gold']=set(g['gold']);r['semantic_pending']=g['semantic_pending'];r['d']=d
                        pt=xy[d,r['site_id']];r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                report={}
                groups=[('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in F.SZ.DISTRICTS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                    ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]
                for name,rs in groups:
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                    report[name]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),
                        corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
                        delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000),
                        added_true_pairs=sum(len(set(r['added_school_ids'])&r['gold']) for r in rs),
                        added_false_pairs=sum(len(set(r['added_school_ids'])-r['gold']) for r in rs))
                reports[arm][obj][label]=report
                if label=='ensemble':
                    F.write(OUT/'diagnostics_not_training'/f'{arm}_{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows if r['changed']])
                    F.write(OUT/'unscored'/f'{arm}_{obj}.json',unscored)
            lines.append(arm+'/'+obj)
            for name in ['全部']+list(F.SZ.DISTRICTS)+['单校','多校']:
                r=reports[arm][obj]['ensemble'][name];a=r['after']
                lines.append(f"{name}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 改正/损害{r['corrected']}/{r['harmed']}; 误报/漏报{a['fp']}/{a['fn']}; 95%变化区间{r['delta_95_pp']}")
            print(arm,obj,reports[arm][obj]['ensemble']['全部'],flush=True)
    F.write(OUT/'results.json',dict(reports=reports,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'completion_integrity.json',dict(frozen_files=len(F.read(OUT/'prediction_freeze.json')['files']),
        prediction_files=132,seeds=10,arms=3,objectives=2,scored_sites=1761,unscored_sites=41,
        original_gold_unchanged=True,target_labels_not_used_to_prepare=True,adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);a=p.parse_args()
    dict(prepare=prepare,score=score)[a.phase]()
