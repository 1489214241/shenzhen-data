"""Round04: fixed prediction intervention from independently reviewed scopes.

All components, models, seeds and populations are declared before scoring.
Only union the independently established positive geographical relations.
No transitive expansion, no full-set claim and no target labels at prepare.
"""
import argparse
import collections
import pickle
import shutil
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'round04_reviewed_scopes'
INPUT=F.REPO/'data/深圳市/局部条件修复输入_20261112'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
BASELINES={'scope_sets':ROOT/'round02_scope_inclusion_v2',
           'masked_bce':ROOT/'round03_partial_labels',
           'assume_negative':ROOT/'round03_partial_labels'}
COMPONENTS={'local_conditions':{'literal_unconditional_name'},
            'street_geography':{'independent_admin_geography'},
            'both':{'literal_unconditional_name','independent_admin_geography'}}
TARGETS=['龙华区','龙岗区']


def add_reviewed_positive_facts(prior,facts,allowed_kinds,school_ids,university_members=()):
    ids=[r['site_id'] for r in prior]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate site identity')
    known=set(ids);schools=set(school_ids);university_members=set(university_members)
    bysite=collections.defaultdict(set)
    for f in facts:
        if f['site_id'] not in known or f['school_id'] not in schools:raise ValueError('Unmatched identity')
        if f['school_id'] in university_members:raise ValueError('University original-base review required')
        if f['evidence_kind'] in allowed_kinds:bysite[f['site_id']].add(f['school_id'])
    out=[]
    for r in prior:
        before=set(r['after'])
        if not before<=schools:raise ValueError('Out-of-roster prediction')
        after=before|bysite[r['site_id']]
        out.append(dict(district=r['district'],site_id=r['site_id'],building_name=r['building_name'],
            before=sorted(before),after=sorted(after),added_school_ids=sorted(after-before),
            changed=after!=before,complete_school_set=False,enrollment_eligibility=False))
    return out


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    for rel,h in F.read(INPUT/'manifest.json')['files'].items():assert F.sha(INPUT/rel)==h,rel
    for folder in set(BASELINES.values()):
        for rel,h in F.read(folder/'prediction_freeze.json')['files'].items():assert F.sha(folder/rel)==h,rel
    c=F.read(BASE/'protocol.json');labels=['ensemble']+[str(s) for s in c['seeds']]
    protocol=dict(round=4,date='2026-11-12',seeds=c['seeds'],labels=labels,baselines={k:str(v) for k,v in BASELINES.items()},
        components={k:sorted(v) for k,v in COMPONENTS.items()},
        hypothesis='Whole-school pending status suppresses independently usable unqualified clauses and explicitly published street scopes.',
        intervention='Union literal unqualified school-name facts and independently estimated administrative membership with frozen predictions. Components tested separately and together.',
        target='geographical school scope relation; not admission eligibility',
        conservative_name_rule='Every name in coordinate group must literally match an unconditional AST named atom; no fuzzy or prefix match.',
        geometry='Independent OSM street boundary, 50m boundary band unknown; no school polygons and no office-point proxy for new facts.',
        qualifications='Household-conditioned clauses remain unknown. Published acceptance types retained. University member expansion prohibited.',
        training=False,recomputed_model_probabilities=False,partial_known_sets_are_not_exhaustive=True,
        target_gold_read=False,target_polygon_read=False,independent_validation=False,adopted=False,
        comparisons='All three components on round02 and both round03 arms, all10seeds plus ensemble. No selection by Shenzhen scores.',
        metrics='Frozen1761map-hit rows; strict exact set, FP/FN, corrected/harmed, districts and single/multi. Paired district-stratified1km bootstrap3000draws.',
        success='No automatic adoption. Stable full-population benefit and independent confirmation still needed.')
    F.write(OUT/'protocol.json',protocol)
    changes={}
    for d in TARGETS:
        meta=F.read(INPUT/d/'model_metadata.json');facts=F.read(INPUT/d/'reviewed_positive_facts.json')
        policy=F.read(INPUT/d/'reviewed_policy.json');sid=[s['school_id'] for s in meta['schools']]
        members={s for r in policy['relations'] if r['kind']=='university' for s in r['school_ids']}
        F.write(OUT/'facts'/f'{d}.json',facts)
        for name,folder in BASELINES.items():
            for label in labels:
                prior=F.read(folder/'predictions'/label/f'{d}.json' if name=='scope_sets' else folder/'predictions'/name/label/f'{d}.json')
                assert [r['site_id'] for r in prior]==[s['coordinate_group_id'] for s in meta['sites']]
                assert all(r['district']==d for r in prior)
                for comp,kinds in COMPONENTS.items():
                    rows=add_reviewed_positive_facts(prior,facts['facts'],kinds,sid,members)
                    F.write(OUT/'predictions'/name/comp/label/f'{d}.json',rows)
                    if label=='ensemble':changes[f'{name}/{comp}/{d}']=dict(changed=sum(r['changed'] for r in rows),
                        new_pairs=sum(len(r['added_school_ids']) for r in rows))
    for src in [Path(__file__),F.REPO/'scripts/build_shenzhen_reviewed_scope_inputs.py',F.REPO/'scripts/shenzhen_qualified_scope.py',
                F.REPO/'tests/test_shenzhen_reviewed_scopes.py',F.REPO/'tests/test_shenzhen_qualified_scope.py',
                F.REPO/'scripts/experiment_shenzhen_set_decode.py',F.REPO/'zoning/evaluate.py']:
        dest=OUT/'code'/src.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    F.write(OUT/'input_audit.json',dict(input_manifest_sha256=F.sha(INPUT/'manifest.json'),
        baseline_prediction_freeze_sha256={str(p):F.sha(p/'prediction_freeze.json') for p in set(BASELINES.values())},
        changes=changes,target_gold_read=False,target_polygons_read=False))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',changes,flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    freeze=F.read(OUT/'prediction_freeze.json')
    for rel,h in freeze['files'].items():assert F.sha(OUT/rel)==h,rel
    gold_hashes=F.read(BASE/'results.json')['gold_sha256']
    for d,h in gold_hashes.items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')}
    xy={}
    for d in TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={};lines=['第四轮：局部户籍条件与明确整街道范围，固定预测实验','',
        '原1761个地图命中样本、学校目录、答案和模型概率不变；新增事实只作为正例下界。开发实验，未采用。','']
    for name in BASELINES:
        results[name]={}
        for comp in COMPONENTS:
            reports={}
            for label in F.read(OUT/'protocol.json')['labels']:
                rows=[];unscored=[]
                for d in TARGETS:
                    for r in F.read(OUT/'predictions'/name/comp/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r['gold']=set(g['gold']);r['semantic_pending']=g['semantic_pending']
                        pt=xy[d,r['site_id']];r['d']=d;r['block']=str((int(pt[0]//1000),int(pt[1]//1000)))
                        rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                reports[label]={}
                for group,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                    ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                    reports[label][group]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),
                        corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
                        delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000),
                        added_true_pairs=sum(len(set(r['added_school_ids'])&r['gold']) for r in rs),
                        added_false_pairs=sum(len(set(r['added_school_ids'])-r['gold']) for r in rs))
                if label=='ensemble':
                    F.write(OUT/'diagnostics_not_training'/f'{name}_{comp}.json',[dict(r,gold=sorted(r['gold'])) for r in rows if r['changed']])
                    F.write(OUT/'unscored'/f'{name}_{comp}.json',unscored)
            results[name][comp]=reports
            lines.append(f'{name}/{comp}')
            for d in ['全部']+TARGETS:
                r=reports['ensemble'][d];a=r['after']
                lines.append(f"{d}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 改正/损害 {r['corrected']}/{r['harmed']}; 误报/漏报 {a['fp']}/{a['fn']}; 变化区间 {r['delta_95_pp']}")
            print(name,comp,reports['ensemble']['全部'],flush=True)
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'verification.json',dict(passed=True,frozen_files=len(freeze['files']),scored_sites=1761,unscored_sites=41,
        seeds=10,baselines=3,components=3,original_gold_unchanged=True,adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);a=p.parse_args()
    dict(prepare=prepare,score=score)[a.phase]()
