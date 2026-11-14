"""Round05: source-only exact threshold search, same frozen round03 models.

Micro exact-set and equally weighted singleton/multiple-set source objectives
are both predeclared. No Shenzhen labels/cardinality select the threshold.
"""
import argparse
import pickle
import shutil
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
from zoning import shenzhen_policy as SP
from shenzhen_partial_model import predict_sets
from shenzhen_exact_calibration import exact_thresholds,threshold_metrics

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'round05_exact_source_calibration'
PREVIOUS=ROOT/'round03_partial_labels'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
INPUT=F.REPO/'data/深圳市/修复实验输入_20261112'
TARGETS=['龙华区','龙岗区']
OBJECTIVES=['micro_exact','balanced_exact']


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen calibration')
    for fname in ['input_freeze.json','probability_freeze.json','prediction_freeze.json']:
        for rel,h in F.read(PREVIOUS/fname)['files'].items():assert F.sha(PREVIOUS/rel)==h,rel
    c=F.read(PREVIOUS/'protocol.json');labels=['ensemble']+[str(s) for s in c['seeds']]
    F.write(OUT/'protocol.json',dict(round=5,date='2026-11-12',arms=c['arms'],objectives=OBJECTIVES,
        seeds=c['seeds'],labels=labels,sources=c['sources'],targets=TARGETS,previous=str(PREVIOUS),
        hypothesis='Round03 calibration optimum hit0.90grid ceiling; exact full-range calibration tests whether that ceiling or singleton-dominated calibration explains failures.',
        search='All exact-set prediction intervals over threshold[0,1], top1fallback; threshold comparison in float64 to preserve strict endpoints.',
        objectives_definition=dict(micro_exact='maximize source complete exact sets',
            balanced_exact='maximize mean of singleton and multi-label source exact accuracies with equal0.5weights; no target cardinality or priors'),
        tie_break='minimumFPthenhighestthreshold; FP is nonincreasing under threshold+top1fallback',
        unchanged='All20models, probabilities, source validation rows, target inputs, policy decoding and partial positive lower bounds unchanged.',
        no_retraining=True,target_gold_read=False,target_cardinality_used=False,independent_validation=False,adopted=False,
        metrics='Same1761map-hit rows, both districts, single/multi,FP/FN,corrected/harmed,3000spatialpairedbootstrap; all10seeds retained.',
        comparison='Each threshold against same round03 model arm. Round04 geographical augmentation not combined. Both objectives reported without Shenzhen-based selection.'))
    ys={d:np.load(PREVIOUS/'inputs'/f'{d}_labels.npz')['valid_labels'] for d in c['sources']}
    audit={}
    for arm in c['arms']:
        loaded=[dict(np.load(PREVIOUS/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in c['sources']+TARGETS}
        cal=exact_thresholds([mean[d] for d in c['sources']],[ys[d] for d in c['sources']])
        old_t=F.read(PREVIOUS/'source_calibration.json')[arm]['threshold']
        cal['previous_threshold']=old_t
        cal['previous_source_metrics']=threshold_metrics([mean[d] for d in c['sources']],[ys[d] for d in c['sources']],old_t)
        F.write(OUT/'source_calibration'/f'{arm}.json',cal)
        audit[arm]=dict(selections=cal['selections'],previous=cal['previous_source_metrics'],
            attainable_rows=cal['attainable_rows'],source_single=cal['source_single'],source_multi=cal['source_multi'])
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],loaded)]:
            for d in TARGETS:
                meta=F.read(PREVIOUS/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
                graph=F.read(PREVIOUS/'source_graphs'/f'{d}.json');policy=SP.load_policy(d)
                prior=F.read(PREVIOUS/'predictions'/arm/label/f'{d}.json')
                output={o:[] for o in OBJECTIVES}
                for q,site in enumerate(meta['sites']):
                    assert prior[q]['site_id']==site['coordinate_group_id']
                    positive=[sid[j] for j in meta['known'].get(str(q),[])]
                    old=predict_sets(values[d][q],old_t,sid,policy,graph,site,positive)
                    assert old['predicted_school_ids']==prior[q]['after'],(arm,label,d,q)
                    for objective in OBJECTIVES:
                        threshold=cal['selections'][objective]['threshold']
                        r=predict_sets(np.asarray(values[d][q],np.float64),threshold,sid,policy,graph,site,positive)
                        output[objective].append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],
                            before=prior[q]['after'],after=r['predicted_school_ids'],decision=r))
                for objective,rows in output.items():F.write(OUT/'predictions'/arm/objective/label/f'{d}.json',rows)
        print('SOURCE CALIBRATED',arm,audit[arm],flush=True)
    F.write(OUT/'input_audit.json',dict(source_only=True,previous_prediction_parity=True,arms=audit,
        old_freeze_sha256={f:F.sha(PREVIOUS/f) for f in ['input_freeze.json','probability_freeze.json','prediction_freeze.json']}))
    for src in [Path(__file__),F.REPO/'scripts/shenzhen_exact_calibration.py',F.REPO/'scripts/shenzhen_partial_model.py',
                F.REPO/'tests/test_shenzhen_exact_calibration.py',F.REPO/'scripts/experiment_shenzhen_set_decode.py',F.REPO/'zoning/evaluate.py']:
        dest=OUT/'code'/src.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        target_gold_read=False,target_cardinality_used=False,previous_prediction_parity=True))
    print('FROZEN',flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    freeze=F.read(OUT/'prediction_freeze.json')
    for rel,h in freeze['files'].items():assert F.sha(OUT/rel)==h,rel
    for d,h in F.read(BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')};xy={}
    for d in TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    c=F.read(OUT/'protocol.json');results={}
    lines=['第五轮：仅来源区完整阈值校准，固定第三轮模型','',
        '全范围阈值与单校/多校等权目标均只使用长沙、杭州留出集；深圳答案不参与选阈值。开发实验，未采用。','']
    for arm in c['arms']:
        results[arm]={}
        for obj in OBJECTIVES:
            reports={}
            for label in c['labels']:
                rows=[];unscored=[]
                for d in TARGETS:
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r['gold']=set(g['gold']);r['semantic_pending']=g['semantic_pending']
                        pt=xy[d,r['site_id']];r['d']=d;r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                report={}
                for name,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                    ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                    report[name]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),corrected=delta.count(1),harmed=delta.count(-1),
                        delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
                reports[label]=report
                if label=='ensemble':F.write(OUT/'diagnostics_not_training'/f'{arm}_{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
            results[arm][obj]=reports;lines.append(arm+'/'+obj)
            for d in ['全部']+TARGETS+['单校','多校']:
                r=reports['ensemble'][d];a=r['after']
                lines.append(f"{d}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 改正/损害 {r['corrected']}/{r['harmed']}; 误报/漏报 {a['fp']}/{a['fn']}; 变化区间 {r['delta_95_pp']}")
            print(arm,obj,reports['ensemble']['全部'],flush=True)
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'verification.json',dict(passed=True,frozen_files=len(freeze['files']),arms=2,objectives=2,seeds=10,
        unchanged_gold=True,scored_sites=1761,unscored_sites=41,source_only_calibration=True,previous_prediction_parity=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);a=p.parse_args()
    dict(prepare=prepare,score=score)[a.phase]()
