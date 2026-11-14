"""Native full-feature Shenzhen external test; freeze before target scoring."""
from pathlib import Path
import argparse, json, pickle, shutil, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_shenzhen_current as B
import shenzhen_full_inputs as F
import numpy as np
from zoning import data,rules,evidence as E,train as T

REPO=B.REPO
B.OUT=REPO/'outputs/shenzhen_full_model_20261111'
OUT=B.OUT
CACHE=REPO/'outputs/shenzhen_full_source_cache_20261111'


def source_inputs(d,cfg):
    cache=CACHE/f'{d}.pkl'
    if cache.exists():
        with cache.open('rb') as f:return pickle.load(f)
    dist=data.with_mca_community(data.with_list_gold(data.load(d)));rc=cfg['rules']
    r=rules.run(dist,rc['preset'],rc['hangzhou_names'],rc['hangzhou_boundary'],rc['hangzhou_community_rule'])
    x=E.build(dist,r,cfg['evidence']);lab=T.uncovered_answers(dist,x)
    ids=sorted(set(x.supervised.tolist())|set(lab));y=x.y.copy()
    for q,s in lab.items():y[q]=s
    entry=(x,dict(ids=ids,labels=y[ids].tolist()))
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('wb') as f:pickle.dump(entry,f,protocol=5)
    return entry


def sources():
    B.guard();cfg=B.read(REPO/'configs/default.json')
    for d in cfg['districts']:
        x,pop=source_inputs(d,cfg);B.log('SOURCE_CACHE',d,x.pair.shape,x.anchor.shape,len(pop['ids']))
    B.write(CACHE/'manifest.json',dict(config=cfg,files={p.name:B.sha(p) for p in CACHE.glob('*.pkl')},target_labels_read=False))


def prepare():
    B.guard()
    if OUT.exists():raise FileExistsError('Keep prior frozen run; directory already exists')
    cfg=B.read(REPO/'configs/default.json')
    protocol=dict(status='frozen_before_model_training_and_target_scoring',model='zoning.model.RetrievalScorer',
        targets=B.TARGETS,sources=cfg['districts'],seeds=cfg['train']['seeds'],steps=cfg['train']['steps'],
        lr=cfg['train']['lr'],regularization=cfg['train']['reg']['default'],width=cfg['model']['width'],
        chunk=64,warm_start_steps=100,source_config=cfg,additional_target_gold_references=0,
        pair_blocks=cfg['evidence']['pair_blocks'],school_blocks=cfg['evidence']['school_blocks'],
        omitted_in_both_sources_and_targets=[],input_dataset=str(F.OUT.relative_to(REPO)),
        input_manifest_sha256=B.sha(F.OUT/'manifest.json'),
        baseline='same native model probabilities with direct/rule-known complete sets retained',
        treatment='same frozen predictions passed through current shared/university policy output rules',
        ensemble='mean of all ten fixed seeds; no seed or configuration selection using target scores',
        metric='exact final school set vs preserved official_map_point_membership; full denominator',
        bootstrap=dict(seed=20261104,draws=3000,block_metres=1000,paired=True,stratified_by_district=True),
        data_limitations='Baidu community POIs are office proxies, not verified boundaries; partial scope, unresolved identities and missing road names stay unknown',
        adoption='development external test; no tuning after target scoring or formal adoption')
    B.write(OUT/'protocol.json',protocol)
    code=[Path(__file__),Path(B.__file__),Path(F.__file__),REPO/'scripts/build_shenzhen_model_inputs.py']+sorted((REPO/'zoning').glob('*.py'))
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    pop,stats={},{}
    for d in protocol['sources']:
        x,pop[d]=source_inputs(d,cfg);ids=pop[d]['ids'];B.save_x(d,x)
        stats[d]=dict(sites=x.n,schools=x.school_n,reference_sites=len(set(x.ref_sites)),references=len(x.ref_sites),
                       training=len(ids),pair_features=x.np_,school_features=x.na,
                       pair_blocks=list(x.pair_blocks),school_blocks=list(x.anchor_blocks))
        B.log('SOURCE_FULL',d,stats[d])
    for d in B.TARGETS:
        with (F.OUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        with (F.OUT/d/'rules.pkl').open('rb') as f:r=pickle.load(f)
        assert dist.answers=={} and dist.train_answers=={}
        x=E.build(dist,r,cfg['evidence']);meta=B.read(F.OUT/d/'model_metadata.json')
        B.save_x(d,x);B.write(OUT/'inputs'/f'{d}.json',meta)
        pop[d]=dict(ids=x.supervised.tolist(),labels=x.y[x.supervised].tolist())
        stats[d]=dict(B.read(F.OUT/d/'input_summary.json'),pair_features=x.np_,school_features=x.na,
                     pair_blocks=list(x.pair_blocks),school_blocks=list(x.anchor_blocks),
                     feature_nonzero={k:int(np.count_nonzero(x.pair[...,v])) for k,v in x.pair_blocks.items()},
                     anchor_nonzero={k:int(np.count_nonzero(x.anchor[...,v])) for k,v in x.anchor_blocks.items()})
        assert x.np_==41 and x.na==27
        B.log('TARGET_FULL',d,stats[d])
    assert all(s['pair_blocks']==cfg['evidence']['pair_blocks'] and s['school_blocks']==cfg['evidence']['school_blocks'] for s in stats.values())
    B.write(OUT/'training_population.json',pop);B.write(OUT/'input_audit.json',stats)
    frozen=[OUT/'protocol.json',OUT/'training_population.json',OUT/'input_audit.json']
    frozen+=sorted((OUT/'inputs').glob('*'))+sorted((OUT/'code').rglob('*.py'))
    B.write(OUT/'input_manifest.json',dict(frozen_sha256={str(p.relative_to(OUT)):B.sha(p) for p in frozen},
        target_gold_read=False,target_polygons_read=False,full_feature_blocks_equal_default=True,
        original_input_manifest=str(F.OUT/'manifest.json'),original_input_manifest_sha256=B.sha(F.OUT/'manifest.json')))


def interval(rows,c):
    # Exactly the main Changsha evaluator's district-stratified 1km block bootstrap.
    from zoning.evaluate import bootstrap
    out=[];deltas=[]
    xy={}
    for d in B.TARGETS:
        with (F.OUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,key):p for key,p in zip(dist.site_ids,dist.coords)})
    for r in rows:
        p=xy[r['district'],r['site_id']]
        out.append(dict(d=r['district'],block=str((int(p[0]//1000),int(p[1]//1000)))))
        deltas.append(int(set(r['after'])==set(r['gold']))-int(set(r['before'])==set(r['gold'])))
    ci=bootstrap(out,deltas,seed=c['seed'],draws=c['draws'])
    return dict(delta_95_percentile=[v/100 for v in ci],blocks=len({(r['d'],r['block']) for r in out}),
                describes='native Changsha evaluator: paired district-stratified 1km blocks,3000draws')


def score():
    B.interval=interval
    B.score()
    r=B.read(OUT/'results.json');r['status']='completed_native_full_feature_development_external_test'
    B.write(OUT/'results.json',r)
    text=['深圳完整特征输入外测（2026-11-11）','',
        '使用与长沙当前默认配置完全一致的41列参照特征、27列学校特征、RetrievalScorer、10个种子、40步训练。',
        '来源为长沙和杭州八区；目标为龙华和龙岗。不用深圳Gold或学区polygon构建输入、选参照或训练。',
        'BD09统一转WGS84，再投影EPSG32650；OSM原生WGS84投影EPSG32650。社区推断沿用长沙跨路约束和400/800米阈值。',
        '道路判边直接执行长沙strict_line原函数；范围解析保留未决条件；社区/道路规则参照仍是估计，不是人工核验真值。',
        '学校坐标含新补高新校区独立百度POI；原学校目录和旧Gold保持原样。',
        '社区名未核实或道路未找到时保留未知，不能声称数据覆盖率与长沙完全相同。','',
        '地区\t输出规则前\t输出规则后\t修正\t损害\t新增原图外学校的小区']
    for d,v in list(r['districts'].items())+[('合计',dict(all_map=r['overall']))]:
        a=v['all_map'];text.append(f"{d}\t{a['before']['correct']}/{a['before']['n']} ({a['before']['exact_accuracy']:.2%})\t{a['after']['correct']}/{a['after']['n']} ({a['after']['exact_accuracy']:.2%})\t{a['corrected']}\t{a['harmed']}\t{a['wrongly_added_sites']}")
    text+=['','评分是完整学校集合对原图命中集合，保留全部1761个地图有命中坐标组；全部1802组预测均输出。',
           '原图存在基础范围/大学区/派位范围语义待核项，results.json已单列。','',
           '结果与逐例：results.json / scored_cases.json；输入覆盖：input_audit.json；输入来源：data/深圳市/完整输入_20261111。']
    (OUT/'实验结果.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['sources','prepare','train','score']);a=ap.parse_args()
    {'sources':sources,'prepare':prepare,'train':B.train,'score':score}[a.phase]()
