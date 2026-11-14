"""Round06: matched continuation controls for rare multi-set learning.

Same ten masked-BCE parent checkpoints, inputs,40steps, optimizer resets and
decoder. Compare continuation, complete-source balancing, balancing plus LSEP.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import shenzhen_full_inputs as F
from zoning import model as M,DEVICE,shenzhen_policy as SP
from shenzhen_partial_model import BernoulliRetrieval,masked_bce,predict_sets
from shenzhen_balanced_ranking import balanced_weights,masked_lsep
from shenzhen_exact_calibration import exact_thresholds
from experiment_shenzhen_partial_labels import probabilities,checkpoint,log

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OLD=ROOT/'round03_partial_labels'
OUT=ROOT/'round06_balanced_ranking'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
INPUT=F.REPO/'data/深圳市/修复实验输入_20261112'
ARMS=['continued_bce','balanced_bce','balanced_rank']
OBJECTIVES=['micro_exact','balanced_exact']
TARGETS=['龙华区','龙岗区']


def load_inputs(c):
    xs={};ys={}
    for d in c['sources']+TARGETS:
        with (OLD/'inputs'/f'{d}.pkl').open('rb') as f:xs[d]=pickle.load(f)
        ys[d]=dict(np.load(OLD/'inputs'/f'{d}_labels.npz'))
    return xs,ys


def make_weights(xs,ys,c):
    active=[d for d in xs if len(ys[d]['train_ids'])]
    total=sum(len(set(xs[d].groups[ys[d]['train_ids']])) for d in active);weights={}
    for d,x in xs.items():
        ids=ys[d]['train_ids'];_,inv,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True)
        w=np.zeros(x.n);w[ids]=len(active)/total/count[inv];weights[d]=w
    balanced,audit=balanced_weights({d:z['labels'] for d,z in ys.items()},weights,c['sources'])
    return weights,balanced,audit


def verify():
    for rel,h in F.read(OUT/'input_freeze.json')['files'].items():assert F.sha(OUT/rel)==h,rel
    for filename in ['input_freeze.json','probability_freeze.json']:
        for rel,h in F.read(OLD/filename)['files'].items():assert F.sha(OLD/rel)==h,rel
    for p in (OUT/'code').rglob('*.py'):assert F.sha(F.REPO/p.relative_to(OUT/'code'))==F.sha(p),p
    return F.read(OUT/'protocol.json')


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    for filename in ['input_freeze.json','probability_freeze.json','prediction_freeze.json']:
        for rel,h in F.read(OLD/filename)['files'].items():assert F.sha(OLD/rel)==h,rel
    old=F.read(OLD/'protocol.json')
    c=dict(round=6,date='2026-11-12',sources=old['sources'],targets=TARGETS,seeds=old['seeds'],arms=ARMS,objectives=OBJECTIVES,
        steps=40,lr=.003,reg=.01,width=32,chunk=64,ranking_coefficient=1.,
        hypothesis='Only94complete-source multi-label training rows versus4790singletons; parent fits2/94multi sets. Source-balanced mass and hard-positive ranking may improve complete-set learning.',
        parent_arm='masked_bce',parent=str(OLD),initialization='same frozen parent checkpoint per seed; fresh Adam for every arm; keep original regularization anchor',
        continuation_control='continued_bce: same original weighted maskedBCE plus40steps, distinguishes extra optimization from intervention',
        balanced_bce='source complete singleton/multi row mass each50%; preserve total complete-source mass, property weighting, source partial and all target weights',
        balanced_rank='balanced_bce plus coefficient1 LSEP Eq7 on observed positive-negative pairs; partial target has no negative cells and contributes no ranking loss',
        scope='Same frozen old inputs and policy decoder; round04geography changes excluded to isolate training changes',
        threshold='exact source-only calibration on[0,1]; both micro and balanced objectives reported; never select by Shenzhen scores',
        validation='same1246source spatial holdout rows,60multi; already used for development, not independent confirmation',
        source_negative_labels_only=True,target_gold_read=False,target_polygons_read=False,target_cardinality_used=False,
        independent_validation=False,adopted=False,
        research=dict(url='https://arxiv.org/pdf/1704.03135',sections_read='3.1,3.2,3.3; equation7',
            adaptation='Exact all-pair factorization, BCE retained for absolute probability calibration; no label-decision neural module or full paper reproduction.'),
        comparison='Both districts/allseeds; strict1761fixedmap-hit set,FP/FN,corrected/harmed,spatialintervals. Compare to matched continuedBCE and frozen parent.')
    F.write(OUT/'protocol.json',c);xs,ys=load_inputs(c);w,b,a=make_weights(xs,ys,c)
    for d in xs:np.savez_compressed(OUT/f'{d}_weights.npz',original=w[d],balanced=b[d])
    F.write(OUT/'weight_audit.json',a)
    F.write(OUT/'input_links.json',dict(parent_freezes={f:F.sha(OLD/f) for f in ['input_freeze.json','probability_freeze.json','prediction_freeze.json']},
        initial_models={str(s):F.sha(OLD/'checkpoints'/f'masked_bce_{s}.pt') for s in c['seeds']},input_changes=False))
    for src in [Path(__file__),F.REPO/'scripts/shenzhen_balanced_ranking.py',F.REPO/'scripts/shenzhen_partial_model.py',
                F.REPO/'scripts/shenzhen_exact_calibration.py',F.REPO/'scripts/experiment_shenzhen_partial_labels.py',
                F.REPO/'scripts/experiment_shenzhen_set_decode.py',F.REPO/'tests/test_shenzhen_balanced_ranking.py']+list((F.REPO/'zoning').glob('*.py')):
        dest=OUT/'code'/src.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    log('INPUTS_FROZEN',a)


def train():
    F.guard();c=verify();xs,ys=load_inputs(c);frames=[]
    for d,x in xs.items():
        ids=ys[d]['train_ids'];weights=dict(np.load(OUT/f'{d}_weights.npz'))
        for lo in range(0,len(ids),c['chunk']):
            q=ids[lo:lo+c['chunk']];t=M.batch(x,q)
            t['labels']=torch.from_numpy(ys[d]['labels'][q]).to(DEVICE)
            for kind in ['original','balanced']:t[kind]=torch.tensor(weights[kind][q],dtype=torch.float32,device=DEVICE)
            frames.append(t)
    log('FRAMES',len(frames),'GPU_GB',torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0)
    start=M.LinearRetrieval(next(iter(xs.values())));ph=F.sha(OUT/'protocol.json');history={}
    if (OUT/'training_history.json').exists():history=F.read(OUT/'training_history.json')
    for seed in c['seeds']:
        parent=torch.load(OLD/'checkpoints'/f'masked_bce_{seed}.pt',map_location=DEVICE,weights_only=False)
        for arm in ARMS:
            model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);ck=OUT/'checkpoints'/f'{arm}_{seed}.pt'
            begin=0;losses=[]
            if ck.exists():
                saved=torch.load(ck,map_location=DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();penalty=model.regularization(c['reg']);penalty.backward()
                bce_total=rank_total=0.
                for t in frames:
                    logits=model(t);weights=t['original' if arm=='continued_bce' else 'balanced']
                    bce=masked_bce(logits,t['labels'],weights)
                    rank=masked_lsep(logits,t['labels'],weights) if arm=='balanced_rank' else logits.sum()*0
                    loss=bce+c['ranking_coefficient']*rank;assert torch.isfinite(loss);loss.backward()
                    bce_total+=float(bce.detach());rank_total+=float(rank.detach())
                norm=float(nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bce_total,ranking=rank_total,regularizer=float(penalty.detach()),grad_norm=norm))
                if (step+1)%10==0:
                    checkpoint(ck,model,opt,'full',step+1,losses,ph);log('TRAIN',arm,seed,step+1,losses[-1])
            model.eval();payload={}
            for d,x in xs.items():
                ids=np.arange(x.n) if d in TARGETS else ys[d]['valid_ids'];payload[d]=probabilities(model,x,ids,c['chunk'])
                if d in c['sources']:
                    y=ys[d]['labels'];ids=np.flatnonzero((y>=0).all(1)&((y==1).sum(1)>1))
                    payload[d+'_train_multi']=probabilities(model,x,ids,c['chunk'])
            dest=OUT/'probabilities'/arm;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{seed}.npz',**payload)
            history[f'{arm}/{seed}']=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest/f'{seed}.npz'))
            F.write(OUT/'training_history.json',history);log('MODEL_COMPLETE',arm,seed)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    log('MODELS_FROZEN')


def calibrate():
    F.guard();c=verify();xs,ys=load_inputs(c)
    for rel,h in F.read(OUT/'probability_freeze.json')['files'].items():assert F.sha(OUT/rel)==h,rel
    audit={}
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in loaded[0]}
        cal=exact_thresholds([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']])
        F.write(OUT/'source_calibration'/f'{arm}.json',cal);audit[arm]=cal['selections']
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],loaded)]:
            for d in TARGETS:
                meta=F.read(OLD/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
                policy=SP.load_policy(d);graph=F.read(OLD/'source_graphs'/f'{d}.json')
                for obj in OBJECTIVES:
                    t=cal['selections'][obj]['threshold'];rows=[]
                    for q,site in enumerate(meta['sites']):
                        positives=[sid[j] for j in meta['known'].get(str(q),[])]
                        r=predict_sets(np.asarray(values[d][q],np.float64),t,sid,policy,graph,site,positives)
                        rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],after=r['predicted_school_ids'],decision=r))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
        log('SOURCE_CALIBRATED',arm,audit[arm])
    F.write(OUT/'calibration_summary.json',audit)
    files=list((OUT/'source_calibration').glob('*.json'))+list((OUT/'predictions').rglob('*.json'))+[OUT/'calibration_summary.json']
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in files},target_gold_read=False))
    log('PREDICTIONS_FROZEN')


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c=F.read(OUT/'protocol.json')
    for rel,h in F.read(OUT/'prediction_freeze.json')['files'].items():assert F.sha(OUT/rel)==h,rel
    for d,h in F.read(BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')};xy={}
    for d in TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={};lines=['第六轮：完整来源区多校样本平衡与排序损失，同起点同40步','',
        '同一1761个原图答案样本，未改Gold。每个方法与相同校准目标的继续BCE对照比较，另报冻结父模型。开发实验，不自动采用。','']
    for arm in ARMS:
        results[arm]={}
        for obj in OBJECTIVES:
            reports={}
            for label in ['ensemble']+[str(s) for s in c['seeds']]:
                rows=[];unscored=0
                for d in TARGETS:
                    control={r['site_id']:r for r in F.read(OUT/'predictions/continued_bce'/obj/label/f'{d}.json')}
                    parent={r['site_id']:r for r in F.read(OLD/'predictions/masked_bce'/label/f'{d}.json')}
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored+=1;continue
                        r['gold']=set(g['gold']);r['before']=control[r['site_id']]['after'];r['parent']=parent[r['site_id']]['after']
                        r['semantic_pending']=g['semantic_pending'];r['d']=d;pt=xy[d,r['site_id']]
                        r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
                assert len(rows)==1761 and unscored==41
                report={}
                for group,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1])]:
                    item=dict(after=metrics(rs,'after'))
                    for cmp in ['before','parent']:
                        delta=[int(set(r['after'])==r['gold'])-int(set(r[cmp])==r['gold']) for r in rs]
                        item[cmp]=dict(metrics=metrics(rs,cmp),corrected=delta.count(1),harmed=delta.count(-1),
                            delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
                    report[group]=item
                reports[label]=report
                if label=='ensemble':F.write(OUT/'diagnostics_not_training'/f'{arm}_{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
            results[arm][obj]=reports;lines.append(arm+'/'+obj)
            for group in ['全部']+TARGETS+['单校','多校']:
                r=reports['ensemble'][group];a=r['after'];b=r['before']
                lines.append(f"{group}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 误报/漏报{a['fp']}/{a['fn']}; 对继续BCE改正/损害{b['corrected']}/{b['harmed']}; 95%变化区间{b['delta_95_pp']}")
            log('SCORED',arm,obj,reports['ensemble']['全部'])
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'verification.json',dict(passed=True,models=30,seeds=10,objectives=2,scored_sites=1761,unscored_sites=41,
        gold_unchanged=True,source_only_calibration=True,adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','calibrate','score']);a=p.parse_args()
    dict(prepare=prepare,train=train,calibrate=calibrate,score=score)[a.phase]()
