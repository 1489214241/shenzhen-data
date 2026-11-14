"""Round03: masked versus assume-negative supervision, same native retrieval scores.

Phases prepare/train/calibrate/score are separate. Source spatial validation sets
the threshold. Shenzhen evaluation labels cannot enter the first three phases.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
import shenzhen_full_inputs as F
from zoning import data as D, evidence as E, model as M, DEVICE
from zoning import shenzhen_policy as SP
from shenzhen_partial_model import BernoulliRetrieval,masked_bce,choose_threshold,predict_sets,school_log_mass

OUT=F.REPO/'outputs/shenzhen_resolution_20261112/round03_partial_labels'
INPUT=F.REPO/'data/深圳市/修复实验输入_20261112'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
PREVIOUS=F.REPO/'outputs/shenzhen_resolution_20261112/round02_scope_inclusion_v2'
CACHE=F.REPO/'outputs/shenzhen_full_source_cache_20261111'
TARGETS=['龙华区','龙岗区']
ARMS=['masked_bce','assume_negative']
START=time.monotonic()


def log(*v):print(round(time.monotonic()-START,1),*v,flush=True)


def save_pickle(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:pickle.dump(value,f,protocol=5)


def source_fold(dist):
    # One fold for each whole property group, assigned via its centroid's 1km cell.
    out=np.empty(dist.n,int)
    for group in np.unique(dist.groups):
        ids=np.flatnonzero(dist.groups==group)
        center=dist.coords[ids].mean(0);cell=tuple(np.floor(center/1000).astype(int))
        key=f'partial-v1|{dist.city}|{dist.name}|{cell[0]}|{cell[1]}'
        out[ids]=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%5
    return out


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    cfg=F.read(F.REPO/'configs/default.json')
    for rel,h in F.read(INPUT/'manifest.json')['files'].items():assert F.sha(INPUT/rel)==h
    for rel,h in F.read(CACHE/'manifest.json')['files'].items():assert F.sha(CACHE/rel)==h
    for rel,h in F.read(PREVIOUS/'prediction_freeze.json')['files'].items():assert F.sha(PREVIOUS/rel)==h
    protocol=dict(round=3,date='2026-11-12',sources=cfg['districts'],targets=TARGETS,arms=ARMS,
        seeds=cfg['train']['seeds'],warm_steps=100,warm_lr=.04,steps=40,lr=.003,reg=.01,width=32,chunk=64,
        model='Native 41/27 feature retrieval logits; per-school log-sum-exp then independent sigmoid, one shared binary bias.',
        primary='masked_bce: source complete sets, target observed-positive estimates, no target unknown negatives',
        control='assume_negative: same model/data/budget; target and any source missing labels become negatives only in this control',
        train_weight='sum BCE over school cells; each property group equal; global weighted average scaled by number of active district batches',
        source_validation='20% spatial cells via fixed SHA256 modulo5; all same-property rows share fold; calibration only, no unseen-city validation claim; no buffer',
        source_references='Independent rule-derived references remain available, including source validation locations; query own property group is masked.',
        warm_start='100 steps on all training districts with linear-only masked/AN objective; reset regularization anchor then 40 nonlinear steps. Same budget for both arms; differs from old target-only singleton warm-start.',
        threshold_grid=[float(v/100) for v in range(10,91,5)],threshold_selection='source validation exact set, then fewer FP, then higher threshold; all-seed ensemble; same selected threshold for seed reports',
        prediction='Independent school membership threshold, fallback argmax if empty; partial positive lower bound retained; source scope implication once; university predicted base only',
        target_gold_read=False,target_cardinality_used=False,target_negative_rule_estimates_as_supervision=False,
        independently_verified_target_support_budget=0,independent_validation=False,adopted=False,
        comparison_limit='Old round02 used 100% source supervision and different single-label objective/warm-start. Matched comparison is masked versus assume-negative; round02 is the existing performance baseline.',
        sources_read=[dict(url='https://arxiv.org/html/2106.09708v1',use='Sections4.1/4.2/4.3 and author code: complete/partial/unknown label distinction; IU collapses with positive-only data, hence source negatives.'),
            dict(url='https://raw.githubusercontent.com/elijahcole/single-positive-multi-label/main/losses.py',use='Reviewed loss_bce, loss_iu, loss_an; sum-over-classes scaling here is documented, not exact reproduction.')])
    F.write(OUT/'protocol.json',protocol)
    stats={};source_hashes={}
    for d in cfg['districts']+TARGETS:
        if d in TARGETS:
            with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
            with (INPUT/d/'rules.pkl').open('rb') as f:rules=pickle.load(f)
            assert dist.answers=={} and dist.train_answers=={}
            x=E.build(dist,rules,cfg['evidence'])
            meta=F.read(INPUT/d/'model_metadata.json');F.write(OUT/'inputs'/f'{d}.json',meta)
            labels=np.full((x.n,x.school_n),-1,np.float32)
            for q,ss in meta['known'].items():labels[int(q),ss]=1
            fold=np.full(x.n,-1);valid_ids=[];valid_labels=np.zeros((0,x.school_n),bool)
            graph=F.read(PREVIOUS/'source_graphs'/f'{d}.json');F.write(OUT/'source_graphs'/f'{d}.json',graph)
        else:
            with (CACHE/f'{d}.pkl').open('rb') as f:x,_=pickle.load(f)
            dist=D.with_list_gold(D.load(d));fold=source_fold(dist)
            labels=np.full((x.n,x.school_n),-1,np.float32)
            for q,s in zip(x.ref_sites,x.ref_schools):labels[q,s]=1
            full={q:ss for q,ss in dist.answers.items() if len(ss) and min(ss)>=0 and max(ss)<x.school_n}
            for q,ss in full.items():labels[q,:]=0;labels[q,ss]=1
            valid_ids=np.array(sorted(q for q in full if fold[q]==0),int)
            valid_labels=labels[valid_ids].astype(bool).copy()
            labels[fold==0]=-1
            for p in (F.REPO/'data'/d).glob('*'):
                if p.is_file() and p.suffix in ['.json','.npz']:source_hashes[str(p.relative_to(F.REPO))]=F.sha(p)
            assert not(set(dist.groups[valid_ids])&set(dist.groups[np.flatnonzero((labels>=0).any(1))]))
        ids=np.flatnonzero((labels>=0).any(1))
        assert x.np_==41 and x.na==27
        save_pickle(OUT/'inputs'/f'{d}.pkl',x)
        np.savez_compressed(OUT/'inputs'/f'{d}_labels.npz',labels=labels,train_ids=ids,
            valid_ids=np.asarray(valid_ids,int),valid_labels=valid_labels,fold=fold)
        stats[d]=dict(sites=x.n,schools=x.school_n,training_rows=len(ids),validation_rows=len(valid_ids),
            source_validation_multilabel=int((valid_labels.sum(1)>1).sum()),
            positive_cells=int((labels==1).sum()),negative_cells=int((labels==0).sum()),unknown_cells=int((labels<0).sum()),
            train_multilabel_rows=int(((labels[ids]==1).sum(1)>1).sum()))
        log('INPUT',d,stats[d])
    F.write(OUT/'input_audit.json',dict(districts=stats,source_hashes=source_hashes,target_gold_read=False,
        source_train_validation_groups_disjoint=True,source_validation_references_are_policy_derived=True))
    code=[Path(__file__),F.REPO/'scripts/shenzhen_partial_model.py',F.REPO/'scripts/shenzhen_scope_implication.py',
        F.REPO/'tests/test_shenzhen_partial_model.py',F.REPO/'scripts/shenzhen_full_inputs.py']+list((F.REPO/'zoning').glob('*.py'))
    for src in code:
        dest=OUT/'code'/src.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    log('PREPARE_FROZEN')


def verify_inputs():
    frozen=F.read(OUT/'input_freeze.json')
    for rel,h in frozen['files'].items():assert F.sha(OUT/rel)==h,rel
    for p in (OUT/'code').rglob('*.py'):assert F.sha(F.REPO/p.relative_to(OUT/'code'))==F.sha(p),p
    return F.read(OUT/'protocol.json')


def load_inputs(c):
    xs={};ys={}
    for d in c['sources']+c['targets']:
        with (OUT/'inputs'/f'{d}.pkl').open('rb') as f:xs[d]=pickle.load(f)
        ys[d]=dict(np.load(OUT/'inputs'/f'{d}_labels.npz'))
    return xs,ys


def frames(xs,ys,c):
    active=[d for d in xs if len(ys[d]['train_ids'])]
    total=sum(len(set(xs[d].groups[ys[d]['train_ids']])) for d in active)
    batches=[]
    for d in active:
        x=xs[d];ids=ys[d]['train_ids'];_,inv,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True)
        weights=len(active)/total/count[inv]
        for lo in range(0,len(ids),c['chunk']):
            q=ids[lo:lo+c['chunk']];t=M.batch(x,q)
            t['labels']=torch.from_numpy(ys[d]['labels'][q]).to(DEVICE)
            t['weight']=torch.tensor(weights[lo:lo+c['chunk']],dtype=torch.float32,device=DEVICE)
            # Keep batches on GPU if feasible: cached contexts are fixed and do not need gradients.
            batches.append(t)
    log('FRAMES',len(batches),'GROUPS',total,'GPU_GB',round(torch.cuda.memory_allocated()/1e9,2) if torch.cuda.is_available() else 0)
    return batches


def checkpoint(path,model,opt,stage,step,history,protocol_hash):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,step=step,history=history,protocol_hash=protocol_hash),tmp)
    os.replace(tmp,path)


def probabilities(model,x,ids,chunk):
    outputs=[]
    with torch.no_grad():
        for lo in range(0,len(ids),chunk):outputs.append(model(M.batch(x,ids[lo:lo+chunk])).sigmoid().cpu().numpy())
    p=np.concatenate(outputs) if outputs else np.zeros((0,x.school_n),np.float32)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def train():
    F.guard();c=verify_inputs();xs,ys=load_inputs(c);batches=frames(xs,ys,c)
    start=M.LinearRetrieval(next(iter(xs.values())))
    ph=F.sha(OUT/'protocol.json');history={}
    for seed in c['seeds']:
        for arm in c['arms']:
            model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE)
            ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';prior=None
            if ck.exists():
                prior=torch.load(ck,map_location=DEVICE,weights_only=False)
                assert prior['protocol_hash']==ph
                model.load_state_dict(prior['model'])
            losses=prior['history'] if prior else {'warm':[],'full':[]}
            an=arm=='assume_negative'
            for stage,steps,lr in [('warm',c['warm_steps'],c['warm_lr']),('full',c['steps'],c['lr'])]:
                if prior and prior['stage']=='full' and stage=='warm':continue
                for name,p in model.named_parameters():p.requires_grad_(stage=='full' or name in ['p','a','offset','binary_bias'])
                if stage=='full' and (not prior or prior['stage']!='full'):
                    model.start.copy_(torch.cat([model.p.detach(),model.a.detach(),model.offset.detach()[None]]))
                opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
                begin=0
                if prior and prior['stage']==stage:opt.load_state_dict(prior['optimizer']);begin=prior['step']
                for step in range(begin,steps):
                    opt.zero_grad();regularizer=model.regularization(c['reg']);regularizer.backward();loss_total=float(regularizer.detach())
                    for t in batches:
                        if stage=='warm':
                            pl=(t['pair']@model.p+t['gb'][None]).masked_fill(~t['valid'],-torch.inf)
                            al=t['anchor']@model.a+model.offset
                            logits=school_log_mass(torch.cat([pl,al],1),t['mem'])+model.binary_bias
                        else:logits=model(t)
                        loss=masked_bce(logits,t['labels'],t['weight'],an)
                        assert torch.isfinite(loss);loss.backward();loss_total+=float(loss.detach())
                    nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses[stage].append(loss_total)
                    if (step+1)%10==0 or step+1==steps:
                        checkpoint(ck,model,opt,stage,step+1,losses,ph)
                        log('TRAIN',arm,seed,stage,step+1,round(loss_total,5))
                prior=None
            out=OUT/'probabilities'/arm;out.mkdir(parents=True,exist_ok=True)
            payload={}
            for d,x in xs.items():
                ids=np.arange(x.n) if d in TARGETS else ys[d]['valid_ids']
                payload[d]=probabilities(model,x,ids,c['chunk'])
            np.savez_compressed(out/f'{seed}.npz',**payload)
            history[f'{arm}/{seed}']=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(out/f'{seed}.npz'))
            F.write(OUT/'training_history.json',history)
            log('MODEL_COMPLETE',arm,seed)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},
        protocol_sha256=ph,target_gold_read=False,device=DEVICE))
    log('ALL_MODELS_FROZEN')


def calibrate():
    F.guard();c=verify_inputs();xs,ys=load_inputs(c)
    for rel,h in F.read(OUT/'probability_freeze.json')['files'].items():assert F.sha(OUT/rel)==h
    thresholds={}
    for arm in c['arms']:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{seed}.npz')) for seed in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in xs}
        threshold,table=choose_threshold([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']],c['threshold_grid'])
        thresholds[arm]=dict(threshold=threshold,source_validation_table=table)
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],loaded)]:
            for d in TARGETS:
                meta=F.read(OUT/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
                policy=SP.load_policy(d);graph=F.read(OUT/'source_graphs'/f'{d}.json')
                rows=[]
                for q,site in enumerate(meta['sites']):
                    positive=[sid[j] for j in meta['known'].get(str(q),[])]
                    r=predict_sets(values[d][q],threshold,sid,policy,graph,site,positive)
                    rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],after=r['predicted_school_ids'],decision=r))
                F.write(OUT/'predictions'/arm/label/f'{d}.json',rows)
        log('CALIBRATED',arm,threshold,max(r['correct'] for r in table),'/',table[0]['n'])
    F.write(OUT/'source_calibration.json',thresholds)
    files=[OUT/'source_calibration.json']+list((OUT/'predictions').rglob('*.json'))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in files},target_gold_read=False,
        calibration='source spatial holdout only',target_cardinality_used=False))
    log('TARGET_PREDICTIONS_FROZEN')


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c=F.read(OUT/'protocol.json')
    for rel,h in F.read(OUT/'prediction_freeze.json')['files'].items():assert F.sha(OUT/rel)==h
    for d,h in F.read(BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')}
    xy={}
    for d in TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    all_reports={}
    for arm in c['arms']:
        reports={}
        for label in ['ensemble']+[str(s) for s in c['seeds']]:
            rows=[]
            for d in TARGETS:
                prior={r['site_id']:r for r in F.read(PREVIOUS/'predictions'/label/f'{d}.json')}
                for r in F.read(OUT/'predictions'/arm/label/f'{d}.json'):
                    g=gold[d,r['site_id']]
                    if not g['has_map_gold']:continue
                    r['gold']=set(g['gold']);r['before']=prior[r['site_id']]['after'];r['semantic_pending']=g['semantic_pending']
                    pt=xy[d,r['site_id']];r['d']=d;r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
            assert len(rows)==1761
            report={}
            for name,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
                delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                report[name]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),corrected=delta.count(1),harmed=delta.count(-1),
                    delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            reports[label]=report
            if label=='ensemble':
                F.write(OUT/f'{arm}_diagnostic_not_training.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
        all_reports[arm]=reports
    F.write(OUT/'results.json',dict(reports=all_reports,adopted=False,independent_validation=False))
    # Matched mechanism comparison, alongside the stronger existing baseline.
    a=F.read(OUT/'masked_bce_diagnostic_not_training.json');b=F.read(OUT/'assume_negative_diagnostic_not_training.json')
    assert [(r['district'],r['site_id']) for r in a]==[(r['district'],r['site_id']) for r in b]
    paired={}
    for d in ['全部']+TARGETS:
        ids=[i for i,r in enumerate(a) if d=='全部' or r['district']==d]
        rs=[a[i] for i in ids]
        delta=[int(set(a[i]['after'])==set(a[i]['gold']))-int(set(b[i]['after'])==set(b[i]['gold'])) for i in ids]
        paired[d]=dict(n=len(ids),corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
            delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
    F.write(OUT/'matched_masked_vs_assume_negative.json',paired)
    lines=['第三轮：完整多校监督 + 深圳部分正例；未知不等于负例','',
        '同一1761个地图答案样本。阈值只用来源区空间留出集选择。开发实验，未自动采用。',
        '与第二轮并非纯损失消融：本轮留出来源区校准集、共同训练两区并改变暖启动。两臂之间才是同数据同预算比较。','']
    for arm,reports in all_reports.items():
        lines.append(arm)
        for d in ['全部']+TARGETS+['单校','多校']:
            r=reports['ensemble'][d]
            lines.append(f"{d}: {r['after']['correct']}/{r['after']['n']}={r['after']['exact_accuracy']:.2%}; 改正/损害 {r['corrected']}/{r['harmed']}; 误报/漏报 {r['after']['fp']}/{r['after']['fn']}")
    lines += ['',str(paired)]
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'verification.json',dict(passed=True,seeds=len(c['seeds']),arms=ARMS,scored_sites=1761,
        target_gold_hashes_unchanged=True,source_only_calibration=True))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','calibrate','score']);a=p.parse_args()
    dict(prepare=prepare,train=train,calibrate=calibrate,score=score)[a.phase]()
