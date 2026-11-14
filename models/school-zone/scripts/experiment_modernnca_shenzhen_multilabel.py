"""ModernNCA-LN single versus multi-label adaptation; source-only selection.

Phases: prepare, select, fit, infer, score. No production default is mutated.
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from zoning import DEVICE,model as MD,evaluate
from zoning.modernnca_multilabel import MultiLabelModernNCA,categorical_positive_loss,observed_bce_loss
from zoning.modernnca_normalization import LayerNormModernNCA
import shenzhen_full_inputs as F
from shenzhen_balanced_ranking import balanced_weights
from shenzhen_exact_calibration import exact_thresholds
from shenzhen_partial_model import predict_sets
from shenzhen_reviewed_supplements import ReviewedSupplements

ROOT=REPO/'outputs/shenzhen_resolution_20261112'
OLD=ROOT/'round03_partial_labels'
CURRENT=ROOT/'community_methods_20261113'
OUT=REPO/'outputs/modernnca_shenzhen_multilabel_20261113'
DOC=REPO.parent/'科研推进/ModernNCA深圳多校验证_20261113'
TARGETS=['龙华区','龙岗区']
METHODS=['categorical','multilabel']
ARMS=['categorical_raw','categorical_policy','multilabel_raw','multilabel_policy']
SEEDS=[20261231,20261232,20261233]
ROW_KEYS={'px','ax','valid','labels','weight'}
START=time.monotonic()


def log(*args):print(round(time.monotonic()-START,1),*args,flush=True)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)


def pkl(p):
    with Path(p).open('rb') as f:return pickle.load(f)


def guard():
    F.guard()
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/')
            if any(t in p for t in ['/diagnostics_not_training/','/scored_cases.json','/results.json','/eval.json','/scored_rows/']):
                raise PermissionError('Target scoring artifact forbidden before freeze: '+p)
    sys.addaudithook(hook)


def verify():
    c=read(OUT/'protocol.json')
    for path,h in read(OUT/'input_links.json').items():assert F.sha(Path(path))==h,path
    for rel,h in read(OUT/'code_hashes.json').items():assert F.sha(REPO/rel)==h,rel
    return c


def inputs(c):
    xs={};ys={}
    for d in c['sources']+TARGETS:
        path=CURRENT/'inputs/current'/f'{d}.pkl' if d in TARGETS else OLD/'inputs'/f'{d}.pkl'
        xs[d]=pkl(path);ys[d]=dict(np.load(OLD/'inputs'/f'{d}_labels.npz'))
    return xs,ys


def prepare():
    guard()
    if (OUT/'protocol.json').exists():raise FileExistsError('Preserve frozen experiment')
    old=read(OLD/'protocol.json')
    c=dict(date='2026-11-13',backbone='ModernNCA-LN full-evidence relation adapter, dim32, all legal references, no SNS/top-k or RetrievalScorer weights',
        methods=METHODS,arms=ARMS,sources=old['sources'],targets=TARGETS,seeds=SEEDS,selection_seed=20261230,
        grid=[dict(lr=lr,reg=reg) for lr in [.001,.003] for reg in [.001,.01]],checkpoints=[40,80,160,320],
        minibatch=64,inference_chunk=64,clip=5.,dim=32,
        categorical='Same ModernNCA categorical probabilities. Mean native NLL across observed positives (equivalent repeated single-label questions with split weight), argmax output. Same underlying positive relations, not historical singleton-only training reproduction.',
        multilabel='Same token logits and school aggregation before across-school normalization, plus a shared binary bias. Masked BCE on observed cells; unknown cells have zero loss; no expected target cardinality.',
        supervision='Reuse frozen complete source labels and partial target positives from round03, identical admissions/folds across methods. Same-property weighting; source complete single/multi mass balanced50/50 for BOTH methods. No new target facts promoted into training.',
        validation='Existing 1246 source spatial holdout rows. Model selection uses mean of single/multi exact rates, then micro exact, fewer FP, earlier step. Each method same four-trial320-step cap; no post-target tuning. Boundary optimum is reported as unresolved optimization.',
        threshold='Exact source-validation balanced-exact threshold selected for multilabel at each selection checkpoint; final threshold recalibrated from final three-seed SOURCE ensemble only. Categorical uses argmax.',
        policy='Both methods receive identical current decoder, university base-only handling, nontransitive scope implications, partial-positive lower bounds and cumulative reviewed supplements. Raw outputs separately reported; policy does NOT use target map geometry.',
        target_input='Same current Shenzhen feature/reference inputs as retained community comparison. Source school anchors preserve zero-support candidates.',
        scoring='Same1761 map-hit coordinate groups;41 outside-map rows output but not scored. Full-set exact, single/multi, districts, FP/FN, precision/recall, corrected/harmed; district-stratified1km bootstrap3000.',
        primary='multilabel_policy minus categorical_policy',secondary='multilabel_raw minus categorical_raw; policy on/off for each method; incumbent959 is historical context with different training protocol',
        status='Exploratory development experiment; no automatic adoption. No target Gold/geometry/label cardinality in selection or inference. Existing source holdout and target geography are not independent unseen validation.',
        target_gold_read=False,target_polygons_read=False,adopted=False,independent_validation=False)
    write(OUT/'protocol.json',c);xs,ys=inputs(c);paths=[];audit={};weights={}
    active=[d for d in xs if len(ys[d]['train_ids'])]
    total=sum(len(set(xs[d].groups[ys[d]['train_ids']])) for d in active)
    for d,x in xs.items():
        y=ys[d];ids=y['train_ids'];_,inv,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True)
        weights[d]=np.zeros(x.n);weights[d][ids]=len(active)/total/count[inv]
        assert not(set(x.groups[ids])&set(x.groups[y['valid_ids']]))
        assert np.array_equal(ids,np.flatnonzero((y['labels']>=0).any(1)))
        paths += [OLD/'inputs'/f'{d}_labels.npz',CURRENT/'inputs/current'/f'{d}.pkl' if d in TARGETS else OLD/'inputs'/f'{d}.pkl']
        if d in TARGETS:
            assert (y['labels']!=0).all() and len(y['valid_ids'])==0
            oldx=pkl(OLD/'inputs'/f'{d}.pkl')
            for key in ['pair','anchor','ref_sites','ref_schools','self_mask','groups','y']:
                assert np.array_equal(getattr(oldx,key),getattr(x,key)),(d,key)
            sup=ReviewedSupplements(d);paths += [Path(p) for p in sup.hashes]
            paths += [CURRENT/'metadata'/f'{d}.json',ROOT/'round08_reference_retraining/policies'/f'{d}.json',OLD/'source_graphs'/f'{d}.json']
            meta=read(CURRENT/'metadata'/f'{d}.json')
            for q,ss in meta['known'].items():assert set(np.flatnonzero(y['labels'][int(q)]==1))==set(ss)
        audit[d]=dict(n=x.n,schools=x.school_n,references=len(x.ref_sites),reference_schools=len(set(x.ref_schools)),
            training_rows=len(ids),complete_rows=int((y['labels'][ids]>=0).all(1).sum()),
            complete_multi_rows=int(((y['labels'][ids]>=0).all(1)&((y['labels'][ids]==1).sum(1)>1)).sum()),
            validation_rows=len(y['valid_ids']),validation_multi=int((y['valid_labels'].sum(1)>1).sum()))
    balanced,bal_audit=balanced_weights({d:y['labels'] for d,y in ys.items()},weights,c['sources'])
    for d,w in balanced.items():np.save(OUT/f'{d}_weights.npy',w)
    paths += [OLD/'input_freeze.json',CURRENT/'input_freeze.json']+[OUT/f'{d}_weights.npy' for d in xs]
    write(OUT/'input_links.json',{str(p):F.sha(p) for p in sorted(set(paths))})
    code=[Path(__file__),REPO/'zoning/modernnca_multilabel.py',REPO/'zoning/modernnca_normalization.py',
        REPO/'zoning/full_evidence.py',REPO/'zoning/model.py',REPO/'zoning/shenzhen_policy.py',REPO/'zoning/evaluate.py',
        REPO/'scripts/shenzhen_partial_model.py',REPO/'scripts/shenzhen_exact_calibration.py',REPO/'scripts/shenzhen_balanced_ranking.py',
        REPO/'scripts/shenzhen_current_supplements.py',REPO/'scripts/shenzhen_reviewed_supplements.py',REPO/'scripts/shenzhen_scope_implication.py',
        REPO/'third_party/TALENT/model/models/modernNCA.py',REPO/'tests/test_modernnca_multilabel.py']
    write(OUT/'code_hashes.json',{str(p.relative_to(REPO)):F.sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'input_audit.json',dict(districts=audit,balancing=bal_audit,train_validation_groups_disjoint=True,target_gold_read=False))
    write(DOC/'方案.json',c);log('PREPARED',audit)


def task(x,ids,labels=None,weights=None):
    parts=[]
    for lo in range(0,len(ids),64):
        q=ids[lo:lo+64];t=MD.batch(x,q)
        parts.append({k:t[k].detach().cpu() for k in ['px','ax','valid']})
    out={k:torch.cat([p[k] for p in parts]) for k in ['px','ax','valid']}
    out.update(mem=x.membership.cpu(),gb=torch.from_numpy(x.group_bias),n=len(ids),ids=ids)
    if labels is not None:out['labels']=torch.from_numpy(labels[ids]);out['weight']=torch.from_numpy(weights[ids].astype(np.float32))
    return out


def device_batch(t,ix):
    return {k:(v[ix] if k in ROW_KEYS else v).to(DEVICE) for k,v in t.items() if torch.is_tensor(v)}


def make_tasks(c,xs,ys):
    tr={};val={}
    for d,x in xs.items():
        y=ys[d]
        if len(y['train_ids']):tr[d]=task(x,y['train_ids'],y['labels'],np.load(OUT/f'{d}_weights.npy'))
        if d in c['sources']:val[d]=task(x,y['valid_ids'])
    size=sum(v.numel()*v.element_size() for t in tr.values() for v in t.values() if torch.is_tensor(v))
    if DEVICE=='cuda' and size+3*1024**3<torch.cuda.mem_get_info()[0]:
        tr={d:{k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in t.items()} for d,t in tr.items()}
        log('TRAIN CACHE GPU',round(size/2**20),'MB')
    log('TASKS READY',len(tr),len(val));return tr,val


def predict(m,t,method):
    m.eval();out=[]
    with torch.no_grad():
        for lo in range(0,t['n'],64):
            z=m(device_batch(t,slice(lo,lo+64)))
            out.append((z.softmax(1) if method=='categorical' else z.sigmoid()).cpu().numpy())
    p=np.concatenate(out);assert np.isfinite(p).all();return p


def validation(m,method,val,ys):
    probabilities=[predict(m,t,method) for t in val.values()]
    labels=[ys[d]['valid_labels'] for d in val]
    if method=='multilabel':
        cal=exact_thresholds(probabilities,labels);s=cal['selections']['balanced_exact']
        return dict(s,balanced=.5*(s['single_correct']/cal['source_single']+s['multi_correct']/cal['source_multi']))
    count=single=multi=fp=fn=ns=nm=0
    for p,y in zip(probabilities,labels):
        pred=np.zeros_like(y,dtype=bool);pred[np.arange(len(p)),p.argmax(1)]=True
        ok=(pred==y).all(1);ism=y.sum(1)>1
        single+=int(ok[~ism].sum());multi+=int(ok[ism].sum());ns+=int((~ism).sum());nm+=int(ism.sum())
        fp+=int((pred&~y).sum());fn+=int((~pred&y).sum());count+=len(y)
    return dict(n=count,correct=single+multi,fp=fp,fn=fn,single_correct=single,multi_correct=multi,threshold=None,
                balanced=.5*(single/ns+multi/nm))


def rank(v):return (v['balanced'],v['correct'],-v['fp'],-v['step'])


def optimize(method,hp,seed,steps,c,tr,val,ys,folder,selection):
    m=MultiLabelModernNCA(41,27,seed,c['dim']).to(DEVICE)
    if method=='categorical':m.binary_bias.requires_grad_(False)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=hp['lr'])
    rng=np.random.default_rng(seed);history=[];best=None
    for step in range(1,steps+1):
        m.train();opt.zero_grad();penalty=m.regularization(hp['reg']);penalty.backward();total=float(penalty.detach())
        for d,t in tr.items():
            ids=rng.choice(t['n'],min(c['minibatch'],t['n']),replace=False);b=device_batch(t,ids)
            z=m(b);loss=(categorical_positive_loss if method=='categorical' else observed_bce_loss)(z,b['labels'],b['weight'])*t['n']/len(ids)
            assert torch.isfinite(loss),(method,step,d);loss.backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c['clip']));opt.step()
        if step%20==0:log('TRAIN',method,hp,seed,step,'loss',round(total,5))
        if selection and step in c['checkpoints']:
            v=validation(m,method,val,ys);v.update(step=step,loss=total,grad_norm=norm);history.append(v)
            if best is None or rank(v)>rank(best):
                best=v;folder.mkdir(parents=True,exist_ok=True);torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},folder/'best.pt')
            write(folder/'progress.json',dict(hp=hp,seed=seed,history=history,best=best,complete=False));log('VALIDATION',method,step,v)
    folder.mkdir(parents=True,exist_ok=True)
    if selection:
        result=dict(hp=hp,seed=seed,history=history,best=best,complete=True,budget_edge=best['step']==steps)
        write(folder/'trial.json',result);return result
    torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},folder/f'{seed}.pt')
    write(folder/f'{seed}.json',dict(hp=hp,seed=seed,steps=steps,final_loss=total,checkpoint_sha256=F.sha(folder/f'{seed}.pt')))
    return m.eval()


def select():
    guard();c=verify();xs,ys=inputs(c);tr,val=make_tasks(c,xs,ys)
    for method in METHODS:
        trials=[]
        for j,hp in enumerate(c['grid']):
            folder=OUT/'selection'/method/f'trial{j}'
            r=read(folder/'trial.json') if (folder/'trial.json').exists() else optimize(method,hp,c['selection_seed'],max(c['checkpoints']),c,tr,val,ys,folder,True)
            assert r['complete'];trials.append(r)
        k=max(range(len(trials)),key=lambda i:rank(trials[i]['best']));selected=trials[k]
        write(OUT/'selection'/f'{method}.json',dict(trial=k,hp=selected['hp'],steps=selected['best']['step'],validation=selected['best'],budget_edge=selected['budget_edge'],target_labels_read=False))
        log('SELECTED',method,selected['hp'],selected['best'])
    write(OUT/'selection_freeze.json',{str(p.relative_to(OUT)):F.sha(p) for p in (OUT/'selection').rglob('*') if p.is_file()})


def check_manifest(name):
    for rel,h in read(OUT/name).items():assert F.sha(OUT/rel)==h,rel


def fit():
    guard();c=verify();check_manifest('selection_freeze.json');xs,ys=inputs(c);tr,val=make_tasks(c,xs,ys)
    for method in METHODS:
        sel=read(OUT/'selection'/f'{method}.json')
        for seed in c['seeds']:
            folder=OUT/'weights'/method
            if (folder/f'{seed}.pt').exists():continue
            optimize(method,sel['hp'],seed,sel['steps'],c,tr,val,ys,folder,False)
            log('FITTED',method,seed)
    write(OUT/'weights_freeze.json',{str(p.relative_to(OUT)):F.sha(p) for p in (OUT/'weights').rglob('*') if p.is_file()})


def infer():
    guard();c=verify();check_manifest('weights_freeze.json');xs,ys=inputs(c);replay={};payload={}
    for method in METHODS:
        payload[method]=[]
        for seed in c['seeds']:
            m=MultiLabelModernNCA(41,27,seed,c['dim']).to(DEVICE)
            m.load_state_dict(torch.load(OUT/'weights'/method/f'{seed}.pt',map_location=DEVICE,weights_only=True));m.eval();pp={}
            for d,x in xs.items():
                ids=np.arange(x.n) if d in TARGETS else ys[d]['valid_ids'];t=task(x,ids);p=predict(m,t,method)
                if d in TARGETS:assert np.array_equal(p,predict(m,t,method)),(method,seed,d)
                if method=='categorical':
                    native=LayerNormModernNCA(41,27,seed,c['dim']).to(DEVICE).eval()
                    native.load_state_dict({k:v for k,v in m.state_dict().items() if k!='binary_bias'})
                    with torch.no_grad():original=native(device_batch(t,slice(0,min(64,len(ids))))).cpu().numpy()
                    np.testing.assert_allclose(p[:len(original)],original,atol=2e-6,rtol=2e-6)
                pp[d]=p
            dest=OUT/'probabilities'/method/f'{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**pp)
            payload[method].append(pp);replay[f'{method}/{seed}']=dict(target_predictions_bitwise_replayed=True,native_categorical_checked=method=='categorical');log('INFERRED',method,seed)
    for method in METHODS:
        mean={d:np.stack([p[d] for p in payload[method]]).mean(0) for d in xs}
        threshold=1.
        if method=='multilabel':
            cal=exact_thresholds([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']])
            write(OUT/'source_calibration.json',cal);threshold=cal['selections']['balanced_exact']['threshold']
        for label,pp in [('ensemble',mean)]+list(zip(map(str,c['seeds']),payload[method])):
            for d in TARGETS:
                meta=read(CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
                policy=read(ROOT/'round08_reference_retraining/policies'/f'{d}.json');graph=read(OLD/'source_graphs'/f'{d}.json')
                raw=[];adjusted=[]
                for q,site in enumerate(meta['sites']):
                    p=np.asarray(pp[d][q],np.float64)
                    selected=np.flatnonzero(p>=threshold).tolist() if method=='multilabel' else [int(p.argmax())]
                    if not selected:selected=[int(p.argmax())]
                    base=dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'])
                    raw.append(dict(base,after=sorted(ids[j] for j in selected)))
                    positive=[ids[j] for j in meta['known'].get(str(q),[])]
                    decision=predict_sets(p,threshold if method=='multilabel' else 2.,ids,policy,graph,site,positive)
                    adjusted.append(dict(base,after=decision['predicted_school_ids'],decision=decision))
                adjusted=ReviewedSupplements(d).apply(adjusted)
                write(OUT/'predictions'/f'{method}_raw'/label/f'{d}.json',raw)
                write(OUT/'predictions'/f'{method}_policy'/label/f'{d}.json',adjusted)
    write(OUT/'inference_audit.json',replay)
    files=[p for folder in ['probabilities','predictions'] for p in (OUT/folder).rglob('*') if p.is_file()]+[OUT/'source_calibration.json',OUT/'inference_audit.json']
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):F.sha(p) for p in files});log('PREDICTIONS SEALED')


def metric(rows):
    n=len(rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows);tp=sum(r['tp'] for r in rows)
    return dict(n=n,correct=sum(r['correct'] for r in rows),exact=100*np.mean([r['correct'] for r in rows]),fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)


def score():
    c=verify();check_manifest('prediction_freeze.json')
    base=REPO/'outputs/shenzhen_aligned_model_20261111'
    for d,h in read(base/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in read(base/'scored_cases.json')}
    xy={}
    for d in TARGETS:
        dist=pkl(REPO/'data/深圳市/修复实验输入_20261112'/d/'district.pkl')
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    rows={};metrics={};seeds={}
    groups={'全部':lambda r:True,**{d:(lambda r,d=d:r['d']==d) for d in TARGETS},
            '单校':lambda r:r['size']==1,'多校':lambda r:r['size']>1,
            '语义待核':lambda r:r['semantic_pending'],'其余':lambda r:not r['semantic_pending']}
    for arm in ARMS:
        seeds[arm]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rr=[];unscored=0
            for d in TARGETS:
                pred=read(OUT/'predictions'/arm/label/f'{d}.json')
                assert len({r['site_id'] for r in pred})==len(pred)
                assert {r['site_id'] for r in pred}=={k[1] for k in gold if k[0]==d}
                for r in pred:
                    key=(d,r['site_id']);g=gold[key]
                    if not g['has_map_gold']:unscored+=1;continue
                    truth=set(g['gold']);p=set(r['after']);pt=xy[key]
                    rr.append(dict(d=d,site_id=key[1],name=r['building_name'],truth=sorted(truth),prediction=sorted(p),
                        correct=p==truth,size=len(truth),tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),semantic_pending=g['semantic_pending'],
                        block=str((int(pt[0]//1000),int(pt[1]//1000)))))
            assert len(rr)==1761 and unscored==41
            seeds[arm][label]={k:metric([r for r in rr if fn(r)]) for k,fn in groups.items()}
            if label=='ensemble':rows[arm]=rr;metrics[arm]=seeds[arm][label];write(OUT/'scored_rows'/f'{arm}.json',rr)
    comparisons={}
    for a,b in [('multilabel_policy','categorical_policy'),('multilabel_raw','categorical_raw'),('multilabel_policy','multilabel_raw'),('categorical_policy','categorical_raw')]:
        assert [(r['d'],r['site_id']) for r in rows[a]]==[(r['d'],r['site_id']) for r in rows[b]]
        comparisons[f'{a} minus {b}']={}
        for g,fn in groups.items():
            ix=[i for i,r in enumerate(rows[a]) if fn(r)];delta=np.array([int(rows[a][i]['correct'])-int(rows[b][i]['correct']) for i in ix])
            comparisons[f'{a} minus {b}'][g]=dict(delta_pp=100*float(delta.mean()),corrected=int((delta>0).sum()),harmed=int((delta<0).sum()),
                ci95=evaluate.bootstrap([rows[a][i] for i in ix],delta,20261104,3000))
    incumbent={}
    for d in TARGETS:
        for r in read(CURRENT/'predictions/current/ensemble'/f'{d}.json'):incumbent[d,r['site_id']]=set(r['after'])
    historical={}
    for arm in ['categorical_policy','multilabel_policy']:
        rr=rows[arm];delta=[int(r['correct'])-int(incumbent[r['d'],r['site_id']]==set(r['truth'])) for r in rr]
        assert sum(incumbent[r['d'],r['site_id']]==set(r['truth']) for r in rr)==959
        historical[arm]=dict(delta_pp=100*float(np.mean(delta)),corrected=delta.count(1),harmed=delta.count(-1),
            ci95=evaluate.bootstrap(rr,delta,20261104,3000),matched_training=False)
    result=dict(metrics=metrics,comparisons=comparisons,seeds=seeds,historical_incumbent=historical,adopted=False,independent_validation=False)
    write(OUT/'results.json',result);write(DOC/'结果.json',result)
    log('RESULTS',metrics,comparisons['multilabel_policy minus categorical_policy']['全部'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','select','fit','infer','score']);a=ap.parse_args()
    globals()[a.phase]()
