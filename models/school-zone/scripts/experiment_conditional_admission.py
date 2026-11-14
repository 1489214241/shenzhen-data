"""Frozen Modern ensemble + school-shared conditional admission, source selected."""
import argparse
from pathlib import Path
import numpy as np
import torch
import continue_modernnca_evidence_v2 as C
from zoning.conditional_admission import Admission,partial_loss,combine,features

C.install();E=C.E;M=E.M
OUT=M.REPO/'outputs/modern_conditional_admission_20261113'
DOC=M.REPO.parent/'科研推进/深圳多校继续调研_20261113'
ARMS=['control_bce','control_zlpr','routed_bce','routed_zlpr']
write,read,sha=E.write,E.read,E.sha

def verify():
    E.verify();M.verify()
    for p,h in read(OUT/'freeze.json').items():assert sha(p)==h,p
    return read(OUT/'protocol.json')

def prepare():
    M.guard()
    if OUT.exists():raise FileExistsError('Do not overwrite sealed run')
    c=E.verify();M.verify();OUT.mkdir(parents=True)
    config=dict(sources=c['sources'],targets=c['targets'],seeds=c['seeds'],selection_seed=c['seeds'][0],
        arms=ARMS,grid=[.001,.003],checkpoints=[0,40,80,160,320],batch=128,
        primary='routed_zlpr_legacy minus control_base_legacy',
        hypothesis='Freeze native teacher first school; separately learn shared conditional membership for every other school, without prefix/cardinality restriction.',
        feature='Native ensemble mean softmax; candidate anchor, top1 anchor, difference, all self-masked reference feature means, relative mass, confidence, entropy, support count. No school ID.',
        backbone='Three frozen categorical Modern teacher seeds; learned admissions have three initialization seeds. No teacher weight update.',
        supervision='Identical routed textual partial labels in both input arms. Preserve source full/partial and original per-property single/multi weights; unknown excluded. Mandatory native top1 excluded from admission loss.',
        loss='Observed BCE versus observed-only adaptation of ZLPR eq1. Not official paper reproduction; missingness not assumed random.',
        selection='Same1246 source rows; singleton loss <=0.5pp, at least1 multi correct, maximize exact then multi then earliest. If none passes, no additions. No target selection.',
        threshold='Exact transition enumeration; native top1 is assigned admission1. Every checkpoint/source ensemble threshold source-only.',
        decoding='Keep previous legacy rules for primary, same in both sides. Separately raw, verified-only ablations. University member suppression unchanged in legacy. Rules/estimates are not promoted to supervision.',
        input_versions=['control','routed'],source_validation_previously_used=True,target_previous_results_seen=True,
        independent_validation=False,adopted=False,target_gold_read=False,
        limits='Frozen top1 cannot fix top1 errors; source complete multisets are only pairs; no claim of full AST encoding or enrollment eligibility.')
    write(OUT/'protocol.json',config)
    links={str(E.OUT/'protocol.json'):sha(E.OUT/'protocol.json'),str(OUT/'protocol.json'):sha(OUT/'protocol.json')}
    for p in [Path(__file__),M.REPO/'zoning/conditional_admission.py',M.REPO/'tests/test_conditional_admission.py']:
        links[str(p)]=sha(p)
    for d in c['sources']+c['targets']:
        y=dict(np.load(E.INPUT/d/'routed_labels.npz'))
        for arm in ['control','routed']:
            x=M.pkl(E.INPUT/d/f'{arm}.pkl'); probs=[]
            for seed in c['seeds']:
                model=E.teacher(c,seed).eval()
                probs.append(M.predict(model,M.task(x,np.arange(x.n)),'categorical'))
            p=np.stack(probs).mean(0)
            f=np.concatenate([features(x,p,lo,min(lo+32,x.n)) for lo in range(0,x.n,32)])
            yy=y['labels'].copy();yy[np.arange(len(yy)),p.argmax(1)]=-1
            w=np.load(E.TEACHER/f'{d}_weights.npy').copy()
            if d in c['targets']:
                unit=w.max();w[:]=0
                ids=y['train_ids'];_,inv,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True)
                w[ids]=unit/count[inv]
            w[~(yy>=0).any(1)]=0
            dest=OUT/'cache'/arm/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,feature=f,prob=p,labels=yy,original_labels=y['labels'],
                train_ids=y['train_ids'],valid_ids=y['valid_ids'],valid_labels=y['valid_labels'],weight=w)
            links[str(dest)]=sha(dest)
            M.log('ADMISSION CACHE',arm,d,f.shape)
    write(OUT/'freeze.json',links)
    M.log('ADMISSION PROTOCOL AND INPUTS SEALED')

def data(c,input_arm):return {d:dict(np.load(OUT/'cache'/input_arm/f'{d}.npz')) for d in c['sources']+c['targets']}

def scaling(xs):
    total=0;su=sq=None
    for x in xs.values():
        f=x['feature'][x['train_ids']].reshape(-1,x['feature'].shape[-1]).astype(np.float64)
        if not len(f):continue
        if su is None:su=f.sum(0);sq=(f*f).sum(0)
        else:su+=f.sum(0);sq+=(f*f).sum(0)
        total+=len(f)
    mean=su/total;std=np.maximum(np.sqrt(np.maximum(sq/total-mean**2,0)),.05)
    return mean.astype(np.float32),std.astype(np.float32)

def scores(model,x,mean,std):
    out=[];model.eval()
    with torch.no_grad():
        for lo in range(0,len(x['feature']),128):
            f=torch.from_numpy(np.clip((x['feature'][lo:lo+128]-mean)/std,-20,20)).to(M.DEVICE)
            out.append(model(f).sigmoid().cpu().numpy())
    return combine(x['prob'],np.concatenate(out))

def calibration(c,pp,xs):
    probs=[pp[d][xs[d]['valid_ids']] for d in c['sources']]
    labels=[xs[d]['valid_labels'] for d in c['sources']]
    native=sum(int((x['prob'][x['valid_ids']].argmax(1)[x['valid_labels'].sum(1)==1]==
        x['valid_labels'].argmax(1)[x['valid_labels'].sum(1)==1]).sum()) for d,x in xs.items() if d in c['sources'])
    cal=E.exact_thresholds(probs,labels);floor=native-.005*cal['source_single']
    ok=[r for r in cal['candidates'] if r['single_correct']>=floor and r['multi_correct']>0]
    if ok:
        r=max(ok,key=lambda a:(a['single_correct']+a['multi_correct'],a['multi_correct'],a['threshold']));t=r['threshold']
    else:t=1.
    return dict(**E.threshold_metrics(probs,labels,t),threshold=t,passed=bool(ok),native_single=native,floor=floor)

def rank(r):return r['passed'],r['correct'],r['multi_correct'],-r['step']

def fit_one(c,arm,lr,seed,steps,xs,mean,std,folder,select):
    model=Admission(len(mean),seed).to(M.DEVICE);opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=.001)
    kind=arm.split('_')[1];rng=np.random.default_rng(seed)
    total=sum(x['weight'][x['train_ids']].sum() for x in xs.values())
    tr={}
    for d,x in xs.items():
        ids=x['train_ids'];ids=ids[x['weight'][ids]>0]
        if not len(ids):continue
        tr[d]=(torch.from_numpy(np.clip((x['feature'][ids]-mean)/std,-20,20)).to(M.DEVICE),
               torch.from_numpy(x['labels'][ids]).to(M.DEVICE),
               torch.from_numpy((x['weight'][ids]/total).astype(np.float32)).to(M.DEVICE))
    hist=[];best=None
    for step in range(steps+1):
        if step:
            model.train();opt.zero_grad()
            for f,y,w in tr.values():
                ix=rng.choice(len(f),min(c['batch'],len(f)),replace=False)
                loss=partial_loss(model(f[ix]),y[ix],w[ix],kind)*len(f)/len(ix)
                assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if select and step in c['checkpoints']:
            pp={d:scores(model,x,mean,std) for d,x in xs.items() if d in c['sources']}
            r=calibration(c,pp,xs);r['step']=step;hist.append(r)
            if best is None or rank(r)>rank(best):
                best=r;folder.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),folder/'best.pt')
            M.log('ADMISSION SOURCE',arm,lr,r)
    folder.mkdir(parents=True,exist_ok=True)
    if select:
        r=dict(lr=lr,best=best,history=hist);write(folder/'trial.json',r);return r
    torch.save(model.state_dict(),folder/f'{seed}.pt')

def fit():
    M.guard();c=verify()
    for input_arm in ['control','routed']:
        xs=data(c,input_arm);mean,std=scaling(xs)
        p=OUT/'scaling'/f'{input_arm}.npz';p.parent.mkdir(exist_ok=True);np.savez(p,mean=mean,std=std)
        for kind in ['bce','zlpr']:
            arm=input_arm+'_'+kind;trials=[]
            for j,lr in enumerate(c['grid']):
                folder=OUT/'selection'/arm/f'trial{j}'
                trials.append(read(folder/'trial.json') if (folder/'trial.json').exists() else
                    fit_one(c,arm,lr,c['selection_seed'],max(c['checkpoints']),xs,mean,std,folder,True))
            j=max(range(len(trials)),key=lambda j:rank(trials[j]['best']))
            selected=dict(trial=j,**trials[j]);write(OUT/'selection'/f'{arm}.json',selected)
            for seed in c['seeds']:
                folder=OUT/'weights'/arm
                if not (folder/f'{seed}.pt').exists():fit_one(c,arm,selected['lr'],seed,selected['best']['step'],xs,mean,std,folder,False)
    write(OUT/'weights_freeze.json',{str(p.relative_to(OUT)):sha(p) for f in ['selection','weights','scaling'] for p in (OUT/f).rglob('*') if p.is_file()})

def check(name):
    for p,h in read(OUT/name).items():assert sha(OUT/p)==h,p

def decode(d,p,q,threshold):
    meta=read(E.INPUT/d/'metadata.json');cons=read(E.INPUT/d/'constraints.json')
    ids=[s['school_id'] for s in meta['schools']]
    policy=read(M.ROOT/'round08_reference_retraining/policies'/f'{d}.json')
    graph=read(M.OLD/'source_graphs'/f'{d}.json')
    rr={k:[] for k in ['raw','legacy','verified']}
    for i,site in enumerate(meta['sites']):
        raw=sorted(ids[j] for j in np.flatnonzero(q[i]>=threshold))
        assert ids[int(p[i].argmax())] in raw
        base=dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],
            complete_school_set=False,enrollment_eligibility=False)
        rr['raw'].append(dict(base,after=raw))
        con=cons[base['site_id']]
        rr['verified'].append(dict(base,after=E.apply_verified_constraints(raw,con['positive'],con['negative'])))
        # Membership is selected by the new learner, but university base selection
        # must retain the native school's ordering, not the admission score ordering.
        selected=np.array([sid in raw for sid in ids])
        legal_scores=np.where(selected,1.+p[i],p[i])
        known=[ids[j] for j in meta['known'].get(str(i),[])]
        r=E.predict_sets(legal_scores,1.,ids,policy,graph,site,known)
        rr['legacy'].append(dict(base,after=r['predicted_school_ids'],decision=r))
    rr['legacy']=E.ReviewedSupplements(d).apply(rr['legacy'])
    return rr

def infer():
    M.guard();c=verify();check('weights_freeze.json');cal={}
    for input_arm in ['control','routed']:
        xs=data(c,input_arm);scale=dict(np.load(OUT/'scaling'/f'{input_arm}.npz'))
        # Top1-only and global relative-threshold controls use exactly same inputs.
        relative={d:combine(x['prob'],x['prob']/x['prob'].max(1,keepdims=True)) for d,x in xs.items()}
        rc=calibration(c,relative,xs);cal[input_arm+'_relative']=rc
        for method,pp,t in [('base',{d:combine(x['prob'],np.zeros_like(x['prob'])) for d,x in xs.items()},1.),
                            ('relative',relative,rc['threshold'])]:
            for d in c['targets']:
                for decoder,rr in decode(d,xs[d]['prob'],pp[d],t).items():write(OUT/'predictions'/f'{input_arm}_{method}_{decoder}'/'ensemble'/f'{d}.json',rr)
        for kind in ['bce','zlpr']:
            arm=input_arm+'_'+kind;payload=[]
            for seed in c['seeds']:
                model=Admission(len(scale['mean']),seed).to(M.DEVICE)
                model.load_state_dict(torch.load(OUT/'weights'/arm/f'{seed}.pt',map_location=M.DEVICE,weights_only=True))
                pp={d:scores(model,x,**scale) for d,x in xs.items()};payload.append(pp)
                p=OUT/'probabilities'/arm/f'{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pp)
            mean={d:np.stack([p[d] for p in payload]).mean(0) for d in xs}
            ca=calibration(c,mean,xs);cal[arm]=ca;M.log('ADMISSION FINAL SOURCE',arm,ca)
            for label,pp in [('ensemble',mean)]+list(zip(map(str,c['seeds']),payload)):
                for d in c['targets']:
                    for decoder,rr in decode(d,xs[d]['prob'],pp[d],ca['threshold']).items():
                        write(OUT/'predictions'/f'{arm}_{decoder}'/label/f'{d}.json',rr)
    write(OUT/'source_calibration.json',cal)
    files=[p for f in ['predictions','probabilities'] for p in (OUT/f).rglob('*') if p.is_file()]+[OUT/'source_calibration.json']
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    M.log('ADMISSION PREDICTIONS SEALED')

def score():
    c=verify();check('prediction_freeze.json')
    gold=read(M.REPO/'outputs/modernnca_partial_sets_20261113/scored_rows/categorical_policy.json')
    truth={(r['d'],r['site_id']):r for r in gold};rows={};metrics={};seed_metrics={}
    groups={'全部':lambda r:True,'单校':lambda r:r['size']==1,'多校':lambda r:r['size']>1,
        '三校及以上':lambda r:r['size']>=3,'语义待核':lambda r:r['semantic_pending'],
        **{d:lambda r,d=d:r['d']==d for d in c['targets']}}
    for folder in sorted((OUT/'predictions').iterdir()):
        arm=folder.name;seed_metrics[arm]={}
        for label in folder.iterdir():
            rr=[];outside=0
            for d in c['targets']:
                preds=read(label/f'{d}.json');meta=read(E.INPUT/d/'metadata.json')
                assert len(preds)==len(meta['sites']) and {r['site_id'] for r in preds}=={s['coordinate_group_id'] for s in meta['sites']}
                for r in preds:
                    key=d,r['site_id']
                    if key not in truth:outside+=1;continue
                    g=truth[key];p=set(r['after']);t=set(g['truth'])
                    rr.append(dict(g,prediction=sorted(p),correct=p==t,tp=len(p&t),fp=len(p-t),fn=len(t-p)))
            assert len(rr)==1761 and outside==41 and sum(r['semantic_pending'] for r in rr)==242
            mm={k:M.metric([r for r in rr if f(r)]) for k,f in groups.items()};seed_metrics[arm][label.name]=mm
            if label.name=='ensemble':rows[arm]=rr;metrics[arm]=mm;write(OUT/'scored_rows'/f'{arm}.json',rr)
    baseline={(r['d'],r['site_id']):r for r in rows['control_base_legacy']}
    assert all(r['prediction']==truth[r['d'],r['site_id']]['prediction'] for r in baseline.values()),'Native decoder replay mismatch'
    comparisons={}
    for arm in ['routed_zlpr_legacy','routed_bce_legacy','control_zlpr_legacy','control_bce_legacy','control_relative_legacy']:
        comparisons[arm]={}
        for k,f in groups.items():
            rr=[r for r in rows[arm] if f(r)];delta=[int(r['correct'])-int(baseline[r['d'],r['site_id']]['correct']) for r in rr]
            comparisons[arm][k]=dict(delta_pp=100*float(np.mean(delta)),corrected=delta.count(1),harmed=delta.count(-1),ci95=M.evaluate.bootstrap(rr,delta,20261104,3000))
    hashes=read(M.REPO/'outputs/modernnca_partial_sets_20261113/results.json')['gold_sha256']
    for d,h in hashes.items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    r=dict(metrics=metrics,seeds=seed_metrics,comparisons=comparisons,source=read(OUT/'source_calibration.json'),
        primary=c['primary'],adopted=False,independent_validation=False,gold_sha256=hashes)
    write(OUT/'results.json',r);M.log('ADMISSION RESULTS',{a:m['全部']['correct'] for a,m in metrics.items()},comparisons['routed_zlpr_legacy']['全部'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','fit','infer','score'])
    globals()[ap.parse_args().phase]()
