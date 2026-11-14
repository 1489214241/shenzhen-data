"""Budgeted complete-reference experiment: plan -> release -> fit -> score.

Standalone development experiment. Never changes production defaults or Gold.
Selected Gold is a simulated verification oracle, not acquired official facts.
"""
import argparse, copy, hashlib, json, os, pickle, re, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as TF

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from zoning import DEVICE, evidence as EV, model as MD
from zoning.modernnca_complete_reference import coverage_select,catalogue,complete_targets,CompleteReferenceModernNCA
from zoning.modernnca_multilabel import categorical_positive_loss
from shenzhen_partial_model import predict_sets
from shenzhen_reviewed_supplements import ReviewedSupplements

OUT=REPO/'outputs/modern_complete_reference_20261114'
PARENT=REPO/'outputs/modernnca_shenzhen_multilabel_20261113'
ROOT=REPO/'outputs/shenzhen_resolution_20261112'
CURRENT=ROOT/'community_methods_20261113'
GOLD=REPO/'outputs/shenzhen_aligned_model_20261111/scored_cases.json'
DISTRICTS=['龙华区','龙岗区'];SEEDS=[20261231,20261232,20261233]
START=time.time()
def log(*args): print(round(time.time()-START,1),*args,flush=True)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pkl(p):
    with open(p,'rb') as f:return pickle.load(f)
def savep(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'wb') as f:pickle.dump(v,f,protocol=5)
def group_name(name):
    s=re.sub(r'[\s·•（）()\-]','',name)
    s=re.sub(r'(第?[一二三四五六七八九十\d]+期|[一二三四五六七八九十\d]+[栋幢座]|[A-HＡ-Ｈ]区)$','',s)
    return s or name
def group_ids(meta):
    # Connected components of public building IDs and normalized public aliases.
    n=len(meta['sites']);parent=list(range(n));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,s in enumerate(meta['sites']):
        keys=['id:'+v for v in s.get('coordinate_group_building_ids',[])]
        keys+=['name:'+group_name(v) for v in s.get('coordinate_group_building_names',[s['building_name']])]
        for k in keys:
            if k in seen:parent[root(i)]=root(seen[k])
            else:seen[k]=i
    return np.array([str(root(i)) for i in range(n)])
def input_paths(d):
    b=REPO/'data/深圳市/修复实验输入_20261112'/d
    return [b/'district.pkl',b/'rules.pkl',CURRENT/'inputs/current'/f'{d}.pkl',CURRENT/'metadata'/f'{d}.json']
def guard():
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/')
            if any(s in p for s in ['/Gold/','/scored_cases.json','/scored_rows/','/polygons.json','/results.json']):
                raise PermissionError('Unreleased evaluation answers forbidden: '+p)
    sys.addaudithook(hook)
def verify():
    c=read(OUT/'protocol.json')
    for p,h in read(OUT/'inputs_sha256.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'code_sha256.json').items():assert sha(REPO/p)==h,p
    return c

def plan():
    if OUT.exists():raise FileExistsError('Preserve existing experiment')
    # Existing cohort membership only; no answer values used for point selection.
    cohort={(r['district'],r['site_id']) for r in read(GOLD) if r['has_map_gold']}
    c=dict(date='2026-11-14',status='development only; not adopted',seeds=SEEDS,
        budget_per_district=256,selection_seed=20261231,coverage_scale_m=500,
        selection='Greedy facility location, sum(exp(-distance/500m)); coordinates only; existing1761map-hit cohort fixed before selection.',
        oracle='Selected Gold simulates complete verification. No claim of actual512manual verifications or real eligibility truth.',
        exclusion='Public building-ID/normalized-name connected components. Own entire group excluded from every reference path, context and training-validation split.',
        validation='20percent of selected public groups, label-blind seed split; these complete labels never enter tuning reference pool. All query labels sealed until prediction freeze.',
        model='Existing ModernNCA-LN categorical weights; all original road/policy/community/school features retained; new complete refs encoded with same41pair features. Full-set branch aggregates same token affinities by complete reference set, plus nearest-set spatial prior.',
        incomplete='Old partial refs retain observed positives only. No new negative labels inferred from partial positives. Complete selected support labels only supply finetuning targets.',
        arms=['archived_current_context','nearest_raw','nearest_policy','augmented_frozen_raw','augmented_frozen_policy','categorical_tuned_raw','categorical_tuned_policy','complete_tuned_raw','complete_tuned_policy'],
        primary='complete_tuned_policy vs nearest_policy and categorical_tuned_policy; raw arms secondary. Archived current is contextual only, not same isolation/input control.',
        fit=dict(lr=.0003,reg=.001,checkpoints=[0,20,40,80],batch=32,clip=5,geo_scale_m=300),
        hyperparameters='Single predefined lr/reg; select checkpoint by complete-set exact on released support validation, then fewerFP, earlier step. Three initial checkpoints selected independently; no query-label tuning.',
        candidate_sets='Full sets observed in tuning/final support pool plus singletons; only sets with legal complete reference can be selected by full-set branch. Unseen query combinations count as errors.',
        policy='Same existing policy decoder for every policy arm; no university member expansion; original partial positives and reviewed supplements preserved. Raw complete sets estimate frozen map membership, not enrollment entitlement.',
        scoring='Same unknown1249points for all primary methods; exact fullset, districts, single/multi, FP/FN, corrected/harmed and paired district1km bootstrap2000.',
        limitations=['New local selection differs from external report; not reproduction of75.74percent.','Pretrained models have seen legal partial policy facts from target area.','No unseen-city/time validation; target geography already explored.','Name grouping may conservatively merge phases; coordinates alone do not prove common catchment.'])
    write(OUT/'protocol.json',c);paths=[];public={};audit={}
    for d in DISTRICTS:
        dist=pkl(input_paths(d)[0]);meta=read(input_paths(d)[3]);groups=group_ids(meta)
        eligible=np.array([i for i,s in enumerate(dist.site_ids) if (d,s) in cohort])
        chosen=eligible[coverage_select(dist.coords[eligible],256,c['selection_seed'])]
        rng=np.random.default_rng(c['selection_seed']);ugs=np.unique(groups[chosen]);rng.shuffle(ugs)
        valgroups=set(ugs[:max(1,round(.2*len(ugs)))])
        rows=[dict(q=int(q),site_id=dist.site_ids[q],name=dist.site_names[q],xy=dist.coords[q].tolist(),group=groups[q],
                   selected=bool(q in chosen),validation=bool(q in chosen and groups[q] in valgroups)) for q in eligible]
        public[d]=dict(rows=rows,all_groups=groups.tolist(),selected_order=chosen.tolist(),crs=dist.crs)
        audit[d]=dict(cohort=len(eligible),selected=len(chosen),unknown=len(eligible)-len(chosen),
                      validation=sum(r['validation'] for r in rows),public_groups=len(set(groups)),coordinate_rows=len(groups))
        paths+=input_paths(d)
        paths += [ROOT/'round08_reference_retraining/policies'/f'{d}.json',ROOT/'round03_partial_labels/source_graphs'/f'{d}.json']
        paths += list(map(Path,ReviewedSupplements(d).hashes))
    paths+=[PARENT/'weights/categorical'/f'{s}.pt' for s in SEEDS]
    paths+=[REPO/'data/lines/深圳_完整输入_20261111.pkl']
    write(OUT/'public_plan.json',public);write(OUT/'plan_audit.json',audit)
    write(OUT/'inputs_sha256.json',{str(p):sha(p) for p in paths})
    code=[Path(__file__),REPO/'zoning/modernnca_complete_reference.py',REPO/'tests/test_modernnca_complete_reference.py',REPO/'scripts/audit_modern_complete_reference.py']
    code += [REPO/'zoning'/f for f in ['modernnca_multilabel.py','full_evidence.py','modernnca_normalization.py','evidence.py','model.py']]
    write(OUT/'code_sha256.json',{str(p.relative_to(REPO)):sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'plan_freeze.json',{n:sha(OUT/n) for n in ['protocol.json','public_plan.json','inputs_sha256.json','code_sha256.json']})
    log('PLAN',audit)

def release():
    verify()
    for p,h in read(OUT/'plan_freeze.json').items():assert sha(OUT/p)==h
    public=read(OUT/'public_plan.json');gold={(r['district'],r['site_id']):r for r in read(GOLD)}
    support={}
    for d,v in public.items():
        support[d]=[dict(r,schools=gold[d,r['site_id']]['gold'],complete=True,source='simulated_gold_oracle') for r in v['rows'] if r['selected']]
        assert len(support[d])==256
    write(OUT/'released_support.json',support)
    write(OUT/'release_receipt.json',dict(plan_sha=sha(OUT/'public_plan.json'),support_sha=sha(OUT/'released_support.json'),gold_sha=sha(GOLD),released_points=512))
    log('RELEASED512 ONLY')

def augmented(d,support,groups):
    groups=np.asarray(groups)
    dist=pkl(input_paths(d)[0]);rules=pkl(input_paths(d)[1]);old=pkl(input_paths(d)[2]);meta=read(input_paths(d)[3])
    ids=[s['school_id'] for s in meta['schools']];by={s:i for i,s in enumerate(ids)}
    complete={r['q']:tuple(sorted(by[s] for s in r['schools'])) for r in support}
    ri=np.array([q for q,ss in complete.items() for s in ss],int)
    rs=np.array([s for q,ss in complete.items() for s in ss],int)
    st=rules.statuses;qst=st[:,rs];rst=st[ri,rs]
    ls=EV.lines(dist.lines_key);unique,inv=np.unique(ri,return_inverse=True)
    cross,within=EV.crossing(ls,dist.coords,dist.coords[unique])
    cn=dist.community_name;strict=dist.community_strict
    same=(cn[:,None]==cn[ri][None])&(cn[:,None]!='');both=strict[:,None]&strict[ri][None]
    # Existing known-only boundary evidence; selected Gold never enters this feature.
    original_known={int(q):ss for q,ss in meta['known'].items()}
    blocks=dict(distance=EV.geometry(dist.coords[ri][None]-dist.coords[:,None],300),
        named_reference=np.zeros((dist.n,len(ri),1),np.float32),policy=EV.policy(qst,rules.negative[:,rs]),
        agreement=((qst>=0)&(rst[None]>=0)&(qst==rst[None])).astype('float32'),
        disagreement=((qst>=0)&(rst[None]>=0)&(qst!=rst[None])).astype('float32'),
        crossing=cross[:,inv],within_2km=within[:,inv],
        learned_boundary=EV.learned_boundary(ls,dist.coords,groups,original_known,ri),
        same_community=np.stack([same&both,(~same)&both],-1).astype('float32'))
    pair,_=EV._stack(blocks,list(old.pair_blocks))
    keep=~np.isin(old.ref_sites,list(complete));x=copy.copy(old)
    oldpair=old.pair[:,keep].copy()
    oldpair[:,:,old.pair_blocks['learned_boundary']]=EV.learned_boundary(ls,dist.coords,groups,original_known,old.ref_sites[keep])
    x.pair=np.concatenate([oldpair,pair],1)
    x.ref_sites=np.r_[old.ref_sites[keep],ri];x.ref_schools=np.r_[old.ref_schools[keep],rs]
    x.groups=np.asarray(groups);rg=x.groups[x.ref_sites]
    _,inv,cnt=np.unique(rg,return_inverse=True,return_counts=True)
    x.group_bias=-np.log(cnt[inv]).astype('float32');x.self_mask=x.groups[:,None]==rg[None]
    x.membership=TF.one_hot(torch.tensor(np.r_[x.ref_schools,np.arange(x.school_n)],dtype=torch.long),x.school_n).float()
    cats=catalogue(complete.values(),x.school_n);ci={s:i for i,s in enumerate(cats)}
    sm=np.zeros((len(x.ref_sites)+x.school_n,len(cats)),np.float32)
    # Only explicitly complete refs contribute to a multi-school set class.
    for j,(q,s) in enumerate(zip(x.ref_sites,x.ref_schools)):sm[j,ci[complete[q] if q in complete else (int(s),)]]=1
    for s in range(x.school_n):sm[len(x.ref_sites)+s,ci[(s,)]]=1
    geo=np.full((x.n,len(cats)),-50.,np.float32);available=np.zeros_like(geo,bool)
    nearest=np.full(x.n,-1,int);nearest_distance=np.full(x.n,np.inf)
    for q,ss in complete.items():
        dd=np.linalg.norm(dist.coords-dist.coords[q],axis=1);valid=x.groups!=x.groups[q];j=ci[ss]
        geo[valid,j]=np.maximum(geo[valid,j],-dd[valid]/300.)
        available[valid,j]=True
        better=valid&(dd<nearest_distance);nearest[better]=j;nearest_distance[better]=dd[better]
    assert (nearest>=0).all()
    labels=np.full((x.n,x.school_n),-1,np.float32)
    for q,ss in complete.items():labels[q]=0;labels[q,list(ss)]=1
    return dict(x=x,cats=cats,set_mem=torch.tensor(sm),geo=geo,available=available,
                nearest=nearest,nearest_distance=nearest_distance,labels=labels,school_ids=ids,
                support_ids=np.array(list(complete)),coords=dist.coords)

def batches(data,q):
    t=MD.batch(data['x'],q)
    t.update(set_mem=data['set_mem'].to(DEVICE),geo_prior=torch.tensor(data['geo'][q],device=DEVICE),
             set_available=torch.tensor(data['available'][q],device=DEVICE))
    return t
def model(seed):
    m=CompleteReferenceModernNCA(41,27,seed).to(DEVICE)
    state=torch.load(PARENT/'weights/categorical'/f'{seed}.pt',map_location=DEVICE,weights_only=True)
    result=m.load_state_dict(state,strict=False);assert result.missing_keys==['set_log_weight'] and not result.unexpected_keys
    return m
@torch.no_grad()
def infer(m,data,ids):
    m.eval();p=[];z=[]
    for lo in range(0,len(ids),32):
        school,sets=m(batches(data,ids[lo:lo+32]));p.append(school.softmax(1).cpu().numpy());z.append(sets.softmax(1).cpu().numpy())
    return np.concatenate(p),np.concatenate(z)
def fit_one(m,method,data,train,steps,config,seed,on_checkpoint=None):
    opt=torch.optim.Adam(m.parameters(),lr=config['lr']);rng=np.random.default_rng(seed)
    targets=complete_targets(data['labels'][train],data['cats']);skipped=0
    if method=='complete':
        valid=data['available'][train,targets];skipped=int((~valid).sum());train=train[valid];targets=targets[valid]
    if not len(train):raise ValueError('No legal training rows')
    for step in range(1,steps+1):
        m.train();ix=rng.choice(len(train),min(config['batch'],len(train)),replace=False);q=train[ix]
        opt.zero_grad();school,sets=m(batches(data,q))
        labels=torch.tensor(data['labels'][q],device=DEVICE)
        if method=='categorical':loss=categorical_positive_loss(school,labels,torch.ones(len(q),device=DEVICE)/len(q))
        else:loss=TF.cross_entropy(sets,torch.tensor(targets[ix],device=DEVICE,dtype=torch.long))
        loss=loss+config['reg']*sum(p.square().mean() for p in m.parameters())
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),config['clip']);opt.step()
        if on_checkpoint and step in config['checkpoints']:on_checkpoint(step,m)
    return skipped
def decode(d,prob,sets,data,queries):
    meta=read(CURRENT/'metadata'/f'{d}.json');ids=data['school_ids']
    policy=read(ROOT/'round08_reference_retraining/policies'/f'{d}.json')
    graph=read(ROOT/'round03_partial_labels/source_graphs'/f'{d}.json');out=[]
    for j,q in enumerate(queries):
        selected=set(sets[j]);p=prob[j]
        # Reuse native university/base and direct shared-scope policy processing.
        # Preserve selected ranking within set while ensuring only selected pass.
        adjusted=np.asarray(p)*.1
        for s in selected:adjusted[s]+=1.
        positive=[ids[s] for s in meta['known'].get(str(q),[])]
        decision=predict_sets(adjusted,.5,ids,policy,graph,meta['sites'][q],positive)
        out.append(dict(district=d,site_id=meta['sites'][q]['coordinate_group_id'],building_name=meta['sites'][q]['building_name'],
                        after=decision['predicted_school_ids'],decision=decision))
    supplement=ReviewedSupplements(d);seen={r['site_id'] for r in out}
    # The inherited adapter expects a full district. Restrict its already-frozen
    # facts to this query batch without acquiring or fabricating new facts.
    for attr in ['positive','negative']:
        values=getattr(supplement,attr)
        for k in list(values):
            if k not in seen:del values[k]
    return supplement.apply(out)

def run():
    guard();c=verify();public=read(OUT/'public_plan.json');support=read(OUT/'released_support.json')
    assert sha(OUT/'released_support.json')==read(OUT/'release_receipt.json')['support_sha']
    allpred={};selection={};audits={}
    for d in DISTRICTS:
        rows=public[d]['rows'];trainrows=[r for r in support[d] if not r['validation']]
        validrows=[r for r in support[d] if r['validation']];groups=public[d]['all_groups']
        log('BUILD TUNING',d);tune=augmented(d,trainrows,groups)
        train=np.array([r['q'] for r in trainrows]);val=np.array([r['q'] for r in validrows]);
        assert not set(np.array(groups)[train])&set(np.array(groups)[val])
        vid={s:i for i,s in enumerate(tune['school_ids'])}
        truth=[{vid[s] for s in r['schools']} for r in validrows]
        selection[d]={};chosen={}
        for method in ['categorical','complete']:
            for seed in SEEDS:
                m=model(seed);trials=[]
                def checkpoint(step,m):
                    p,z=infer(m,tune,val)
                    raw=[(int(a),) for a in p.argmax(1)] if method=='categorical' else [tune['cats'][j] for j in z.argmax(1)]
                    # Select the same policy output used for the primary experiment.
                    decoded=decode(d,p,raw,tune,val)
                    predicted=[{vid[s] for s in r['after']} for r in decoded]
                    rec=dict(step=step,n=len(val),correct=sum(a==b for a,b in zip(predicted,truth)),fp=sum(len(a-b) for a,b in zip(predicted,truth)))
                    trials.append(rec);log('VALID',d,method,seed,rec)
                checkpoint(0,m)
                skipped=fit_one(m,method,tune,train,max(c['fit']['checkpoints']),c['fit'],seed,checkpoint)
                best=max(trials,key=lambda r:(r['correct'],-r['fp'],-r['step']))
                selection[d][f'{method}/{seed}']=dict(trials=trials,chosen=best,unavailable_training_targets=skipped)
                chosen[method,seed]=best['step'];del m
        del tune
        write(OUT/'selection.json',selection)
        log('BUILD FINAL',d);data=augmented(d,support[d],groups);q=np.array([r['q'] for r in rows if not r['selected']]);
        train=np.array([r['q'] for r in support[d]]);predictions={};ps=[]
        for seed in SEEDS:
            m=model(seed);p,z=infer(m,data,q);ps.append(p);del m
        frozen=np.mean(ps,0);predictions['augmented_frozen']=(frozen,[(int(i),) for i in frozen.argmax(1)])
        nearest=[data['cats'][i] for i in data['nearest'][q]]
        predictions['nearest']=(frozen,nearest)
        for method in ['categorical','complete']:
            pp=[];zz=[]
            for seed in SEEDS:
                m=model(seed);skipped=fit_one(m,method,data,train,chosen[method,seed],c['fit'],seed)
                p,z=infer(m,data,q);pp.append(p);zz.append(z)
                dest=OUT/'weights'/d/f'{method}_{seed}.pt';dest.parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),dest)
                log('FINAL',d,method,seed,'steps',chosen[method,seed],'skipped',skipped)
                del m
            p=np.mean(pp,0);z=np.mean(zz,0)
            raw=[(int(i),) for i in p.argmax(1)] if method=='categorical' else [data['cats'][i] for i in z.argmax(1)]
            predictions[f'{method}_tuned']=(p,raw)
        meta=read(CURRENT/'metadata'/f'{d}.json');district_pred={}
        for method,(p,raw) in predictions.items():
            rawrows=[dict(district=d,site_id=meta['sites'][i]['coordinate_group_id'],building_name=meta['sites'][i]['building_name'],after=sorted(data['school_ids'][s] for s in ss)) for i,ss in zip(q,raw)]
            district_pred[method+'_raw']=rawrows
            district_pred[method+'_policy']=decode(d,p,raw,data,q)
        # Existing contextual baseline is archived, never used to train/select.
        prior=REPO/'outputs/modern_shared_school_prior_20261114/predictions/baseline_current/ensemble'/f'{d}.json'
        old={r['site_id']:r for r in read(prior)}
        district_pred['archived_current_context']=[old[meta['sites'][i]['coordinate_group_id']] for i in q]
        allpred[d]=district_pred
        nearest_provenance=[]
        for i in q:
            legal=[r for r in support[d] if groups[r['q']]!=groups[i]]
            best=min(legal,key=lambda r:np.linalg.norm(data['coords'][i]-data['coords'][r['q']]))
            nearest_provenance.append(dict(query=meta['sites'][i]['coordinate_group_id'],reference=best['site_id'],
                                          distance_m=float(np.linalg.norm(data['coords'][i]-data['coords'][best['q']])),schools=best['schools']))
        write(OUT/'nearest_provenance'/f'{d}.json',nearest_provenance)
        audits[d]=dict(queries=len(q),complete_points=len(train),reference_relations=len(data['x'].ref_sites),
                       full_set_candidates=len(data['cats']),same_group_mask=int(data['x'].self_mask.sum()),
                       school_count=data['x'].school_n,all_features_retained=True)
        write(OUT/'predictions'/f'{d}.json',district_pred);write(OUT/'input_audit.json',audits)
        del data
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['predictions','nearest_provenance','weights'] for p in (OUT/folder).rglob('*') if p.is_file()})
    write(OUT/'selection_freeze.json',dict(selection=sha(OUT/'selection.json'),support=sha(OUT/'released_support.json')))
    log('PREDICTIONS SEALED')

def metric(rows):
    n=len(rows);tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return dict(n=n,correct=sum(r['correct'] for r in rows),exact=100*sum(r['correct'] for r in rows)/max(n,1),fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
def score():
    verify()
    for p,h in read(OUT/'prediction_freeze.json').items():assert sha(OUT/p)==h,p
    assert sha(GOLD)==read(OUT/'release_receipt.json')['gold_sha']
    gold={(r['district'],r['site_id']):r for r in read(GOLD)};public=read(OUT/'public_plan.json');allrows={}
    for d in DISTRICTS:
        rows={r['site_id']:r for r in public[d]['rows'] if not r['selected']}
        for arm,pred in read(OUT/'predictions'/f'{d}.json').items():
            assert set(rows)=={r['site_id'] for r in pred} and len(rows)==len(pred)
            for r in pred:
                g=gold[d,r['site_id']];truth=set(g['gold']);p=set(r['after']);xy=rows[r['site_id']]['xy']
                allrows.setdefault(arm,[]).append(dict(district=d,site_id=r['site_id'],name=r['building_name'],truth=sorted(truth),prediction=sorted(p),
                    correct=p==truth,tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),size=len(truth),semantic_pending=g['semantic_pending'],block=f'{d}/{int(xy[0]//1000)}/{int(xy[1]//1000)}'))
    groups={'all':lambda r:True,**{d:(lambda r,d=d:r['district']==d) for d in DISTRICTS},'single':lambda r:r['size']==1,'multi':lambda r:r['size']>1,'three_plus':lambda r:r['size']>=3,'semantic_pending':lambda r:r['semantic_pending']}
    metrics={a:{g:metric([r for r in rr if f(r)]) for g,f in groups.items()} for a,rr in allrows.items()}
    names=list(allrows);n=len(allrows[names[0]]);assert n==1249
    matrix=np.array([[r['correct'] for r in allrows[a]] for a in names],float)
    for a in names:assert [(r['district'],r['site_id']) for r in allrows[a]]==[(r['district'],r['site_id']) for r in allrows[names[0]]]
    rng=np.random.default_rng(20261114);boot=[]
    blocks={d:sorted({r['block'] for r in allrows[names[0]] if r['district']==d}) for d in DISTRICTS}
    bi={b:np.array([i for i,r in enumerate(allrows[names[0]]) if r['block']==b]) for bb in blocks.values() for b in bb}
    for _ in range(2000):
        ids=np.concatenate([bi[b] for d in DISTRICTS for b in rng.choice(blocks[d],len(blocks[d]),replace=True)])
        boot.append(matrix[:,ids].mean(1)*100)
    boot=np.array(boot);comparisons={}
    for baseline in ['nearest_policy','categorical_tuned_policy','augmented_frozen_policy','archived_current_context','nearest_raw']:
        arm='complete_tuned_raw' if baseline=='nearest_raw' else 'complete_tuned_policy';a=names.index(arm);b=names.index(baseline)
        comparisons[f'{arm} minus {baseline}']=dict(delta_pp=float(100*(matrix[a]-matrix[b]).mean()),
            ci95=np.quantile(boot[:,a]-boot[:,b],[.025,.975]).tolist(),corrected=int(((matrix[a]==1)&(matrix[b]==0)).sum()),harmed=int(((matrix[a]==0)&(matrix[b]==1)).sum()))
    result=dict(metrics=metrics,comparisons=comparisons,ci95={a:np.quantile(boot[:,i],[.025,.975]).tolist() for i,a in enumerate(names)},adopted=False,independent_validation=False)
    write(OUT/'results.json',result)
    for arm,rr in allrows.items():write(OUT/'scored_rows'/f'{arm}.json',rr)
    log('RESULTS',{a:round(m['all']['exact'],2) for a,m in metrics.items()});log('COMPARISONS',comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','release','run','score']);args=p.parse_args();globals()[args.phase]()
