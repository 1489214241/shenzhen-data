"""Per-school total reference targets1/2/3, existing verified groups included.

prepare: no target Gold values; sample: explicit school-label oracle; fit:
released supports only; score: after all three budgets are prediction-sealed.
"""
import argparse,collections,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as TF
import experiment_modern_complete_reference as E
from zoning.school_gap_reference import fill_school_gaps
from zoning.modernnca_complete_reference import catalogue,complete_targets
from zoning.modernnca_multilabel import categorical_positive_loss

OUT=E.REPO/'outputs/modern_school_gap_reference_20261114'
LEDGER=E.REPO/'data/深圳市/全链路输入_v2_20261113'
TARGETS=[1,2,3];read,write,sha=E.read,E.write,E.sha

def verify():
    for path,h in read(OUT/'inputs_sha256.json').items():assert sha(path)==h,path
    for rel,h in read(OUT/'code_sha256.json').items():assert sha(E.REPO/rel)==h,rel
    return read(OUT/'protocol.json')

def prepare():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    E.guard()
    c=dict(date='2026-11-14',targets=TARGETS,seeds=E.SEEDS,selection_seed=20261231,
        policy='Existing verified references count toward per-school TOTAL1/2/3; missing schools first; not extraKper school.',
        verified='Only positive textual_proof rows in frozen evidence ledger count. Distinct public building/name groups; weak estimates do not count. Partial positives remain partial, not complete sets.',
        sampling='Explicit Gold oracle simulates asking each deficient school for a reference. Schools alphabetical; fixed random candidate order; no set-cover lookahead. Full set released only for selected site. Same project selected and billed once; targets nested.',
        no_free_completion='Existing partial points are not converted to complete labels for free. If selected for another missing school they incur one full-set verification cost.',
        gaps='No candidate -> preserve deficit; no fabricated support. Gold-simulated success does not prove official/manual acquisition is possible.',
        cohort='Original1761map-hit cohort. Exclude EVERY existing model reference group, every added textual reference group and newly verified project group from scoring.',
        primary_queries='Intersection of remaining unknowns across all three targets, fixed before fitting. Also report each budget natural remaining cohort separately.',
        model='Reuse previous Modern complete-reference architecture, weights, feature blocks and training hyperparameters. Include all391verified partial relations in every arm; preserve older weak refs as estimates.',
        fit=dict(lr=.0003,reg=.001,checkpoints=[0,20,40,80],batch=32,clip=5,geo_scale_m=300),
        validation='Seeded20percent selected project groups per district, complete labels withheld from tuning references. Per-method per-seed checkpoint selected by policy exact, fewerFP, earlierstep. No query-label tuning. If no legal full-set training targets, freeze at0.',
        routing='Additional low-budget safety control: validation-only distance switch between complete branch and SAME-fit categorical output; grid0/300/600/1000/infinity metres. No new school labels.',
        arms=['modern_frozen','modern_tuned','nearest','complete_tuned','complete_with_fallback'],
        scoring='Raw and identical existing policy processing; exactset, per-district,single/multi,FP/FN,corrected/harmed and district1km paired bootstrap2000; all semantic-pending points retained.',
        predeclared_primary='K1 complete_tuned_policy minus modern_tuned_policy and nearest_policy on common queries; K2/K3 and fallback are secondary.',
        limitations=['Prior512point results seen; current geographic snapshot is development data.','School-stratified Gold selection is label-aware simulation, not label-blind acquisition.','Three seeds are neural initializations, not independent query splits.','Existing official text facts and map Gold can disagree; count policy facts without Gold-based correction.'],
        adopted=False,independent_validation=False)
    write(OUT/'protocol.json',c);public={};verified={};paths={};summary={}
    previous=read(E.OUT/'public_plan.json')
    for d in E.DISTRICTS:
        meta=read(E.input_paths(d)[3]);dist=E.pkl(E.input_paths(d)[0]);x=E.pkl(E.input_paths(d)[2]);groups=E.group_ids(meta)
        school_ids=[s['school_id'] for s in meta['schools']]
        ledger=read(LEDGER/d/'evidence_ledger.json');vv=[]
        for r in ledger:
            if r['positive'] and r['routes']['evidence_type']=='textual_proof':
                assert r['site_id']==meta['sites'][r['q']]['coordinate_group_id']
                assert r['school_id']==school_ids[r['school_index']]
                assert r['routes']['complete_school_set'] is False
                vv.append(dict(q=r['q'],site_id=r['site_id'],school=r['school_id'],group=groups[r['q']],evidence=r['evidence'],complete=False))
        assert len({(r['site_id'],r['school']) for r in vv})==len(vv)
        verified[d]=vv
        basegroups=set(groups[x.ref_sites])|{r['group'] for r in vv}
        oldrows=previous[d]['rows']
        public[d]=dict(rows=[dict(r,selected=False,validation=False,base_reference=groups[r['q']] in basegroups) for r in oldrows],
                       all_groups=groups.tolist(),schools=meta['schools'],base_reference_groups=sorted(basegroups))
        counts={s:len({r['group'] for r in vv if r['school']==s}) for s in school_ids}
        summary[d]=dict(schools=len(school_ids),verified_partial_relations=len(vv),verified_projects=len({r['group'] for r in vv}),
            schools_without_verified_reference=sum(v==0 for v in counts.values()),
            preexisting_reference_points_excluded=sum(r['base_reference'] for r in public[d]['rows']),initial_counts=counts)
        for p in E.input_paths(d)+[LEDGER/d/'evidence_ledger.json']:paths[str(p)]=sha(p)
    paths.update(read(E.OUT/'inputs_sha256.json'))
    paths[str(E.OUT/'public_plan.json')]=sha(E.OUT/'public_plan.json')
    write(OUT/'public_population.json',public);write(OUT/'verified_partial_references.json',verified);write(OUT/'baseline_inventory.json',summary)
    write(OUT/'inputs_sha256.json',paths)
    files=[Path(__file__),E.REPO/'scripts/experiment_modern_complete_reference.py',E.REPO/'zoning/school_gap_reference.py',E.REPO/'tests/test_school_gap_reference.py',E.REPO/'scripts/audit_school_gap_reference.py']
    files += [E.REPO/p for p in read(E.OUT/'code_sha256.json') if E.REPO/p not in files]
    write(OUT/'code_sha256.json',{str(p.relative_to(E.REPO)):sha(p) for p in files})
    for p in files:
        dest=OUT/'code'/p.relative_to(E.REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'protocol_freeze.json',{n:sha(OUT/n) for n in ['protocol.json','public_population.json','verified_partial_references.json','inputs_sha256.json','code_sha256.json']})
    E.log('PREPARED',{d:{k:v for k,v in s.items() if k!='initial_counts'} for d,s in summary.items()})

def sample():
    c=verify()
    for p,h in read(OUT/'protocol_freeze.json').items():assert sha(OUT/p)==h,p
    public=read(OUT/'public_population.json');verified=read(OUT/'verified_partial_references.json')
    gold={(r['district'],r['site_id']):r for r in read(E.GOLD) if r['has_map_gold']};plans={};summary={}
    for d,v in public.items():
        schools=[s['school_id'] for s in sorted(v['schools'],key=lambda s:(s['school_name'],s['school_id']))]
        candidates=[dict(r,schools=gold[d,r['site_id']]['gold']) for r in v['rows']]
        selected=fill_school_gaps(schools,verified[d],candidates,seed=c['selection_seed'])
        common_excluded=set(v['base_reference_groups'])|{r['group'] for r in selected[3]['selected']}
        common={r['site_id'] for r in v['rows'] if r['group'] not in common_excluded}
        plans[d]={};summary[d]={}
        for k,result in selected.items():
            refs=result['selected'];rng=np.random.default_rng(c['selection_seed'])
            groups=sorted({r['group'] for r in refs});rng.shuffle(groups)
            nval=min(max(1,round(.2*len(groups))),max(0,len(groups)-2));validation=set(groups[:nval])
            released=[dict(r,complete=True,validation=r['group'] in validation,source='school_gap_gold_oracle') for r in refs]
            excluded=set(v['base_reference_groups'])|{r['group'] for r in refs}
            query=[r for r in v['rows'] if r['group'] not in excluded]
            plans[d][str(k)]=dict(support=released,query=query,common_ids=sorted(common),counts=result['counts'],deficits=result['deficits'],initial_counts=result['initial_counts'])
            summary[d][str(k)]=dict(new_verification_points=len(refs),existing_partial_upgrades=sum(r['group'] in v['base_reference_groups'] for r in refs),
                unknown_natural=len(query),unknown_common=len(common),validation=nval,schools_meeting_target=sum(n>=k for n in result['counts'].values()),
                schools_with_gap=len(result['deficits']),remaining_reference_slots=sum(result['deficits'].values()),school_deficits=result['deficits'])
    write(OUT/'released_plans.json',plans);write(OUT/'budget_summary.json',summary)
    write(OUT/'selection_freeze.json',dict(plans=sha(OUT/'released_plans.json'),protocol=sha(OUT/'protocol.json'),gold=sha(E.GOLD)))
    E.log('SAMPLED',summary)

def build(d,full,partial,groups):
    """Explicitly separate partial relations from complete set labels."""
    groups=np.asarray(groups);dist=E.pkl(E.input_paths(d)[0]);rules=E.pkl(E.input_paths(d)[1]);old=E.pkl(E.input_paths(d)[2]);meta=read(E.input_paths(d)[3])
    ids=[s['school_id'] for s in meta['schools']];by={s:i for i,s in enumerate(ids)}
    assert all(r['complete'] is True for r in full)
    complete={r['q']:tuple(sorted(by[s] for s in r['schools'])) for r in full}
    # Existing partial observations add positive relations only, never full sets.
    additions={(r['q'],by[r['school']]) for r in partial if r['q'] not in complete}
    original=set(zip(map(int,old.ref_sites),map(int,old.ref_schools)))
    additions-=original;additions|={(q,s) for q,ss in complete.items() for s in ss}
    extra=sorted(additions);ri=np.array([q for q,s in extra],int);rs=np.array([s for q,s in extra],int)
    ls=E.EV.lines(dist.lines_key);original_known={int(q):ss for q,ss in meta['known'].items()}
    if len(ri):
        st=rules.statuses;qst=st[:,rs];rst=st[ri,rs];unique,inv=np.unique(ri,return_inverse=True)
        cross,within=E.EV.crossing(ls,dist.coords,dist.coords[unique]);cn=dist.community_name;strict=dist.community_strict
        same=(cn[:,None]==cn[ri][None])&(cn[:,None]!='');both=strict[:,None]&strict[ri][None]
        blocks=dict(distance=E.EV.geometry(dist.coords[ri][None]-dist.coords[:,None],300),named_reference=np.zeros((dist.n,len(ri),1),np.float32),
            policy=E.EV.policy(qst,rules.negative[:,rs]),agreement=((qst>=0)&(rst[None]>=0)&(qst==rst[None])).astype('float32'),
            disagreement=((qst>=0)&(rst[None]>=0)&(qst!=rst[None])).astype('float32'),crossing=cross[:,inv],within_2km=within[:,inv],
            learned_boundary=E.EV.learned_boundary(ls,dist.coords,groups,original_known,ri),same_community=np.stack([same&both,(~same)&both],-1).astype('float32'))
        pair,_=E.EV._stack(blocks,list(old.pair_blocks))
    else:pair=np.zeros((dist.n,0,old.np_),np.float32)
    keep=~np.isin(old.ref_sites,list(complete));oldpair=old.pair[:,keep].copy()
    oldpair[:,:,old.pair_blocks['learned_boundary']]=E.EV.learned_boundary(ls,dist.coords,groups,original_known,old.ref_sites[keep])
    x=copy.copy(old);x.pair=np.concatenate([oldpair,pair],1);x.ref_sites=np.r_[old.ref_sites[keep],ri];x.ref_schools=np.r_[old.ref_schools[keep],rs]
    x.groups=groups;rg=groups[x.ref_sites];_,inv,cnt=np.unique(rg,return_inverse=True,return_counts=True)
    x.group_bias=-np.log(cnt[inv]).astype('float32');x.self_mask=groups[:,None]==rg[None]
    x.membership=TF.one_hot(torch.tensor(np.r_[x.ref_schools,np.arange(x.school_n)],dtype=torch.long),x.school_n).float()
    cats=catalogue(complete.values(),x.school_n);ci={s:i for i,s in enumerate(cats)}
    sm=np.zeros((len(x.ref_sites)+x.school_n,len(cats)),np.float32)
    for j,(q,s) in enumerate(zip(x.ref_sites,x.ref_schools)):sm[j,ci[complete[q] if q in complete else (int(s),)]]=1
    for s in range(x.school_n):sm[len(x.ref_sites)+s,ci[(s,)]]=1
    geo=np.full((x.n,len(cats)),-50.,np.float32);available=np.zeros_like(geo,bool)
    nearest=np.full(x.n,-1,int);ddmin=np.full(x.n,np.inf)
    for q,ss in complete.items():
        dd=np.linalg.norm(dist.coords-dist.coords[q],axis=1);valid=groups!=groups[q];j=ci[ss]
        geo[valid,j]=np.maximum(geo[valid,j],-dd[valid]/300);available[valid,j]=True
        better=valid&(dd<ddmin);nearest[better]=j;ddmin[better]=dd[better]
    labels=np.full((x.n,x.school_n),-1,np.float32)
    for q,ss in complete.items():labels[q]=0;labels[q,list(ss)]=1
    return dict(x=x,cats=cats,set_mem=torch.tensor(sm),geo=geo,available=available,nearest=nearest,nearest_distance=ddmin,
                labels=labels,school_ids=ids,coords=dist.coords,support_ids=np.array(list(complete),int))

def full_sets(data,p,z,q):
    return [data['cats'][i] if data['available'][row].any() else (int(p[j].argmax()),) for j,(row,i) in enumerate(zip(q,z.argmax(1)))]

def train(m,method,data,ids,steps,cfg,seed,callback=None):
    targets=complete_targets(data['labels'][ids],data['cats'])
    if method=='complete':
        usable=data['available'][ids,targets];ids=ids[usable];targets=targets[usable]
    if not len(ids):return 0
    rng=np.random.default_rng(seed);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'])
    for step in range(1,steps+1):
        m.train();ix=rng.choice(len(ids),min(cfg['batch'],len(ids)),replace=False);q=ids[ix]
        opt.zero_grad();school,sets=m(E.batches(data,q));labels=torch.tensor(data['labels'][q],device=E.DEVICE)
        loss=categorical_positive_loss(school,labels,torch.ones(len(q),device=E.DEVICE)/len(q)) if method=='categorical' else TF.cross_entropy(sets,torch.tensor(targets[ix],device=E.DEVICE,dtype=torch.long))
        loss+=cfg['reg']*sum(p.square().mean() for p in m.parameters());assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip']);opt.step()
        if callback and step in cfg['checkpoints']:callback(step,m)
    return len(ids)

def fit():
    E.guard();c=verify();plans=read(OUT/'released_plans.json');public=read(OUT/'public_population.json');partial=read(OUT/'verified_partial_references.json')
    assert sha(OUT/'released_plans.json')==read(OUT/'selection_freeze.json')['plans']
    allselection={};audits={}
    for k in TARGETS:
        for d in E.DISTRICTS:
            E.log('TUNE',k,d);plan=plans[d][str(k)];support=plan['support'];groups=public[d]['all_groups']
            tr=[r for r in support if not r['validation']];va=[r for r in support if r['validation']]
            tune=build(d,tr,partial[d],groups);trainids=np.array([r['q'] for r in tr],int);val=np.array([r['q'] for r in va],int)
            by={s:i for i,s in enumerate(tune['school_ids'])};truth=[set(r['schools']) for r in va];chosen={};selection={};models={}
            def evaluate(pred):return dict(n=len(truth),correct=sum(set(r['after'])==g for r,g in zip(pred,truth)),fp=sum(len(set(r['after'])-g) for r,g in zip(pred,truth)))
            for method in ['categorical','complete']:
                for seed in E.SEEDS:
                    m=E.model(seed);trials=[];beststate={};bestkey=None
                    def callback(step,m):
                        nonlocal bestkey,beststate
                        if len(val):
                            p,z=E.infer(m,tune,val);ss=[(int(i),) for i in p.argmax(1)] if method=='categorical' else full_sets(tune,p,z,val)
                            v=evaluate(E.decode(d,p,ss,tune,val))
                        else:v=dict(n=0,correct=0,fp=0)
                        v['step']=step;trials.append(v);key=(v['correct'],-v['fp'],-step)
                        if bestkey is None or key>bestkey:bestkey=key;beststate={a:b.detach().cpu().clone() for a,b in m.state_dict().items()}
                    callback(0,m);used=train(m,method,tune,trainids,80,c['fit'],seed,callback)
                    selected=max(trials,key=lambda r:(r['correct'],-r['fp'],-r['step']));chosen[method,seed]=selected['step']
                    selection[f'{method}/{seed}']=dict(trials=trials,chosen=selected,usable_train=used)
                    m.load_state_dict(beststate);models[method,seed]=m
            # No query labels: calibrate a distance fallback entirely on paid supports.
            routing=[];distance_threshold=0
            if len(val):
                pp=[];zz=[];cp=[]
                for seed in E.SEEDS:
                    p,z=E.infer(models['complete',seed],tune,val);pp.append(p);zz.append(z)
                    p,_=E.infer(models['categorical',seed],tune,val);cp.append(p)
                p=np.mean(pp,0);z=np.mean(zz,0);catp=np.mean(cp,0);sets=full_sets(tune,p,z,val);single=[(int(i),) for i in catp.argmax(1)]
                for distance in [0,300,600,1000,1e12]:
                    use=tune['nearest_distance'][val]<=distance
                    ss=[sets[j] if use[j] else single[j] for j in range(len(val))];probs=np.where(use[:,None],p,catp)
                    rr=evaluate(E.decode(d,probs,ss,tune,val));rr['distance_m']=distance;routing.append(rr)
                distance_threshold=max(routing,key=lambda r:(r['correct'],-r['fp'],-r['distance_m']))['distance_m']
            selection['fallback']=dict(trials=routing,distance_m=distance_threshold)
            allselection[f'{k}/{d}']=selection;write(OUT/'training_selection.json',allselection)
            del models,tune
            E.log('FINAL',k,d,'supports',len(support),'fallback_m',distance_threshold)
            data=build(d,support,partial[d],groups);q=np.array([r['q'] for r in plan['query']],int);trainids=np.array([r['q'] for r in support],int)
            assert not set(np.asarray(groups)[q])&set(np.asarray(groups)[data['x'].ref_sites])
            predictions={};pp=[]
            for seed in E.SEEDS:
                p,_=E.infer(E.model(seed),data,q);pp.append(p)
            frozen=np.mean(pp,0);predictions['modern_frozen']=(frozen,[(int(i),) for i in frozen.argmax(1)])
            nn=[data['cats'][i] if i>=0 else (int(frozen[j].argmax()),) for j,i in enumerate(data['nearest'][q])]
            predictions['nearest']=(frozen,nn)
            for method in ['categorical','complete']:
                pp=[];zz=[]
                for seed in E.SEEDS:
                    m=E.model(seed);train(m,method,data,trainids,chosen[method,seed],c['fit'],seed)
                    p,z=E.infer(m,data,q);pp.append(p);zz.append(z)
                    dest=OUT/'weights'/str(k)/d/f'{method}_{seed}.pt';dest.parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),dest)
                p=np.mean(pp,0);z=np.mean(zz,0)
                ss=[(int(i),) for i in p.argmax(1)] if method=='categorical' else full_sets(data,p,z,q)
                predictions['modern_tuned' if method=='categorical' else 'complete_tuned']=(p,ss)
            cp,cs=predictions['complete_tuned'];sp,ss=predictions['modern_tuned'];use=data['nearest_distance'][q]<=distance_threshold
            predictions['complete_with_fallback']=(np.where(use[:,None],cp,sp),[cs[j] if use[j] else ss[j] for j in range(len(q))])
            meta=read(E.input_paths(d)[3]);output={}
            for arm,(p,ss) in predictions.items():
                output[arm+'_raw']=[dict(district=d,site_id=meta['sites'][i]['coordinate_group_id'],building_name=meta['sites'][i]['building_name'],after=sorted(data['school_ids'][s] for s in schools)) for i,schools in zip(q,ss)]
                output[arm+'_policy']=E.decode(d,p,ss,data,q)
            provenance=[]
            for i in q:
                legal=[r for r in support if groups[r['q']]!=groups[i]]
                if legal:
                    ref=min(legal,key=lambda r:np.linalg.norm(data['coords'][i]-data['coords'][r['q']]))
                    provenance.append(dict(query=meta['sites'][i]['coordinate_group_id'],reference=ref['site_id'],distance_m=float(np.linalg.norm(data['coords'][i]-data['coords'][ref['q']]))))
            write(OUT/'predictions'/str(k)/f'{d}.json',output);write(OUT/'nearest_provenance'/str(k)/f'{d}.json',provenance)
            audits[f'{k}/{d}']=dict(released_complete_points=len(support),partial_verified_relations=len(partial[d]),reference_rows=len(data['x'].ref_sites),query=len(q),common=len(plan['common_ids']),reference_query_groups_disjoint=True,
                complete_multi_templates=sum(len(s)>1 for s in data['cats']),fallback_uses_complete=int(use.sum()),university_member_expansion=False)
            write(OUT/'input_audit.json',audits)
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['predictions','weights','nearest_provenance'] for p in (OUT/folder).rglob('*') if p.is_file()})
    write(OUT/'training_freeze.json',dict(selection=sha(OUT/'training_selection.json'),plans=sha(OUT/'released_plans.json')))
    E.log('ALL THREE BUDGETS SEALED')

def score():
    verify()
    for p,h in read(OUT/'prediction_freeze.json').items():assert sha(OUT/p)==h,p
    assert sha(E.GOLD)==read(OUT/'selection_freeze.json')['gold']
    gold={(r['district'],r['site_id']):r for r in read(E.GOLD) if r['has_map_gold']};plans=read(OUT/'released_plans.json')
    result={};allrows={}
    for k in TARGETS:
        cases={}
        for d in E.DISTRICTS:
            plan=plans[d][str(k)];queries={r['site_id']:r for r in plan['query']};common=set(plan['common_ids'])
            for arm,pred in read(OUT/'predictions'/str(k)/f'{d}.json').items():
                assert len(pred)==len(queries) and {r['site_id'] for r in pred}==set(queries)
                for r in pred:
                    sid=r['site_id'];g=gold[d,sid];truth=set(g['gold']);p=set(r['after']);xy=queries[sid]['xy']
                    cases.setdefault(arm,[]).append(dict(district=d,site_id=sid,name=r['building_name'],truth=sorted(truth),prediction=sorted(p),correct=p==truth,
                        tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),size=len(truth),semantic_pending=g['semantic_pending'],common=sid in common,block=f'{d}/{int(xy[0]//1000)}/{int(xy[1]//1000)}'))
        result[str(k)]={}
        for population in ['common','natural']:
            rr={a:[r for r in rows if population=='natural' or r['common']] for a,rows in cases.items()}
            filters={'all':lambda r:True,**{d:(lambda r,d=d:r['district']==d) for d in E.DISTRICTS},'single':lambda r:r['size']==1,'multi':lambda r:r['size']>1,'semantic_pending':lambda r:r['semantic_pending']}
            metrics={a:{g:E.metric([r for r in rows if f(r)]) for g,f in filters.items()} for a,rows in rr.items()}
            names=list(rr);matrix=np.array([[r['correct'] for r in rr[a]] for a in names],int);base=rr[names[0]]
            for a in names:assert [(r['district'],r['site_id']) for r in rr[a]]==[(r['district'],r['site_id']) for r in base]
            rng=np.random.default_rng(20261114);blocks={d:sorted({r['block'] for r in base if r['district']==d}) for d in E.DISTRICTS}
            indexes={b:np.array([i for i,r in enumerate(base) if r['block']==b]) for bs in blocks.values() for b in bs};samples=[]
            for _ in range(2000):
                ix=np.concatenate([indexes[b] for d in E.DISTRICTS for b in rng.choice(blocks[d],len(blocks[d]),replace=True)])
                samples.append(matrix[:,ix].mean(1)*100)
            samples=np.array(samples);comparisons={}
            for a,b in [('complete_tuned_policy','modern_tuned_policy'),('complete_tuned_policy','nearest_policy'),('complete_with_fallback_policy','modern_tuned_policy'),('complete_tuned_raw','nearest_raw')]:
                aa=names.index(a);bb=names.index(b);diff=matrix[aa]-matrix[bb]
                comparisons[a+' minus '+b]=dict(delta_pp=float(diff.mean()*100),corrected=int((diff==1).sum()),harmed=int((diff==-1).sum()),ci95=np.quantile(samples[:,aa]-samples[:,bb],[.025,.975]).tolist())
            result[str(k)][population]=dict(metrics=metrics,comparisons=comparisons,ci95={a:np.quantile(samples[:,i],[.025,.975]).tolist() for i,a in enumerate(names)})
        for a,rows in cases.items():write(OUT/'scored_rows'/str(k)/f'{a}.json',rows)
        E.log('SCORED',k,{a:round(m['all']['exact'],2) for a,m in result[str(k)]['common']['metrics'].items()})
    write(OUT/'results.json',dict(budgets=result,adopted=False,independent_validation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','sample','fit','score']);args=p.parse_args();globals()[args.phase]()
