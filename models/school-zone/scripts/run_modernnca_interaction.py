"""Frozen full-evidence residual comparison; validation screen before target fits."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_full_evidence as base
from run_modernnca_relation import tasks,device_batch,predict,digest,save
from run_modernnca_support import val_task,training_audit
from zoning import DEVICE,evaluate
from zoning.modernnca_interaction import InteractionModernNCA

OUT=REPO/'outputs/modernnca_interaction_20261113'
DOC=REPO.parent/'科研推进/ModernNCA参照交互尝试_20261113'
METHODS=['identity','rowmlp','statistics','isab']
CHECKPOINTS=[20,40,80,160]
PAIRS=[('rowmlp','identity'),('statistics','identity'),('isab','identity'),('isab','rowmlp'),('isab','statistics')]


def make(kind,x,start,seed):
    return InteractionModernNCA(x.np_,x.na,seed,kind).to(DEVICE)


def freeze():
    oldpath=REPO/'outputs/modernnca_mechanism_20261112/normalization_frozen.json'
    old=json.loads(oldpath.read_text(encoding='utf8'))
    for p,h in old['files'].items(): assert digest(REPO/p)==h,p
    paths=list(dict.fromkeys([REPO/p for p in old['files']]+[Path(__file__),
        REPO/'zoning/modernnca_interaction.py',REPO/'scripts/run_modernnca_support.py',
        REPO/'tests/test_modernnca_interaction.py',REPO/'tests/test_full_evidence.py']))
    plan=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),methods=METHODS,
        grid=base.GRID,checkpoints=CHECKPOINTS,seeds=base.SEEDS,selection_seed=base.SEEDS[0],
        validation=base.VALIDATION,batches=base.BATCHES,batch_size=base.BS,
        input_hashes=old['input_hashes'],source_hashes={str(p.relative_to(REPO)):digest(p) for p in paths},
        architecture='LN encoder and original NCA readout. Zero-initialized additive residual. ISAB: 8 inducing points, 4 heads, dim32, LN, sqrt(total dimension) scaling; key logits include log project mass before first pooling. Row MLP hidden154. Statistics MLP hidden51 reads token plus school-specific weighted reference mean/variance/log1p mass and their across-school means. No trainable school IDs. All legal rows and school anchors retained.',
        training='All four models trained fresh with identical task ids/labels/weights, group exclusion, optimizer, minibatch draws, common parameter initialization and dropout streams. Adam, clip5, four lr/reg settings. Added modules have no dropout. Statistics are derived from the same input, not extra labels.',
        selection='Fixed K0 validation district only, excluded from all training. Accuracy then NLL then earlier step. 20/40/80/160 checkpoints, fixed cap160 for EVERY method, no post-score extensions or new settings. Four trials each; a bounded development pilot, not convergence certification. K1 inherits selections.',
        gate='Each residual candidate must improve pooled selected validation accuracy in BOTH cities and >=5 of 8 target/validation scenarios. ISAB also must strictly beat BOTH rowMLP and statistics pooled across cities. These are screening conditions, not significance. Repeated validation districts count as repeated training scenarios, not independent samples.',
        confirmation='If any candidate passes, freeze ALL selections then evaluate ALL four methods, K0 and K1 three draws, three seeds. Five declared pair comparisons use Bonferroni99% spatial-block CIs within each draw. Candidate-vs-LN K1 lower bounds must all exceed0, no city95% confirmed harm. ISAB additionally must beat both controls in all three draws. No replacement of main model or claim over GBT/RetrievalScorer.',
        diagnostic='Before new fits, replay archived short-budget LN on validation only; describe errors against true-school reference mass and within-school embedding variance. This is observational, no causal or noise-label claim. No target evaluation labels accessed for selection/gate.',
        status='pre_v1 reused development data; 8 scenarios but 6 distinct validation districts; Hangzhou Gold unfinished. Whole cached objects contain historical labels: code isolation, not a blind vault.')
    plan=json.loads(json.dumps(plan));f=OUT/'frozen.json'
    if f.exists():
        prev=json.loads(f.read_text(encoding='utf8'));plan['created']=prev['created'];assert plan==prev,'Frozen sources changed'
    else:
        save(f,plan);save(DOC/'方案.json',plan)
        for p in paths:
            dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    return plan


def rank(z):return z['accuracy'],-z['nll'],-z['step']


def diagnose():
    if (OUT/'diagnosis.json').exists():return
    _,D,R,X=base.loaded(0,1);rows=[];hashes={}
    archive=REPO/'outputs/modernnca_support_20261113'
    for d in D:
        v=base.VALIDATION[d];t=val_task(d,D,R,X)
        weights=archive/'validation/identity'/f'{d}.pt'
        m=make('identity',X[d],None,base.SEEDS[0])
        m.load_state_dict(torch.load(weights,map_location=DEVICE,weights_only=True));m.eval()
        original=np.load(archive/'validation/identity'/f'{d}.npz')
        pp=predict(m,t);assert np.array_equal(original['ids'],t['ids'])
        assert np.allclose(pp,original['P'],atol=3e-6)
        hashes[d]=dict(weights=digest(weights),probabilities=digest(weights.with_suffix('.npz')))
        with torch.no_grad():
            for st in range(0,len(t['ids']),128):
                b=device_batch(t,slice(st,st+128));e,w,valid=m.encode(b);r=b['px'].shape[1]
                mem=b['mem'][:r];count=w[:,:r]@mem
                den=count[...,None].clamp_min(1e-12)
                mu=((e[:,:r]*w[:,:r,None]).transpose(1,2)@mem).transpose(1,2)/den
                second=((e[:,:r].square()*w[:,:r,None]).transpose(1,2)@mem).transpose(1,2)/den
                spread=(second-mu.square()).clamp_min(0).mean(-1)
                for j,y in enumerate(b['y'].tolist()):
                    i=st+j;q=int(t['ids'][i]);rows.append(dict(target=d,validation=v,city=D[v].city,q=q,
                        truth=y,pred=int(pp[i].argmax()),true_reference_mass=float(count[j,y]),
                        true_school_spread=float(spread[j,y]),block=evaluate.block(D[v],q)))
        del m,t;gc.collect();torch.cuda.empty_cache()
    save(OUT/'diagnosis.json',dict(archive_hashes=hashes,rows=rows,note='Validation-only old LN, descriptive not causal.'))
    print('DIAGNOSIS',len(rows),flush=True)


def fit_trial(kind,hp,d,X,tr,val,path):
    tic=time.time();seed=base.SEEDS[0];m=make(kind,X[d],None,seed)
    opt=torch.optim.Adam(m.parameters(),lr=hp['lr']);rng=np.random.default_rng(seed)
    history=[];best=None;state=None;probs=None
    for step in range(1,CHECKPOINTS[-1]+1):
        m.train();opt.zero_grad();loss=m.regularization(hp['reg']);loss.backward();total=float(loss.detach())
        for _,t in tr:
            n=len(t['ids']);ids=rng.choice(n,min(base.BS,n),replace=False);b=device_batch(t,ids)
            p=m(b);loss=-(p[torch.arange(len(ids),device=DEVICE),b['y']].clamp_min(1e-30).log()*b['weight']).sum()*n/len(ids)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if step in CHECKPOINTS:
            P=predict(m,val);y=val['y'].numpy()
            row=dict(step=step,accuracy=float((P.argmax(1)==y).mean()),nll=float(-np.log(P[np.arange(len(y)),y].clip(1e-30)).mean()),loss=total,seconds=time.time()-tic)
            history.append(row)
            if best is None or rank(row)>rank(best):
                best=row.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};probs=P
            save(path,dict(hp=hp,history=history,best=best,complete=False))
    torch.save(state,path.with_suffix('.pt'));np.savez_compressed(path.with_suffix('.npz'),P=probs,ids=val['ids'])
    info=dict(hp=hp,history=history,best=best,complete=True,seconds=time.time()-tic,parameters=sum(p.numel() for p in m.parameters()),
        state_hash=digest(path.with_suffix('.pt')),probability_hash=digest(path.with_suffix('.npz')))
    save(path,info);print('TRIAL',d,kind,hp,'best',best,'seconds',round(info['seconds'],1),flush=True)
    return info


def select():
    _,D,R,X=base.loaded(0,1)
    for d in D:
        tr=tasks(d,D,R,X,exclude={base.VALIDATION[d]});val=val_task(d,D,R,X)
        audit=training_audit(tr);assert base.VALIDATION[d] not in audit
        assert audit[d]['ids']==sorted(int(q) for q in X[d].supervised)
        save(OUT/'training_audit'/f'{d}.json',audit)
        for kind in METHODS:
            dest=OUT/'selection'/f'{d}_{kind}.json'
            if dest.exists():continue
            trials=[]
            for i,hp in enumerate(base.GRID):
                f=OUT/'trials'/f'{d}_{kind}_{i}.json'
                prior=json.loads(f.read_text(encoding='utf8')) if f.exists() else {}
                if prior.get('complete'):
                    assert digest(f.with_suffix('.pt'))==prior['state_hash']
                    assert digest(f.with_suffix('.npz'))==prior['probability_hash']
                    info=prior
                else:info=fit_trial(kind,hp,d,X,tr,val,f)
                trials.append(info)
            i=max(range(len(trials)),key=lambda j:rank(trials[j]['best']));chosen=trials[i]
            src=OUT/'trials'/f'{d}_{kind}_{i}';dst=OUT/'validation'/kind/d;dst.parent.mkdir(parents=True,exist_ok=True)
            for ext in ['.pt','.npz']:dst.with_suffix(ext).write_bytes(src.with_suffix(ext).read_bytes())
            save(dest,dict(hp=chosen['hp'],steps=chosen['best']['step'],validation=base.VALIDATION[d],best=chosen['best'],trial=i,trials=trials))
            print('SELECTED',d,kind,chosen['best'],flush=True)
        del tr,val;gc.collect();torch.cuda.empty_cache()


def comparison(rows,a,b):
    aa=np.array([r['predictions'][a]==r['truth'] for r in rows]);bb=np.array([r['predictions'][b]==r['truth'] for r in rows])
    if not len(aa):return dict(n=0)
    return dict(n=len(rows),accuracy_a=float(aa.mean()*100),accuracy_b=float(bb.mean()*100),
        delta=float((aa.astype(float)-bb).mean()*100),fixed=int((aa&~bb).sum()),broken=int((~aa&bb).sum()))


def interval(rows,delta,alpha=.05):
    rng=np.random.default_rng(20261113);num=np.zeros(10000);den=np.zeros(10000)
    delta=np.asarray(delta,dtype=float)
    for d in sorted({r['d'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['d']==d]
        keys=sorted({rows[i]['block'] for i in ix});lk={v:j for j,v in enumerate(keys)}
        bi=np.array([lk[rows[i]['block']] for i in ix]);n=np.bincount(bi);ds=np.bincount(bi,weights=delta[ix])
        draw=rng.integers(0,len(keys),(10000,len(keys)));num+=ds[draw].sum(1);den+=n[draw].sum(1)
    return (np.quantile(num/den,[alpha/2,1-alpha/2])*100).tolist()


def gate_and_audit():
    _,D,R,X=base.loaded(0,1);rows=[];replay={};maxerr=0.
    for d in D:
        v=base.VALIDATION[d];t=val_task(d,D,R,X);pp={};disabled={}
        for kind in METHODS:
            folder=OUT/'validation'/kind;m=make(kind,X[d],None,base.SEEDS[0])
            m.load_state_dict(torch.load(folder/f'{d}.pt',map_location=DEVICE,weights_only=True))
            saved=np.load(folder/f'{d}.npz');assert np.array_equal(saved['ids'],t['ids'])
            p=predict(m,t);err=float(np.abs(p-saved['P']).max());assert err<3e-6
            maxerr=max(maxerr,err);pp[kind]=p
            choice=json.loads((OUT/'selection'/f'{d}_{kind}.json').read_text(encoding='utf8'))
            assert float((p.argmax(1)==t['y'].numpy()).mean())==choice['best']['accuracy']
            bypass=[]
            with torch.no_grad():
                for st in range(0,len(t['ids']),128):bypass.append(m(device_batch(t,slice(st,st+128)),bypass=True).cpu().numpy())
            disabled[kind]=np.concatenate(bypass).argmax(1)
            replay[f'{d}_{kind}']=dict(error=err,state_hash=digest(folder/f'{d}.pt'),probability_hash=digest(folder/f'{d}.npz'),
                parameters=sum(p.numel() for p in m.parameters()),steps=choice['steps'])
            del m
        for i,q in enumerate(t['ids']):rows.append(dict(target=d,validation=v,d=v,city=D[v].city,q=int(q),truth=int(t['y'][i]),
            block=evaluate.block(D[v],int(q)),predictions={k:int(p[i].argmax()) for k,p in pp.items()},bypass={k:int(p[i]) for k,p in disabled.items()}))
        gc.collect();torch.cuda.empty_cache()
    save(OUT/'validation_cases.json',rows)
    report={}
    subsets=[('两城',rows)]+[(city,[r for r in rows if r['city']==city]) for city in ['长沙','杭州']]+[(d,[r for r in rows if r['target']==d]) for d in D]
    for a,b in PAIRS:
        result={name:comparison(rr,a,b) for name,rr in subsets}
        delta=[int(r['predictions'][a]==r['truth'])-int(r['predictions'][b]==r['truth']) for r in rows]
        result['descriptive_postselection_ci']=interval(rows,delta)
        report[f'{a}-{b}']=result
    verdict={}
    for kind in METHODS[1:]:
        z=report[f'{kind}-identity'];positive=sum(z[d]['delta']>0 for d in D)
        yes=positive>=5 and all(z[c]['delta']>0 for c in ['长沙','杭州'])
        if kind=='isab':yes=yes and all(report[f'isab-{c}']['两城']['delta']>0 for c in ['rowmlp','statistics'])
        verdict[kind]=dict(promoted=bool(yes),positive_scenarios=positive,city_deltas={c:z[c]['delta'] for c in ['长沙','杭州']})
    save(DOC/'验证结果.json',dict(comparisons=report,verdict=verdict,note='Post-selection descriptive only; repeated districts clustered together.'))
    save(DOC/'重放审计.json',dict(max_error=maxerr,models=replay))
    save(OUT/'gate.json',verdict)
    save(OUT/'sealed_selection.json',{str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'selection').glob('*.json'))})
    lines=['ModernNCA参照交互：验证区筛查','','四种方法各4组超参、20/40/80/160步，单选择种子；8个情景/6个不同验证区。']
    for a,b in PAIRS:
        z=report[f'{a}-{b}'];c=z['两城'];lines.append(f"{a} - {b}: {c['accuracy_a']:.3f}% vs {c['accuracy_b']:.3f}%, 差 {c['delta']:+.3f} 点；改对/误伤 {c['fixed']}/{c['broken']}；选择后描述区间 {z['descriptive_postselection_ci']}")
        lines.append(f"  长沙 {z['长沙']['delta']:+.3f} 点；杭州 {z['杭州']['delta']:+.3f} 点")
    lines+=['',json.dumps(verdict,ensure_ascii=False,indent=2),'','达到160步上限的选择数：']
    for k in METHODS:lines.append(f"{k}: {sum(z['steps']==160 for name,z in replay.items() if name.endswith('_'+k))}/8")
    lines+=['','这些分数用于选择与筛查，不能宣布确认提升。已训练模块的bypass只检验依赖，不替代重新训练对照。',f'32组所选模型重放最大概率误差：{maxerr}。']
    (DOC/'验证结果.txt').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines),flush=True)
    return verdict


def confirm(verdict):
    seal=json.loads((OUT/'sealed_selection.json').read_text(encoding='utf8'))
    assert len(seal)==32 and all(digest(OUT/p)==h for p,h in seal.items())
    base.OUT=OUT;base.make=make;base.METHODS=METHODS
    for k,s in base.BATCHES:base.run(k,s)
    allresults={}
    for k,s in base.BATCHES:
        c,D,R,X=base.loaded(k,s);folder=OUT/f'k{k}_s{s}';scored={};whole={}
        hashes=json.loads((folder/'prediction_hashes.json').read_text(encoding='utf8'))
        assert len(hashes)==len(D)*len(METHODS) and all(digest(folder/p)==h for p,h in hashes.items())
        for kind in METHODS:
            scored[kind]=[];whole[kind]=[]
            for d in D:
                P=np.load(folder/kind/'probs'/f'{d}.npz')['probs'].mean(0)
                er=evaluate.eval_rows(D[d],R[d].known)
                scored[kind]+=[dict(r,q=e['q'],truth=e['truth']) for r,e in zip(evaluate.correct(P,D[d],er),er)]
                whole[kind]+=[r for q,r in zip(sorted(D[d].answers),evaluate.overall(P,D[d],R[d])) if R[d].kind.get(q)!='verified']
        contrasts={}
        for a,b in PAIRS:
            z={}
            for city in ['两城','长沙','杭州']:
                aa=[r for r in scored[a] if city=='两城' or r['city']==city];bb=[r for r in scored[b] if city=='两城' or r['city']==city]
                assert [(r['d'],r['q'],r['block'],r['truth']) for r in aa]==[(r['d'],r['q'],r['block'],r['truth']) for r in bb]
                delta=[int(x['ok'])-int(y['ok']) for x,y in zip(aa,bb)]
                z[city]=dict(delta=float(np.mean(delta)*100),ci=interval(aa,delta,.01 if city=='两城' else .05),fixed=sum(x==1 for x in delta),broken=sum(x==-1 for x in delta))
            contrasts[f'{a}-{b}']=z
        allresults[f'k{k}_s{s}']=dict(model={m:evaluate.summary(rr) for m,rr in scored.items()},overall_excluding_verified={m:evaluate.summary(rr) for m,rr in whole.items()},contrasts=contrasts)
        save(DOC/'目标结果.json',allresults)
    decisions={}
    for kind in METHODS[1:]:
        required=[f'{kind}-identity']+(['isab-rowmlp','isab-statistics'] if kind=='isab' else [])
        decisions[kind]=dict(confirmed=bool(verdict[kind]['promoted'] and all(allresults[f'k1_s{s}']['contrasts'][p]['两城']['ci'][0]>0 for s in [1,2,3] for p in required)
            and not any(allresults[f'k1_s{s}']['contrasts'][p][city]['ci'][1]<0 for s in [1,2,3] for p in required for city in ['长沙','杭州'])))
    save(DOC/'目标判定.json',decisions)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','diagnose','select','report','all']);a=parser.parse_args()
    freeze()
    if a.phase in ['diagnose','all']:diagnose()
    if a.phase in ['select','all']:select()
    if a.phase in ['report','all']:
        verdict=gate_and_audit()
        if a.phase=='all':
            if any(v['promoted'] for v in verdict.values()):confirm(verdict)
            save(OUT/'completed.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),verdict=verdict,
                target_evaluated=any(v['promoted'] for v in verdict.values()),frozen_sha256=digest(OUT/'frozen.json')))


if __name__=='__main__':main()
