"""Staged, validation-only mechanism screen, then frozen development evaluation.

No target evaluation labels are used in diagnose/select/gate. Full input caches
contain historical labels; code-level isolation is audited, not a blind data vault.
"""
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/'scripts'))
import run_full_evidence as base
from run_modernnca_relation import tasks, cpu_task, device_batch, predict, digest, save
from zoning import DEVICE, train, evaluate
from zoning.modernnca_support import SupportModernNCA

OUT = REPO/'outputs/modernnca_support_20261113'
DOC = REPO.parent/'科研推进/ModernNCA参照条件化尝试_20261113'
LN = REPO/'outputs/modernnca_mechanism_20261112'
METHODS = ['identity', 'scalar', 'diagonal', 'covariance', 'ridge']
CHECKPOINTS = [20, 40, 80]


def make(kind, x, start, seed):
    return SupportModernNCA(x.np_, x.na, seed, kind).to(DEVICE)


def freeze():
    old = json.loads((LN/'normalization_frozen.json').read_text(encoding='utf8'))
    for p, h in old['files'].items(): assert digest(REPO/p) == h, p
    paths = list(dict.fromkeys([REPO/p for p in old['files']] + [Path(__file__),
        REPO/'zoning/modernnca_support.py', REPO/'tests/test_modernnca_support.py']))
    plan = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), methods=METHODS,
        seeds=base.SEEDS, grid=base.GRID, checkpoints=CHECKPOINTS, extensions=[],
        batches=base.BATCHES, batch_size=base.BS, validation=base.VALIDATION,
        input_hashes=old['input_hashes'], training=old['training'],
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in paths},
        baseline_selection_hashes={p.name:digest(p) for p in (LN/'selection').glob('*_layernorm.json')},
        metric='Reference-only group-weighted population covariance. alpha=.5*n/(n+1) for n>1 groups, otherwise 0. Q=(1-alpha)I+alpha*C; distance sqrt(delta^T Q^-1 delta). Scalar/diagonal controls replace C with trace(C)/32*I or diag(C). Every reference and school anchor stays in original softmax/membership readout; query/encoder unchanged.',
        ridge='All reference and anchor embeddings divided by sqrt(32); weighted mean-square membership regression with unpenalized intercept, lambda=.1. Primal differentiable solve. Softmax(10*learned_temperature*predicted_membership). No new model parameters.',
        selection='Each method four lr/reg trials, checkpoints 20/40/80, no extensions. Accuracy then NLL then earlier step on fixed K0 validation district. LN replay uses existing histories restricted to these checkpoints; verifies replay accuracy and NLL. This is a new equal-budget SHORT pilot, not the earlier up-to-1280-step benchmark.',
        order='1. Replay restricted LN selections excluding validation district, frozen-readout interventions. 2. Train scalar/diagonal/covariance, then ridge with same four-trial budget. 3. Compute gate using validation only. 4. If promoted, freeze all selections and evaluate K0 and three K1 batches, three seeds; otherwise stop without target scoring.',
        gate='For covariance/ridge: selected validation pooled accuracy improves vs identity in BOTH cities, at least 5 of 8 target/validation folds strictly improve. Covariance additionally pooled fixed>broken vs BOTH scalar and diagonal controls. No significance claim from selection-set gate. Failure stops this concrete pilot, not the paper family. Repeated validation districts are not independent folds.',
        confirmation='If promoted: rerun matched short-budget identity and candidate on same training rows; K1 three batches all Bonferroni-adjusted spatial-block CI lower bounds>0, no city 95% confirmed harm. All promoted-vs-identity comparisons corrected; covariance controls also required. No default replacement; strong GBT/RetrievalScorer comparisons required before any main-model superiority claim.',
        status='pre_v1 development. Targets/validation previously studied; Hangzhou answers unfinished. Group self-reference exclusion inherited. Validation scores after selection are optimistic, not generalization estimates. Diagnostic intervention uses same frozen weights; end-to-end training is distinct. No novelty claim.')
    plan=json.loads(json.dumps(plan)); f=OUT/'frozen.json'
    if f.exists():
        prev=json.loads(f.read_text(encoding='utf8')); plan['created']=prev['created']
        assert plan==prev, 'Frozen protocol/source changed'
    else:
        save(f,plan); save(DOC/'方案.json',plan)
        for p in paths:
            dest=OUT/'code'/p.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(p.read_bytes())
    return plan


def rank(row): return row['accuracy'], -row['nll'], -row['step']


def short_ln(d):
    raw=json.loads((LN/'selection'/f'{d}_layernorm.json').read_text(encoding='utf8'))
    choices=[dict(hp=tr['hp'], **row) for tr in raw['trials'] for row in tr['history'] if row['step'] in CHECKPOINTS]
    return max(choices, key=rank)


def val_task(d,D,R,X):
    v=base.VALIDATION[d]
    labels=train.uncovered_answers(D[v],X[v])
    t=cpu_task(v, sorted(labels), D, X, labels)
    # Group exclusion must cover *every* reference from the query building.
    expected=D[v].groups[t['ids']][:,None] != D[v].groups[X[v].ref_sites][None]
    assert np.array_equal(t['valid'].numpy(),expected)
    return t


def training_audit(tr):
    return {d:dict(ids=t['ids'].tolist(), labels=t['y'].cpu().tolist(), groups=t['groups'].tolist()) for d,t in tr}


def diagnostic():
    c,D,R,X=base.loaded(0,1)
    for d in D:
        dest=OUT/'diagnostic'/f'{d}.json'
        if dest.exists(): continue
        v=base.VALIDATION[d]; tr=tasks(d,D,R,X,exclude={v}); val=val_task(d,D,R,X)
        audit=training_audit(tr)
        assert v not in audit and audit[d]['ids']==sorted(int(q) for q in X[d].supervised)
        save(OUT/'validation_training_audit'/f'{d}.json',audit)
        choice=short_ln(d)
        m,info=base.fit('identity',choice['hp'],base.SEEDS[0],d,X,tr,None,steps=choice['step'])
        m.eval(); pp={k:[] for k in METHODS}; counts=[]; has_ref=[]; conditions=[]
        with torch.no_grad():
            for st in range(0,len(val['ids']),128):
                b=device_batch(val,slice(st,st+128)); enc=m.encode_support(b)
                for kind in METHODS:
                    p,details=m.readout(*enc,b,variant=kind,details=True); pp[kind].append(p.cpu().numpy())
                    if kind=='covariance':
                        counts.extend(details['group_count'].cpu().tolist())
                        eig=torch.linalg.eigvalsh(details['covariance'])
                        conditions.extend((eig[:,-1]/eig.mean(-1).clamp_min(1e-12)).cpu().tolist())
                r=b['px'].shape[1]
                support=enc[2][:,:r] @ b['mem'][:r]
                has_ref.extend((support[torch.arange(len(b['y']),device=DEVICE),b['y']]>0).cpu().tolist())
        pp={k:np.concatenate(p) for k,p in pp.items()}; y=val['y'].numpy()
        acc=float((pp['identity'].argmax(1)==y).mean()); nll=float(-np.log(pp['identity'][np.arange(len(y)),y].clip(1e-30)).mean())
        assert abs(acc-choice['accuracy'])<1e-10, (d,acc,choice)
        assert abs(nll-choice['nll'])<2e-5, (d,nll,choice['nll'])
        folder=OUT/'validation/identity'; folder.mkdir(parents=True,exist_ok=True)
        torch.save(m.cpu().state_dict(), folder/f'{d}.pt'); np.savez_compressed(folder/f'{d}.npz',P=pp['identity'],ids=val['ids'])
        save(OUT/'selection'/f'{d}_identity.json',dict(hp=choice['hp'],steps=choice['step'],validation=v,best=choice,replayed=True))
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest.with_suffix('.npz'),**pp)
        rows=[dict(q=int(q),truth=int(y[i]),group=str(val['groups'][i]),block=evaluate.block(D[v],int(q)),
            has_true_reference=has_ref[i],group_count=counts[i],largest_eigenvalue_over_mean=conditions[i],
            predictions={k:int(p[i].argmax()) for k,p in pp.items()}) for i,q in enumerate(val['ids'])]
        save(dest,dict(target=d,validation=v,city=D[v].city,choice=choice,rows=rows,
            metrics={k:float((p.argmax(1)==y).mean()*100) for k,p in pp.items()},training=info))
        print('DIAG',d,v,{k:round(float((p.argmax(1)==y).mean()*100),3) for k,p in pp.items()},flush=True)
        del tr,val,m; gc.collect(); torch.cuda.empty_cache()


def fit_selection(kind,hp,seed,d,X,tr,val,log):
    m=make(kind,X[d],None,seed); opt=torch.optim.Adam(m.parameters(),lr=hp['lr'])
    rng=np.random.default_rng(seed); history=[]; best=None; state=None; probs=None; tic=time.time()
    for step in range(1,max(CHECKPOINTS)+1):
        m.train(); opt.zero_grad(); loss=m.regularization(hp['reg']); loss.backward(); total=float(loss.detach())
        for _,t in tr:
            n=len(t['ids']); ids=rng.choice(n,min(base.BS,n),replace=False); b=device_batch(t,ids)
            p=m(b); loss=-(p[torch.arange(len(ids),device=DEVICE),b['y']].clamp_min(1e-30).log()*b['weight']).sum()*n/len(ids)
            assert torch.isfinite(loss); loss.backward(); total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
        if step in CHECKPOINTS:
            P=predict(m,val); y=val['y'].numpy()
            row=dict(step=step,accuracy=float((P.argmax(1)==y).mean()),nll=float(-np.log(P[np.arange(len(y)),y].clip(1e-30)).mean()),loss=total,seconds=time.time()-tic)
            history.append(row)
            if best is None or rank(row)>rank(best):
                best=row.copy(); state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; probs=P
            save(log,dict(hp=hp,best=best,history=history,complete=False))
    # All selected artifacts written before completion marker permits resumption.
    torch.save(state,log.with_suffix('.pt')); np.savez_compressed(log.with_suffix('.npz'),P=probs,ids=val['ids'])
    info=dict(hp=hp,best=best,history=history,complete=True,seconds=time.time()-tic)
    save(log,info); print('TRIAL',d,kind,hp,'best',best,'elapsed',info['seconds'],flush=True)
    return info


def select(methods):
    c,D,R,X=base.loaded(0,1)
    # Complete distance route first; classifier route comes after all metric fits.
    for kind in methods:
        for d in D:
            final=OUT/'selection'/f'{d}_{kind}.json'
            if final.exists(): continue
            tr=tasks(d,D,R,X,exclude={base.VALIDATION[d]}); val=val_task(d,D,R,X)
            audit=training_audit(tr)
            assert audit==json.loads((OUT/'validation_training_audit'/f'{d}.json').read_text(encoding='utf8'))
            trials=[]
            for i,hp in enumerate(base.GRID):
                f=OUT/'selection/trials'/f'{d}_{kind}_{i}.json'
                old=json.loads(f.read_text(encoding='utf8')) if f.exists() else {}
                info=old if old.get('complete') else fit_selection(kind,hp,base.SEEDS[0],d,X,tr,val,f)
                trials.append(info)
            i=max(range(len(trials)),key=lambda j:rank(trials[j]['best'])); chosen=trials[i]
            src=OUT/'selection/trials'/f'{d}_{kind}_{i}'
            dst=OUT/'validation'/kind/d; dst.parent.mkdir(parents=True,exist_ok=True)
            for ext in ['.pt','.npz']: dst.with_suffix(ext).write_bytes(src.with_suffix(ext).read_bytes())
            save(final,dict(hp=chosen['hp'],steps=chosen['best']['step'],validation=base.VALIDATION[d],best=chosen['best'],trial=i,trials=trials))
            print('SELECTED',d,kind,chosen['best'],flush=True)
            del tr,val; gc.collect(); torch.cuda.empty_cache()


def gate():
    allrows=[]; folds={}; results={}; diagnostic={}; selections={}
    for f in sorted((OUT/'diagnostic').glob('*.json')):
        raw=json.loads(f.read_text(encoding='utf8')); d=raw['target']; rows=raw['rows']
        pred={m:np.load(OUT/'validation'/m/f'{d}.npz')['P'] for m in METHODS}
        ids=np.load(OUT/'validation/identity'/f'{d}.npz')['ids']
        assert ids.tolist()==[r['q'] for r in rows]
        for m in METHODS:
            z=np.load(OUT/'validation'/m/f'{d}.npz'); assert np.array_equal(z['ids'],ids)
            assert np.isfinite(pred[m]).all() and np.allclose(pred[m].sum(1),1,atol=2e-5)
            selections[f'{d}_{m}']=json.loads((OUT/'selection'/f'{d}_{m}.json').read_text(encoding='utf8'))
        out=[dict(r,target=d,validation=raw['validation'],city=raw['city'],frozen_predictions=r['predictions'],
                  predictions={m:int(pred[m][i].argmax()) for m in METHODS}) for i,r in enumerate(rows)]
        allrows+=out; folds[d]=out
    assert len(folds)==8
    def comparison(rows,a,b,key='predictions'):
        aa=np.array([r[key][a]==r['truth'] for r in rows]); bb=np.array([r[key][b]==r['truth'] for r in rows])
        return dict(n=len(rows),accuracy_a=float(aa.mean()*100),accuracy_b=float(bb.mean()*100),
            delta=float((aa.astype(float)-bb).mean()*100),fixed=int((aa&~bb).sum()),broken=int((~aa&bb).sum()))
    for m in METHODS[1:]:
        results[m]={name:comparison(rr,m,'identity') for name,rr in [('两城',allrows),('长沙',[r for r in allrows if r['city']=='长沙']),('杭州',[r for r in allrows if r['city']=='杭州'])]+list(folds.items())}
        for comparator in ['scalar','diagonal']:
            results[m][f'vs_{comparator}']=comparison(allrows,m,comparator)
        results[m]['baseline_wrong_with_true_reference']=comparison([r for r in allrows if r['has_true_reference'] and r['predictions']['identity']!=r['truth']],m,'identity')
        diagnostic[m]={name:comparison(rr,m,'identity','frozen_predictions') for name,rr in [('两城',allrows),('长沙',[r for r in allrows if r['city']=='长沙']),('杭州',[r for r in allrows if r['city']=='杭州'])]}
    verdict={}
    for m in ['covariance','ridge']:
        z=results[m]; positive=sum(z[d]['delta']>0 for d in folds)
        yes=all(z[c]['delta']>0 for c in ['长沙','杭州']) and positive>=5
        if m=='covariance': yes=yes and all(z[f'vs_{c}']['delta']>0 for c in ['scalar','diagonal'])
        verdict[m]=dict(promoted=bool(yes),positive_folds=positive,city_deltas={c:z[c]['delta'] for c in ['长沙','杭州']})
    save(OUT/'validation_cases.json',allrows); save(DOC/'验证结果.json',dict(trained=results,frozen_intervention=diagnostic,verdict=verdict))
    save(OUT/'gate.json',verdict)
    sealed={str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'selection').glob('*.json'))}
    save(OUT/'sealed_selection.json',sealed)
    print('GATE',json.dumps(verdict,ensure_ascii=False),flush=True)
    lines=['参照条件化改进：验证区筛查结果','','同一短预算：4组学习率/正则，20/40/80步选择；单个选择种子。开发数据，非独立确认。',
        '重复验证区的题目按各目标训练情景重复计入，两城汇总不是独立样本统计。','']
    for m in METHODS[1:]:
        lines.append(m)
        for city in ['两城','长沙','杭州']:
            z=results[m][city]; lines.append(f"  {city}: {z['accuracy_a']:.3f}% vs LN {z['accuracy_b']:.3f}%, 差 {z['delta']:+.3f} 点, 改对/误伤 {z['fixed']}/{z['broken']}")
        f=diagnostic[m]['两城']; lines.append(f"  固定LN权重只换读出: {f['delta']:+.3f} 点, 改对/误伤 {f['fixed']}/{f['broken']}")
    lines += ['',json.dumps(verdict,ensure_ascii=False,indent=2),'','晋级只表示值得继续，未晋级也不证明论文方法无效。80步上限限制了结论。']
    DOC.mkdir(parents=True,exist_ok=True); (DOC/'验证结果.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['freeze','diagnostic','select','gate','screen'])
    p.add_argument('--methods',nargs='+',choices=METHODS[1:],default=METHODS[1:]); a=p.parse_args()
    freeze(); base.make=make; base.CHECKPOINTS=CHECKPOINTS
    if a.phase in ['diagnostic','screen']: diagnostic()
    if a.phase in ['select','screen']: select(a.methods)
    if a.phase in ['gate','screen']: gate()


if __name__=='__main__': main()
