"""Train a set representation/readout on frozen Modern evidence; sealed target test."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as TF
import experiment_modernnca_scope_cells as X
from zoning.modernnca_set_learning import LearnedSetReadout,evidence_latents,complete_set_targets,training_loss

M=X.E.M;REPO=X.E.REPO;OUT=REPO/'outputs/modernnca_learned_sets_20261113'
DOC=REPO.parent/'科研推进/ModernNCA集合学习_20261113'
read=X.read;write=X.write;sha=X.sha;DEVICE=M.DEVICE
ARMS=['categorical_policy','untrained_set_readout','learned_set_readout']
START=time.monotonic()
def log(*a):print(round(time.monotonic()-START,1),*a,flush=True)


def verify():
    for p,h in read(OUT/'inputs_sha256.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'code_sha256.json').items():assert sha(REPO/p)==h,p
    return read(OUT/'protocol.json')


def check(name):
    for p,h in read(OUT/name).items():assert sha(OUT/p)==h,p


def prepare():
    M.guard();X.verify();X.check('candidate_freeze.json');X.check('prediction_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve set-learning experiment')
    old=read(M.OUT/'protocol.json')
    c=dict(date='2026-11-13',sources=old['sources'],targets=M.TARGETS,seeds=M.SEEDS,selection_teacher=M.SEEDS[0],
        backbone='Frozen ModernNCA-LN full-evidence encoder/query/matching from categorical control; each final seed uses its own previous encoder.',
        learning='New learned per-school transform and joint-set mean/variance pooling + residual readout. Complete-set categorical cross entropy; zero residual recovers geometric decoder. This is trained representation/readout, not end-to-end backbone retraining.',
        source_labels='Only existing fully observed source training rows. Partial source/target positives NEVER become complete labels; target partial facts already influenced the frozen backbone and remain output constraints.',
        target_candidates='Exactly frozen scope-cell candidates and per-site lower bounds from preceding experiment. No new candidate, fact, map, Gold, or threshold.',
        source_candidates='Frozen all singletons plus complete source training-label sets from earlier candidate experiment. Train/validation project groups remain disjoint.',
        reference_boundary='Same reference tokens and self-project exclusion. Complete source sets are training supervision; no extra references inserted and no target complete labels manufactured.',
        loss='Existing same-project balanced source weights renormalized over complete source rows. Set cross entropy +0.25 conditional-singleton KL to frozen teacher on complete singleton rows; shared L2 regularization.',
        grid=[dict(lr=lr,reg=reg) for lr in [.001,.003] for reg in [.001,.01]],checkpoints=[80,160,320,640],minibatch=64,width=32,
        selection='Max source overall full-set exact among checkpoints whose singleton accuracy is no more than0.5percentage points below native teacher singleton accuracy; tie multi exact, fewer FP, earlier checkpoint. If none pass, report rejected gate and evaluate best diagnostic candidate only.',
        source_validation='Existing1246 source spatial holdout rows,60multi; reused development validation, not new independent confirmation.',
        ensemble='Average per-candidate softmax probabilities across3seeds for BOTH untrained and learned readouts, same candidate order. Per-seed results retained. Native-policy baseline copied byte-semantically.',
        primary='learned_set_readout minus untrained_set_readout',secondary='learned_set_readout minus categorical_policy',
        validation_gate='Also report final source ensemble singleton safeguard; no post-target model selection or tuning.',
        evaluation='Same1761map-hit rows,41unscored outputs,242semantic-pending retained. Exact full sets, FP/FN, corrections/harms,1km district-stratified bootstrap3000.',
        source_target_candidate_shift=True,prior_target_results_seen=True,adopted=False,independent_validation=False,
        target_gold_read=False,target_polygons_read=False)
    write(OUT/'protocol.json',c)
    links={}
    for parent,manifest in [(M.OUT,'input_links.json'),(M.OUT,'weights_freeze.json'),(M.OUT,'prediction_freeze.json'),
        (X.E.OUT,'candidate_freeze.json'),(X.OUT,'candidate_freeze.json'),(X.OUT,'prediction_freeze.json')]:
        path=parent/manifest;links[str(path)]=sha(path)
        for p,h in read(path).items():links[str(parent/p) if not Path(p).is_absolute() else p]=h
    for p in [X.OUT/'protocol.json',X.E.OUT/'source_selection.json']:
        links[str(p)]=sha(p)
    audit={}
    for d in c['sources']:
        y=dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'));ids=y['train_ids'];ids=ids[(y['labels'][ids]>=0).all(1)]
        x=M.pkl(M.OLD/'inputs'/f'{d}.pkl');assert not(set(x.groups[ids])&set(x.groups[y['valid_ids']]))
        candidates=read(X.E.OUT/'source_candidates'/f'{d}.json')
        write(OUT/'source_candidates'/f'{d}.json',candidates)
        audit[d]=dict(complete_train=len(ids),complete_multi_train=int(((y['labels'][ids]==1).sum(1)>1).sum()),
            validation=len(y['valid_ids']),validation_multi=int((y['valid_labels'].sum(1)>1).sum()))
    write(OUT/'input_audit.json',audit);write(OUT/'inputs_sha256.json',links)
    code=[Path(__file__),REPO/'zoning/modernnca_set_learning.py',REPO/'tests/test_modernnca_set_learning.py',REPO/'scripts/audit_modernnca_set_learning.py']
    write(OUT/'code_sha256.json',{str(p.relative_to(REPO)):sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'protocol_receipt.json',dict(protocol_sha256=sha(OUT/'protocol.json'),target_gold_read=False))
    write(DOC/'方案.json',c);log('PREPARED',audit)


def cache():
    M.guard();c=verify();xs,ys=M.inputs(read(M.OUT/'protocol.json'))
    for seed in c['seeds']:
        m=M.MultiLabelModernNCA(41,27,seed,32).to(DEVICE).eval()
        m.load_state_dict(torch.load(M.OUT/'weights/categorical'/f'{seed}.pt',map_location=DEVICE,weights_only=True))
        original=dict(np.load(M.OUT/'probabilities/categorical'/f'{seed}.npz'))
        for d,x in xs.items():
            path=OUT/'cache'/str(seed)/f'{d}.pt'
            if path.exists():log('CACHE EXISTS',seed,d);continue
            parts=[];probs=[]
            for lo in range(0,x.n,64):
                t=M.MD.batch(x,np.arange(lo,min(lo+64,x.n)))
                f,p=evidence_latents(m,t);parts.append(f.cpu());probs.append(p.cpu())
            f=torch.cat(parts);p=torch.cat(probs)
            ids=np.arange(x.n) if d in M.TARGETS else ys[d]['valid_ids']
            np.testing.assert_allclose(p.numpy()[ids],original[d],atol=2e-6,rtol=2e-6)
            assert torch.isfinite(f).all()
            path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(features=f,probabilities=p),path)
            log('CACHED',seed,d,tuple(f.shape))
    write(OUT/'cache_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'cache').rglob('*.pt')})
    log('CACHE SEALED')


def membership(candidates,school_n,device=DEVICE):
    a=torch.zeros(len(candidates),school_n,device=device)
    for i,r in enumerate(candidates):a[i,r['school_ids']]=1
    return a


def tasks(c,seed):
    alltasks={};total_weight=0
    for d in c['sources']:
        data=torch.load(OUT/'cache'/str(seed)/f'{d}.pt',map_location=DEVICE,weights_only=True)
        y=dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'));ids=y['train_ids'];ids=ids[(y['labels'][ids]>=0).all(1)]
        cand=read(OUT/'source_candidates'/f'{d}.json');mem=membership(cand,y['labels'].shape[1])
        yy=torch.tensor(y['labels'][ids],device=DEVICE);target=complete_set_targets(yy,mem)
        w=torch.tensor(np.load(M.OUT/f'{d}_weights.npy')[ids],dtype=torch.float32,device=DEVICE);total_weight+=float(w.sum())
        single_columns=[next(i for i,r in enumerate(cand) if r['school_ids']==[s]) for s in range(y['labels'].shape[1])]
        alltasks[d]=dict(**data,train=ids,valid=y['valid_ids'],labels=y['valid_labels'],target=target,
            membership=mem,candidates=cand,weight=w,singleton_columns=single_columns,is_single=(yy.sum(1)==1))
    for t in alltasks.values():t['weight']/=total_weight
    return alltasks


@torch.no_grad()
def task_probs(model,t,ids):
    model.eval();parts=[]
    for lo in range(0,len(ids),128):
        parts.append(model(t['features'][ids[lo:lo+128]],t['membership']).softmax(1).cpu().numpy())
    return np.concatenate(parts)


def validation(model,tt):
    ns=nm=cs=cm=fp=teacher_single=0;payload={}
    for d,t in tt.items():
        p=task_probs(model,t,t['valid']);payload[d]=p
        native=t['probabilities'][t['valid']].argmax(1).cpu().numpy()
        for i,(row,j) in enumerate(zip(t['labels'],p.argmax(1))):
            gold=set(np.flatnonzero(row));pred=set(t['candidates'][j]['school_ids'])
            if len(gold)>1:nm+=1;cm+=int(gold==pred)
            else:ns+=1;cs+=int(gold==pred);teacher_single+=int(int(native[i]) in gold)
            fp+=len(pred-gold)
    return dict(n=ns+nm,correct=cs+cm,single_n=ns,multi_n=nm,single_correct=cs,multi_correct=cm,fp=fp,
        native_single_correct=teacher_single,single_gate_passed=cs/ns>=teacher_single/ns-.005),payload


def rank(v):return (v['single_gate_passed'],v['correct'],v['multi_correct'],-v['fp'],-v['step'])


def optimize(c,seed,hp,steps,folder,selection):
    tt=tasks(c,seed);m=LearnedSetReadout(seed,width=c['width']).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=hp['lr']);rng=np.random.default_rng(seed)
    best=None;history=[]
    for step in range(1,steps+1):
        m.train();opt.zero_grad();reg=hp['reg']*sum(p.square().mean() for p in m.parameters());reg.backward();total=float(reg.detach())
        for d,t in tt.items():
            n=len(t['train']);ix=rng.choice(n,min(c['minibatch'],n),replace=False);q=t['train'][ix]
            z=m(t['features'][q],t['membership'])
            loss=training_loss(z,t['target'][ix],t['probabilities'][q],t['singleton_columns'],t['is_single'][ix],t['weight'][ix])*n/len(ix)
            assert torch.isfinite(loss),(d,step);loss.backward();total+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(norm);opt.step()
        if step%40==0:log('TRAIN',seed,hp,step,round(total,5))
        if selection and step in c['checkpoints']:
            v,_=validation(m,tt);v.update(step=step,loss=total);history.append(v)
            if best is None or rank(v)>rank(best):best=v;folder.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),folder/'best.pt')
            write(folder/'progress.json',dict(hp=hp,best=best,history=history,complete=False));log('SOURCE VALID',v)
    folder.mkdir(parents=True,exist_ok=True)
    if selection:
        result=dict(hp=hp,best=best,history=history,complete=True,budget_edge=best['step']==steps)
        write(folder/'trial.json',result);return result
    torch.save(m.state_dict(),folder/f'{seed}.pt');write(folder/f'{seed}.json',dict(seed=seed,hp=hp,steps=steps,final_loss=total))
    log('FIT COMPLETE',seed)


def select():
    M.guard();c=verify();check('cache_freeze.json');trials=[]
    for j,hp in enumerate(c['grid']):
        folder=OUT/'selection'/f'trial{j}'
        r=read(folder/'trial.json') if (folder/'trial.json').exists() else optimize(c,c['selection_teacher'],hp,max(c['checkpoints']),folder,True)
        trials.append(r)
    j=max(range(len(trials)),key=lambda j:rank(trials[j]['best']));r=trials[j]
    selected=dict(trial=j,hp=r['hp'],steps=r['best']['step'],validation=r['best'],source_gate_passed=r['best']['single_gate_passed'],budget_edge=r['budget_edge'])
    write(OUT/'selected.json',selected)
    write(OUT/'selection_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in list((OUT/'selection').rglob('*'))+[OUT/'selected.json'] if p.is_file()})
    log('SELECTED',selected)


def fit():
    M.guard();c=verify();check('cache_freeze.json');check('selection_freeze.json');sel=read(OUT/'selected.json')
    for seed in c['seeds']:
        folder=OUT/'weights'
        if not (folder/f'{seed}.pt').exists():optimize(c,seed,sel['hp'],sel['steps'],folder,False)
    write(OUT/'weights_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'weights').glob('*') if p.is_file()})


def site_matrices(d):
    meta=read(M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']];index={s:j for j,s in enumerate(ids)}
    sites=read(X.OUT/'site_candidates'/f'{d}.json');limit=max(len(s['scope_cell_candidates']) for s in sites)
    matrix=torch.zeros(len(sites),limit,len(ids),device=DEVICE);valid=torch.zeros(len(sites),limit,dtype=torch.bool,device=DEVICE)
    for q,s in enumerate(sites):
        for j,cc in enumerate(s['scope_cell_candidates']):matrix[q,j,[index[a] for a in cc['school_ids']]]=1;valid[q,j]=True
        # Dummy padding copies one valid set; masked after forward, no empty sets.
        matrix[q,len(s['scope_cell_candidates']):]=matrix[q,0]
    return sites,matrix,valid


def infer():
    M.guard();c=verify();check('cache_freeze.json');check('weights_freeze.json');check('selection_freeze.json')
    source={};target={};replay={}
    for seed in c['seeds']:
        m=LearnedSetReadout(seed,width=c['width']).to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT/'weights'/f'{seed}.pt',map_location=DEVICE,weights_only=True))
        untrained=LearnedSetReadout(seed,width=c['width']).to(DEVICE).eval();tt=tasks(c,seed)
        source[seed]={}
        for arm,model in [('learned_set_readout',m),('untrained_set_readout',untrained)]:
            v,p=validation(model,tt);source[seed][arm]=p
            dest=OUT/'source_probabilities'/arm/f'{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**p)
        del tt
        target[seed]={}
        for d in M.TARGETS:
            sites,mem,valid=site_matrices(d);cache=torch.load(OUT/'cache'/str(seed)/f'{d}.pt',map_location=DEVICE,weights_only=True)
            for arm,model in [('learned_set_readout',m),('untrained_set_readout',untrained)]:
                parts=[]
                with torch.no_grad():
                    for lo in range(0,len(sites),64):
                        f=cache['features'][lo:lo+64];mm=mem[lo:lo+64];mask=valid[lo:lo+64]
                        p=model(f,mm).masked_fill(~mask,-torch.inf).softmax(1)
                        assert torch.equal(p,model(f,mm).masked_fill(~mask,-torch.inf).softmax(1))
                        parts.append(p.cpu().numpy())
                p=np.concatenate(parts);target[seed][arm,d]=p
                dest=OUT/'target_probabilities'/arm/f'{seed}_{d}.npy';dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,p)
                replay[f'{seed}/{d}/{arm}']=True
    # Source ensemble safeguard is calculated before target scoring.
    source_summary={}
    for arm in ['learned_set_readout','untrained_set_readout']:
        correct=cs=cm=ns=nm=0
        for d in c['sources']:
            p=np.stack([source[s][arm][d] for s in c['seeds']]).mean(0)
            y=dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'))['valid_labels'];cand=read(OUT/'source_candidates'/f'{d}.json')
            for row,j in zip(y,p.argmax(1)):
                g=set(np.flatnonzero(row));ok=g==set(cand[j]['school_ids'])
                if len(g)>1:nm+=1;cm+=ok
                else:ns+=1;cs+=ok
        source_summary[arm]=dict(correct=int(cs+cm),single_correct=int(cs),multi_correct=int(cm),single_n=ns,multi_n=nm)
    native_single=0
    for d in c['sources']:
        p=np.stack([dict(np.load(M.OUT/'probabilities/categorical'/f'{s}.npz'))[d] for s in c['seeds']]).mean(0)
        y=dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'))['valid_labels'];single=y.sum(1)==1
        native_single+=int((p.argmax(1)[single]==y.argmax(1)[single]).sum())
    source_summary['native_single_correct']=native_single
    source_summary['ensemble_single_gate_passed']=source_summary['learned_set_readout']['single_correct']/ns>=native_single/ns-.005
    write(OUT/'source_ensemble.json',source_summary)
    for label in ['ensemble']+list(map(str,c['seeds'])):
        for d in M.TARGETS:
            sites,_,_=site_matrices(d)
            for arm in ['learned_set_readout','untrained_set_readout']:
                p=np.stack([target[s][arm,d] for s in c['seeds']]).mean(0) if label=='ensemble' else target[int(label)][arm,d]
                rows=[]
                for q,site in enumerate(sites):
                    chosen=site['scope_cell_candidates'][int(p[q].argmax())]
                    rows.append(dict(district=d,site_id=site['site_id'],building_name=site['building_name'],after=chosen['school_ids'],
                        status='learned_set_hypothesis' if arm=='learned_set_readout' else 'untrained_set_hypothesis',
                        complete_school_set=False,enrollment_eligibility=False))
                write(OUT/'predictions'/arm/label/f'{d}.json',rows)
            write(OUT/'predictions/categorical_policy'/label/f'{d}.json',read(M.OUT/'predictions/categorical_policy'/label/f'{d}.json'))
    write(OUT/'inference_replay.json',replay)
    files=[p for folder in ['source_probabilities','target_probabilities','predictions'] for p in (OUT/folder).rglob('*') if p.is_file()]+[OUT/'source_ensemble.json',OUT/'inference_replay.json']
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in files});log('PREDICTIONS SEALED',source_summary)


def score():
    c=verify();check('prediction_freeze.json');base=read(X.OUT/'scored_rows/scope_cell_sets.json');truth={(r['d'],r['site_id']):r for r in base}
    groups={'全部':lambda r:True,**{d:(lambda r,d=d:r['d']==d) for d in M.TARGETS},'单校':lambda r:r['size']==1,'多校':lambda r:r['size']>1,
        '语义待核':lambda r:r['semantic_pending'],'其余':lambda r:not r['semantic_pending']}
    metrics={};seeds={};allrows={}
    for arm in ARMS:
        seeds[arm]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rr=[];outside=0
            for d in M.TARGETS:
                rows=read(OUT/'predictions'/arm/label/f'{d}.json')
                expected={s['site_id'] for s in read(X.OUT/'site_candidates'/f'{d}.json')}
                assert len(rows)==len(expected) and {r['site_id'] for r in rows}==expected
                for row in rows:
                    k=(d,row['site_id'])
                    if k not in truth:outside+=1;continue
                    r=truth[k];p=set(row['after']);g=set(r['truth'])
                    rr.append(dict(r,prediction=sorted(p),correct=p==g,tp=len(p&g),fp=len(p-g),fn=len(g-p)))
            assert len(rr)==1761 and outside==41
            seeds[arm][label]={g:M.metric([r for r in rr if f(r)]) for g,f in groups.items()}
            if label=='ensemble':metrics[arm]=seeds[arm][label];allrows[arm]=rr;write(OUT/'scored_rows'/f'{arm}.json',rr)
    comparisons={}
    for b in ['untrained_set_readout','categorical_policy']:
        a='learned_set_readout';comparisons[f'{a} minus {b}']={}
        for g,f in groups.items():
            pairs=[(r,s) for r,s in zip(allrows[a],allrows[b]) if f(r)];delta=[int(r['correct'])-int(s['correct']) for r,s in pairs]
            comparisons[f'{a} minus {b}'][g]=dict(delta_pp=100*float(np.mean(delta)),corrected=delta.count(1),harmed=delta.count(-1),
                ci95=M.evaluate.bootstrap([r for r,s in pairs],delta,20261104,3000))
    gold_hash=read(X.OUT/'results.json')['gold_sha256']
    for d,h in gold_hash.items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    result=dict(metrics=metrics,seeds=seeds,comparisons=comparisons,gold_sha256=gold_hash,
        source_selection=read(OUT/'selected.json'),source_ensemble=read(OUT/'source_ensemble.json'),
        candidate_pool_unchanged=True,adopted=False,independent_validation=False)
    write(OUT/'results.json',result);write(DOC/'结果.json',result);log('RESULTS',metrics,'COMPARISONS',comparisons)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','cache','select','fit','infer','score']);globals()[ap.parse_args().phase]()
