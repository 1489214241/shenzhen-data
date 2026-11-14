"""Round36: continue source-only optimization, retaining the original convergence tolerance.

Round34 files/code remain immutable. Only output-root injection is shared with
its deterministic reporting and independent scoring functions; it is restored
on exit. All dependencies are frozen explicitly in this round.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp,expit
from scipy.optimize import brentq
import torch
import experiment_shenzhen_adaptive_threshold as T
from shenzhen_adaptive_threshold import ThresholdHead,at_loss
F,R,E=T.F,T.R,T.E
PARENT=R.ROOT/'round35_threshold_convergence'
OUT=R.ROOT/'round36_threshold_convergence'


@contextmanager
def output_root():
    old=T.OUT;T.OUT=OUT
    try:yield
    finally:T.OUT=old


def verify():
    with output_root():return T.verify()


def prepare():
    F.guard();old=T.verify();R.check(PARENT,'prediction_freeze.json');R.check(PARENT,'cache_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve round36')
    c=dict(old);c.update(steps=10000,optimizer='LBFGS float64 lr1 strong_wolfe history100; max10000inner iterations in400blocks of25.',
        initializer='Each arm continues its own frozen round35 final head; school scorer remains fixed.',
        stopping='Training gradient infinity norm<=1e-6; max10000iterations;3unchanged objective blocks with improvement<=1e-12 stops as stalled, not converged. No validation access.',
        convergence_gate='All10adaptive heads must satisfy gradient-infinity tolerance in addition to original source gates; no checkpoint selection by validation.',
        sufficient_statistics='Exact AT loss via positive/negative logsumexp per complete source row; independently verify against uncompressed school loss.',
        parameter_matching=False,checkpoint='Save after each25iteration optimizer block, resume saved optimizer; no validation early stopping.',
        gradient_tolerance=1e-6,block_iterations=25,max_blocks=400,optimizer_lr=1.,history_size=100)
    F.write(OUT/'protocol.json',c)
    links=dict(F.read(PARENT/'input_links.json')['files'])
    for name in ['input_freeze.json','cache_freeze.json','prediction_freeze.json','source_report.json','independent_replay_audit.json']:
        links[str(PARENT/name)]=F.sha(PARENT/name)
    for rel,h in F.read(PARENT/'prediction_freeze.json')['files'].items():links[str(PARENT/rel)]=h
    for p in [Path(__file__),F.REPO/'scripts/audit_shenzhen_round36_metrics.py',R.ROOT/'round35_independent_completion_audit.json',PARENT/'admission_gate.json']:
        links[str(p)]=F.sha(p)
    for p in (PARENT/'cache').glob('*.npz'):
        dest=OUT/'cache'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        assert F.sha(dest)==F.sha(p);links[str(p)]=F.sha(p)
    shutil.copyfile(PARENT/'cache_freeze.json',OUT/'cache_freeze.json')
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('PREPARED_ROUND36',flush=True)


def sufficient(bank,c):
    parts={k:[] for k in ['a','b','n','p','w','f']}
    for d in c['sources']:
        z=bank[d+'_train_z'].astype(float);z-=z.max(1,keepdims=True);y=bank[d+'_train_y']
        assert (y>=0).all()
        parts['a'].append(logsumexp(np.where(y==1,z,-np.inf),axis=1))
        parts['b'].append(logsumexp(np.where(y==0,z,-np.inf),axis=1))
        parts['n'].append((y==1).sum(1));parts['p'].append(np.where(y==1,z,0).sum(1))
        parts['w'].append(bank[d+'_train_w']);parts['f'].append(bank[d+'_train_f'])
    return {k:torch.as_tensor(np.concatenate(v),dtype=torch.float64,device=R.DEVICE) for k,v in parts.items()}


def objective(head,stats,reg):
    delta=head(torch.zeros((len(stats['a']),1),dtype=torch.float64,device=R.DEVICE),stats['f'])
    terms=stats['n']*torch.logaddexp(delta,stats['a'])-stats['p']+torch.logaddexp(delta,stats['b'])-delta
    return (terms*stats['w']).sum()+head.penalty(reg)


def train():
    F.guard();c=verify()
    if (OUT/'prediction_freeze.json').exists():R.check(OUT,'prediction_freeze.json');return
    history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        bank=dict(np.load(OUT/'cache'/f'{seed}.npz'));stats=sufficient(bank,c)
        for arm in T.ARMS:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'margins'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['margins_sha256'];continue
            head=ThresholdHead(bank['mean'],bank['scale'],arm=='adaptive_head',seed,c['width']).to(R.DEVICE).double()
            parent=torch.load(PARENT/'checkpoints'/f'{arm}_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            head.load_state_dict(parent['model'])
            opt=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=c['block_iterations'],max_eval=40,tolerance_grad=c['gradient_tolerance'],tolerance_change=1e-13,history_size=c['history_size'],line_search_fn='strong_wolfe')
            losses=[];stalled=0;reason='max_iterations';begin=0
            if ck.exists():
                saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==F.sha(OUT/'protocol.json')
                head.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);losses=saved['history'];begin=len(losses);stalled=saved['stalled']
                if saved['reason'] in ['gradient_converged','stalled']:begin=c['max_blocks'];reason=saved['reason']
            # Exact compressed objective must match the original complete-source loss.
            with torch.no_grad():
                brute=head.penalty(c['reg'])
                for d in c['sources']:
                    z,f,y,w=[torch.as_tensor(bank[d+'_train_'+k],device=R.DEVICE).double() for k in ['z','f','y','w']]
                    brute=brute+at_loss(z,head(z,f),y,w)
                torch.testing.assert_close(brute,objective(head,stats,c['reg']),rtol=1e-12,atol=1e-12)
            def closure():
                opt.zero_grad();v=objective(head,stats,c['reg']);assert torch.isfinite(v);v.backward();return v
            for block in range(begin,c['max_blocks']):
                before=float(objective(head,stats,c['reg']).detach());opt.step(closure);value=closure()
                gi=max(float(p.grad.abs().max()) for p in head.parameters());after=float(value.detach())
                state=opt.state[next(iter(head.parameters()))]
                assert after<=before+1e-9
                stalled=stalled+1 if before-after<=1e-12 else 0
                reason='gradient_converged' if gi<=c['gradient_tolerance'] else 'stalled' if stalled>=3 else 'max_iterations'
                losses.append(dict(block=block+1,objective=after,gradient_inf=gi,optimizer_iterations=int(state['n_iter']),function_evaluations=int(state['func_evals'])))
                ck.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(model=head.state_dict(),optimizer=opt.state_dict(),history=losses,protocol_hash=F.sha(OUT/'protocol.json'),reason=reason,stalled=stalled),ck)
                if (block+1)%5==0 or reason!='max_iterations':print('OPTIMIZE',key,losses[-1],reason,flush=True)
                if reason!='max_iterations':break
            payload={}
            with torch.no_grad():
                for d in c['sources']:
                    for view in ['train','valid','stress_0','stress_1']:
                        k=d+'_'+view;z=bank[k+'_z'];f=bank[k+'_f']
                        th=head(torch.as_tensor(z,device=R.DEVICE).double(),torch.as_tensor(f,device=R.DEVICE).double()).cpu().numpy()
                        payload[k+'_threshold']=th;payload[k]=z.astype(float)-th[:,None]
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(reason=reason,final=losses[-1],checkpoint_sha256=F.sha(ck),margins_sha256=F.sha(dest),parent_sha256=F.sha(PARENT/'checkpoints'/f'{arm}_{seed}.pt'))
            F.write(OUT/'training_history.json',history);print('COMPLETE',key,reason,flush=True)
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['checkpoints','margins'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    with output_root():T.report()
    result=F.read(OUT/'source_report.json');history=F.read(OUT/'training_history.json')
    converged=all(history[f'adaptive_head/{s}']['reason']=='gradient_converged' for s in verify()['seeds'])
    # Keep the shared predictive report immutable; convergence is a separate stricter gate.
    F.write(OUT/'admission_gate.json',dict(predictive_candidate=result['selected_target_candidate'],all_adaptive_converged=converged,
        selected_target_candidate='adaptive_head' if converged and result['selected_target_candidate']=='adaptive_head' else None,
        target_predictions_generated=False,adopted=False))


def replay():
    F.guard();c=verify();R.check(OUT,'prediction_freeze.json');R.check(OUT,'cache_freeze.json')
    dest=OUT/'independent_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    history=F.read(OUT/'training_history.json');checks={}
    assert F.read(PARENT/'independent_replay_audit.json')['cached_inputs_identical_to_verified_round34']
    for seed in c['seeds']:
        path=OUT/'cache'/f'{seed}.npz';assert F.sha(path)==F.sha(PARENT/'cache'/path.name)
        bank=dict(np.load(path));stats=sufficient(bank,c)
        for arm in T.ARMS:
            ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';path=OUT/'margins'/arm/f'{seed}.npz';h=history[f'{arm}/{seed}']
            assert F.sha(ck)==h['checkpoint_sha256'] and F.sha(path)==h['margins_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==F.sha(OUT/'protocol.json')
            head=ThresholdHead(bank['mean'],bank['scale'],arm=='adaptive_head',seed,c['width']).to(R.DEVICE).double();head.load_state_dict(saved['model'])
            loss=objective(head,stats,c['reg']);loss.backward();gi=max(float(p.grad.abs().max()) for p in head.parameters())
            assert gi==h['final']['gradient_inf'] and float(loss.detach())==h['final']['objective']
            assert (h['reason']=='gradient_converged')==(gi<=c['gradient_tolerance'])
            payload=dict(np.load(path));rows=0;brute=head.penalty(c['reg'])
            with torch.no_grad():
                for d in c['sources']:
                    for view in ['train','valid','stress_0','stress_1']:
                        k=d+'_'+view;z=torch.as_tensor(bank[k+'_z'],device=R.DEVICE).double();f=torch.as_tensor(bank[k+'_f'],device=R.DEVICE).double()
                        th=head(z,f);np.testing.assert_array_equal(th.cpu().numpy(),payload[k+'_threshold'])
                        np.testing.assert_array_equal(z.cpu().numpy()-th.cpu().numpy()[:,None],payload[k]);rows+=len(z)
                        if view=='train':brute=brute+at_loss(z,th,torch.as_tensor(bank[k+'_y'],device=R.DEVICE),torch.as_tensor(bank[k+'_w'],device=R.DEVICE))
                torch.testing.assert_close(brute,loss,rtol=1e-12,atol=1e-12)
            check=dict(rows=rows,gradient_inf=gi,converged=h['reason']=='gradient_converged',loss_decreased=saved['history'][-1]['objective']<=saved['history'][0]['objective'])
            if arm=='constant_head':
                a,b,n,w=[stats[k].cpu().numpy() for k in ['a','b','n','w']]
                def deriv(v):return np.sum(w*(n*expit(v-a)+expit(v-b)-1))+2*c['reg']*(v+1)
                optimum=brentq(deriv,-100,100,xtol=1e-13);offset=float(head.offset.detach())
                assert abs(optimum-offset)<1e-5
                check.update(exact_optimum=optimum,fitted_offset=offset,exact_optimum_verified=True)
            checks[f'{arm}/{seed}']=check
        print('REPLAYED',seed,flush=True)
    F.write(dest,dict(heads=checks,cached_inputs_identical_to_verified_round35=True,all_head_outputs_bitwise_replayed=True,full_and_compressed_training_losses_match=True,target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))


def score():
    import audit_shenzhen_round36_metrics as A
    with output_root():A.score()
    c=verify();gate=F.read(OUT/'admission_gate.json');history=F.read(OUT/'training_history.json');result=F.read(OUT/'source_report.json')
    conv=all(history[f'adaptive_head/{s}']['reason']=='gradient_converged' for s in c['seeds'])
    assert gate['all_adaptive_converged']==conv and gate['selected_target_candidate']==('adaptive_head' if conv and result['selected_target_candidate']=='adaptive_head' else None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report','replay','score']);globals()[p.parse_args().phase]()
