"""Post-scoring source-only optimization/decision diagnostics, not deployment."""
import numpy as np
import torch
from scipy.special import logsumexp,expit
from scipy.optimize import brentq
import experiment_shenzhen_adaptive_threshold as T
from shenzhen_adaptive_threshold import ThresholdHead,at_loss,choose
from shenzhen_exact_calibration import exact_thresholds
F,R,E=T.F,T.R,T.E


def main():
    F.guard();c=T.verify();R.check(T.OUT,'prediction_freeze.json')
    assert (T.OUT/'source_report.json').exists()
    dest=T.OUT/'diagnostics_not_training/optimization_and_loss.json'
    if dest.exists():raise FileExistsError('Preserve diagnostic')
    checks={};loss_optimal_counts={};_,ys=E.load(c,'features_labels')
    for seed in c['seeds']:
        bank=dict(np.load(T.OUT/'cache'/f'{seed}.npz'));aa=[];bb=[];mm=[];ww=[]
        for d in c['sources']:
            z=bank[d+'_train_z'].astype(float);z-=z.max(1,keepdims=True);y=bank[d+'_train_y']
            aa.append(logsumexp(np.where(y==1,z,-np.inf),axis=1))
            bb.append(logsumexp(np.where(y==0,z,-np.inf),axis=1))
            mm.append((y==1).sum(1));ww.append(bank[d+'_train_w'])
        a=np.concatenate(aa);b=np.concatenate(bb);m=np.concatenate(mm);w=np.concatenate(ww)
        def derivative(offset):return float(np.sum(w*(m*expit(offset-a)+expit(offset-b)-1))+2*c['reg']*(offset+1))
        def objective(offset):return float(np.sum(w*(m*np.logaddexp(a,offset)+np.logaddexp(b,offset)-offset))+c['reg']*(offset+1)**2)
        optimum=brentq(derivative,-100,100,xtol=1e-13)
        constant=torch.load(T.OUT/'checkpoints'/f'constant_head_{seed}.pt',map_location='cpu',weights_only=False)
        fitted=float(constant['model']['offset'])
        candidate=torch.load(T.OUT/'checkpoints'/f'adaptive_head_{seed}.pt',map_location=R.DEVICE,weights_only=False)
        head=ThresholdHead(bank['mean'],bank['scale'],True,seed,c['width']).to(R.DEVICE);head.load_state_dict(candidate['model'])
        loss=head.penalty(c['reg']);terms=[]
        for d in c['sources']:
            z,f,y,w0=[torch.from_numpy(bank[d+'_train_'+k]).to(R.DEVICE) for k in ['z','f','y','w']]
            part=at_loss(z,head(z,f),y,w0);loss=loss+part;terms.append(part)
        loss.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in head.parameters())))
        checks[str(seed)]=dict(constant_fitted_offset=fitted,constant_training_optimal_offset=optimum,
            constant_fitted_derivative=derivative(fitted),constant_optimum_derivative=derivative(optimum),
            constant_avoidable_training_objective=objective(fitted)-objective(optimum),
            adaptive_final_gradient_norm=gn,adaptive_training_objective=float(loss.detach()))
    # Full validation labels below are used only for quarantined explanatory bounds.
    reports={};labels=[ys[d]['valid_labels'].astype(bool) for d in c['sources']]
    for arm in T.ARMS:
        payloads=[dict(np.load(T.OUT/'margins'/arm/f'{s}.npz')) for s in c['seeds']]
        means=[np.stack([p[d+'_valid'] for p in payloads]).mean(0) for d in c['sources']]
        # Sigmoid is monotone; this is a diagnostic global-offset envelope, not a selected target decoder.
        pp=[expit(p) for p in means];cal=exact_thresholds(pp,labels)
        # exact_thresholds stores all attainable breakpoints for direct, auditable interpretation.
        reports[arm]=dict(selections=cal['selections'],warning='Source post-scoring diagnostic only. No alternate threshold may rescue the failed round34 gate.')
    # Analytic optimum of per-row AT loss at fixed school scores; uses labels, not deployable.
    counts=dict(n=0,exact=0,multi=0,multi_exact=0,multi_rank_separable=0,rank_separable_multi_loss_miss=0)
    max_derivative=0.
    for d,y in zip(c['sources'],labels):
        z=np.stack([np.load(T.OUT/'cache'/f'{s}.npz')[d+'_valid_z'].astype(float) for s in c['seeds']]).mean(0)
        for zr,yr in zip(z,y):
            r=zr-zr.max();m=int(yr.sum());a=float(logsumexp(r[yr]));b=float(logsumexp(r[~yr]))
            # Positive root of m*x*x+(m-1)*B*x-A*B=0, evaluated without cancellation.
            if m==1:t=(a+b)/2
            else:
                logc=np.log(m-1)+b
                logsqrt=.5*np.logaddexp(2*logc,np.log(4*m)+a+b)
                t=np.log(2)+a+b-np.logaddexp(logc,logsqrt)
            derivative=m*expit(t-a)+expit(t-b)-1
            max_derivative=max(max_derivative,abs(derivative));assert abs(derivative)<1e-10
            exact=bool((choose((r-t)[None])[0]==yr).all());separable=min(r[yr])>max(r[~yr])
            counts['n']+=1;counts['exact']+=exact;counts['multi']+=m>1;counts['multi_exact']+=m>1 and exact
            counts['multi_rank_separable']+=m>1 and separable
            counts['rank_separable_multi_loss_miss']+=m>1 and separable and not exact
    F.write(dest,dict(per_seed_optimization=checks,source_threshold_envelopes=reports,
        label_using_per_row_AT_optimum={k:int(v) for k,v in counts.items()},max_optimum_derivative=float(max_derivative),
        target_gold_read=False,not_for_training=True,not_a_new_target_candidate=True,
        source_validation_reused=True,diagnostic_code_sha256=F.sha(__file__)))
    print('OPTIMIZATION',checks,flush=True);print('LABEL_USING_DIAGNOSTIC',counts,flush=True)


if __name__=='__main__':main()
