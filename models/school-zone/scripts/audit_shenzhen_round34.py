"""Independent full-row cache, head, and source-score audits for round34."""
import argparse
import math
import numpy as np
import torch
import experiment_shenzhen_adaptive_threshold as T
from shenzhen_adaptive_threshold import ThresholdHead,at_loss
from zoning.evaluate import bootstrap
F,R,E,S=T.F,T.R,T.E,T.S


def replay():
    F.guard();c=T.verify();R.check(T.OUT,'cache_freeze.json');R.check(T.OUT,'prediction_freeze.json')
    dest=T.OUT/'independent_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    xs,ys=E.load(c,'features_labels');history=F.read(T.OUT/'training_history.json');checked=0;checks={};max_feature_error=0.
    for seed in c['seeds']:
        bank=dict(np.load(T.OUT/'cache'/f'{seed}.npz'));model=T.scorer(xs,c,seed)
        for d in c['sources']:
            ids=ys[d]['train_ids'];ids=ids[(ys[d]['labels'][ids]>=0).all(1)]
            np.testing.assert_array_equal(ids,bank[d+'_train_ids'])
            np.testing.assert_array_equal(ys[d]['labels'][ids],bank[d+'_train_y'])
            w=np.load(E.OUT/'weights/features_labels'/f'{d}.npz')['weights'][ids]
            np.testing.assert_array_equal(w,bank[d+'_train_w'])
            assert not set(xs[d].groups[ids])&set(xs[d].groups[ys[d]['valid_ids']])
            for view in ['train','valid','stress_0','stress_1']:
                vi=ids if view=='train' else ys[d]['valid_ids']
                vx=S.view(xs[d],d,view) if view.startswith('stress') else xs[d]
                key=d+'_'+view
                with torch.no_grad():
                    for lo in range(0,len(vi),c['chunk']):
                        t=R.M.batch(vx,vi[lo:lo+c['chunk']]);z=model(t).cpu().numpy()
                        np.testing.assert_array_equal(z,bank[key+'_z'][lo:lo+len(z)])
                        ax=t['ax'].cpu().numpy().astype(np.float64);zz=z.astype(np.float64)
                        ex=np.exp(zz-zz.max(1,keepdims=True));p=ex/ex.sum(1,keepdims=True)
                        f=(ax*p[:,:,None]).sum(1);h=-(p*np.log(p.clip(1e-30))).sum(1)/math.log(p.shape[1])
                        ordered=np.sort(zz,axis=1);gap=ordered[:,-1]-ordered[:,-2]
                        expected=np.c_[f,h,gap];actual=bank[key+'_f'][lo:lo+len(z)]
                        max_feature_error=max(max_feature_error,float(np.abs(expected-actual).max()))
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=3e-6)
                        checked+=len(z)
        ff=np.concatenate([bank[d+'_train_f'] for d in c['sources']]).astype(float)
        ww=np.concatenate([bank[d+'_train_w'] for d in c['sources']]).astype(float)
        mean=(ff*ww[:,None]).sum(0)/ww.sum()
        scale=np.sqrt(((ff-mean)**2*ww[:,None]).sum(0)/ww.sum()).clip(1e-3)
        np.testing.assert_array_equal(mean.astype(np.float32),bank['mean'])
        np.testing.assert_array_equal(scale.astype(np.float32),bank['scale'])
        for arm in T.ARMS:
            ck=T.OUT/'checkpoints'/f'{arm}_{seed}.pt';path=T.OUT/'margins'/arm/f'{seed}.npz'
            hist=history[f'{arm}/{seed}'];assert F.sha(ck)==hist['checkpoint_sha256'] and F.sha(path)==hist['margins_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40 and saved['history']==hist['losses']
            assert all(np.isfinite(list(v.values())).all() for v in saved['history'])
            assert saved['protocol_hash']==F.sha(T.OUT/'protocol.json')
            head=ThresholdHead(bank['mean'],bank['scale'],arm=='adaptive_head',seed,c['width']).to(R.DEVICE)
            head.load_state_dict(saved['model']);payload=dict(np.load(path));nrows=0
            with torch.no_grad():
                for d in c['sources']:
                    for view in ['train','valid','stress_0','stress_1']:
                        key=d+'_'+view;z=bank[key+'_z'];f=bank[key+'_f']
                        th=head(torch.from_numpy(z).to(R.DEVICE),torch.from_numpy(f).to(R.DEVICE)).cpu().numpy()
                        np.testing.assert_array_equal(th,payload[key+'_threshold'])
                        np.testing.assert_array_equal(z.astype(float)-th[:,None].astype(float),payload[key])
                        np.testing.assert_array_equal(np.argsort(z,axis=1),np.argsort(payload[key],axis=1))
                        nrows+=len(z)
                        if view=='train':
                            y=bank[d+'_train_y'];w=bank[d+'_train_w'];zz=z.astype(float);tt=th.astype(float)
                            # Explicit per-row categorical cross entropy, no vectorized training formula.
                            loss=0.
                            for zr,tr,yr,wr in zip(zz,tt,y,w):
                                pos=zr[yr==1];neg=zr[yr==0]
                                pp=np.r_[tr,pos];nn=np.r_[tr,neg]
                                lp=pp.max()+np.log(np.exp(pp-pp.max()).sum())
                                ln=nn.max()+np.log(np.exp(nn-nn.max()).sum())
                                loss+=(len(pos)*lp-pos.sum()+ln-tr)*wr
                            actual=float(at_loss(torch.from_numpy(zz),torch.from_numpy(tt),torch.from_numpy(y),torch.from_numpy(w)))
                            np.testing.assert_allclose(actual,loss,rtol=1e-11,atol=1e-11)
            checks[f'{arm}/{seed}']=dict(rows=nrows,rankings_unchanged=True,thresholds_and_margins_bitwise_equal=True,training_loss_independently_recomputed=True)
        print('REPLAYED',seed,flush=True)
    F.write(dest,dict(base_rows_replayed=checked,heads=checks,max_numpy_feature_abs_error=max_feature_error,
        normalization_complete_training_only=True,original_weights_identical=True,target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))


def decisions(a,threshold=0.,inclusive=False):
    p=a>=threshold if inclusive else a>threshold
    for i in range(len(p)):
        if not p[i].any():p[i,np.argmax(a[i])]=True
    return p


def metric(pred,gold):
    n=correct=single=multi=fp=fn=0
    for ps,ys in zip(pred,gold):
        for p,y in zip(ps,ys):
            a=set(np.flatnonzero(p));b=set(np.flatnonzero(y));good=a==b
            n+=1;correct+=good;single+=good and len(b)==1;multi+=good and len(b)>1
            fp+=len(a-b);fn+=len(b-a)
    return dict(n=n,correct=correct,single_correct=single,multi_correct=multi,fp=fp,fn=fn)


def score():
    F.guard();c=T.verify();R.check(T.OUT,'prediction_freeze.json')
    assert (T.OUT/'independent_replay_audit.json').exists()
    dest=R.ROOT/'round34_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    _,ys=E.load(c,'features_labels');gold=[ys[d]['valid_labels'] for d in c['sources']]
    report=F.read(T.OUT/'source_report.json');seed_report=F.read(T.OUT/'source_seed_metrics.json');rows=F.read(T.OUT/'paired_source_rows.json')
    old=F.read(E.OUT/'source_report.json')['reports']['features_labels_relative'];threshold=old['selections']['balanced_exact']['threshold']
    preds={};allmetrics={};groups=0;attainable=0
    for arm in ['original_relative']+T.ARMS:
        raw=[dict(np.load(E.OUT/'probabilities/features_labels'/f'{s}.npz' if arm=='original_relative' else T.OUT/'margins'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([r[k] for r in raw]).mean(0) for k in raw[0]}
        def predict(r,view):
            ans=[]
            for d in c['sources']:
                a=r[d if view=='valid' else d+'_'+view] if arm=='original_relative' else r[d+'_'+view]
                a=np.asarray(a,np.float64)
                ans.append(decisions(a/a.max(1,keepdims=True),threshold,True) if arm=='original_relative' else decisions(a))
            return ans
        preds[arm]=predict(mean,'valid');full=metric(preds[arm],gold)
        stress={v:metric(predict(mean,v),gold) for v in ['stress_0','stress_1']}
        result=dict(full=full,stress=stress,balanced_exact=full['single_correct']/1186/2+full['multi_correct']/60/2)
        assert result==report['reports'][arm];allmetrics[arm]=result;groups+=3
        for seed,r in zip(c['seeds'],raw):
            ps=predict(r,'valid');expected=dict(all=metric(ps,gold),districts={d:metric([p],[g]) for d,p,g in zip(c['sources'],ps,gold)})
            assert expected==seed_report[arm][str(seed)];groups+=9
        if arm=='adaptive_head':
            for d,y in zip(c['sources'],gold):
                for z,g in zip(mean[d+'_valid'],y):
                    g=g.astype(bool)
                    attainable+=g.sum()>1 and min(z[g])>max(z[~g])
    k=0
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            row=rows[k];k+=1
            assert row['d']==d and row['q']==q and row['gold']==np.flatnonzero(gold[di][i]).tolist()
            assert row['block']==str(tuple(int(v//1000) for v in xy[q]))
            assert row['multi']==(gold[di][i].sum()>1)
            for a,p in preds.items():
                assert row['predicted'][a]==np.flatnonzero(p[di][i]).tolist()
                assert row['correct'][a]==bool((p[di][i]==gold[di][i]).all())
    assert k==1246 and sum(r['multi'] for r in rows)==60
    for a,b in [('adaptive_head','original_relative'),('adaptive_head','constant_head'),('constant_head','original_relative')]:
        for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
            value=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            assert value==report['paired'][a+'_vs_'+b][name];groups+=1
    a=allmetrics['adaptive_head'];g=c['source_gate']
    checks=dict(single=a['full']['single_correct']>=g['min_single_correct'],multi=a['full']['multi_correct']>=g['min_multi_correct'],attainable=bool(attainable>=g['min_attainable_multi']))
    for b in ['original_relative','constant_head']:
        checks['balanced_vs_'+b]=a['balanced_exact']>=allmetrics[b]['balanced_exact']
        for v in ['stress_0','stress_1']:checks[v+'_vs_'+b]=a['stress'][v]['correct']>=allmetrics[b]['stress'][v]['correct']
    assert checks==report['checks'] and attainable==report['attainable_multi']
    assert report['selected_target_candidate']==('adaptive_head' if all(checks.values()) else None)
    F.write(dest,dict(source_rows=k,metric_groups=groups,all_metrics_independently_recomputed=True,source_gate_passed=all(checks.values()),target_predictions_generated=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('SCORE_AUDIT_PASS',groups,'GATE',all(checks.values()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','score']);globals()[p.parse_args().phase]()
