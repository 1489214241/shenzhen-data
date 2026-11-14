"""Replay round24 models and independently recompute pooling and set metrics."""
import argparse
import numpy as np
import torch
import experiment_shenzhen_set_evidence as D
from audit_shenzhen_round16 import select, counts
from audit_shenzhen_round18 import transform
from zoning.evaluate import bootstrap

F=D.F;R=D.R;X=D.X


def replay():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    dest=D.OUT/'probability_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    xs,ys=X.load(c,'all_multi_training');start=R.M.LinearRetrieval(next(iter(xs.values())))
    history=F.read(D.OUT/'training_history.json');checks={};pool_checks={}
    for arm in c['arms']:
        for seed in c['seeds']:
            key=f'{arm}/{seed}';ck=D.OUT/'checkpoints'/f'{arm}_{seed}.pt';path=D.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(path)==history[key]['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40
            assert all(np.isfinite(list(row.values())).all() for row in saved['history'])
            if arm=='scalar_control':
                assert F.sha(ck)==F.sha(X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt')
                assert saved['protocol_hash']==F.sha(X.OUT/'protocol.json')
                model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
            else:
                assert saved['protocol_hash']==F.sha(D.OUT/'protocol.json')
                assert saved['history']==history[key]['loss']
                model=D.new_model(start,seed,c)
                assert saved['model']['set_rho.2.weight'].abs().sum()>0
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model.load_state_dict(saved['model']);model.eval()
            ps=dict(np.load(path));expected={d+s for d in c['sources'] for s in ['', '_train_multi','_stress_0','_stress_1']}
            assert set(ps)==expected;n=0
            for d in c['sources']:
                for suffix in ['', '_train_multi','_stress_0','_stress_1']:
                    ids=ys[d]['valid_ids'];x=xs[d]
                    if suffix=='_train_multi':
                        ids=ys[d]['train_ids'];z=ys[d]['labels'][ids];ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
                    elif suffix:x=D.S.view(x,d,suffix[1:])
                    p=R.probabilities(model,x,ids,c['chunk']) if len(ids) else np.zeros((0,x.school_n),np.float32)
                    np.testing.assert_array_equal(p,ps[d+suffix]);n+=len(p)
            assert n==3832;checks[key]=dict(rows=n,steps=40);print('REPLAYED',key,flush=True)
            if arm=='vector_residual' and seed==c['seeds'][0]:
                # Full training population, independent per-school NumPy weighted
                # sums, rather than the model's batched matrix multiplication.
                from shenzhen_set_evidence import school_pool, SCALES
                with torch.no_grad():
                    for d,x in xs.items():
                        checked=0;maximum=0.
                        ids=ys[d]['train_ids']
                        for lo in range(0,len(ids),c['chunk']):
                            t=R.M.batch(x,ids[lo:lo+c['chunk']]);embedding=model.set_phi(t['px'])
                            e=embedding.cpu().numpy().astype(np.float64)
                            dist=t['pair'][:,:,t['meta'][0]]*300
                            membership=t['mem'][:len(x.ref_sites)]
                            for scale in SCALES:
                                w=torch.exp(t['gb'][None]-dist/scale)*t['valid']
                                actual=school_pool(embedding,w,membership).cpu().numpy()
                                weights=w.cpu().numpy().astype(np.float64);expected_pool=np.zeros(actual.shape,np.float64)
                                for school in range(x.school_n):
                                    ix=np.flatnonzero(x.ref_schools==school);mass=weights[:,ix].sum(1)
                                    numerator=(e[:,ix]*weights[:,ix,None]).sum(1)
                                    expected_pool[:,school,:-1]=numerator/np.maximum(mass,1e-12)[:,None]
                                    expected_pool[:,school,-1]=np.log1p(mass)
                                np.testing.assert_allclose(actual,expected_pool,rtol=3e-5,atol=2e-6)
                                maximum=max(maximum,float(np.max(np.abs(actual-expected_pool))))
                            checked+=len(e)
                        pool_checks[d]=dict(all_training_rows=checked,max_abs_difference=maximum)
                        print('NUMPY_POOL',d,pool_checks[d],flush=True)
    F.write(dest,dict(models=checks,pooling=pool_checks,all_source_probabilities_bitwise_reproduced=True,
        full_training_pooling_independently_recomputed=True,target_predictions_generated=False,target_gold_read=False,
        audit_code_sha256=F.sha(__file__)))


def score():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    dest=R.ROOT/'round24_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    assert F.read(D.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    source=F.read(D.OUT/'source_report.json');claims=source['reports'];_,ys=X.load(c,'all_multi_training')
    loaded={a:[dict(np.load(D.OUT/'probabilities'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    mean={a:{k:np.stack([p[k] for p in loaded[a]]).mean(0) for k in loaded[a][0]} for a in c['arms']}
    actual={};predictions={};transition_counts={}
    for arm in c['arms']:
        for mode in ['relative','absolute']:
            key=arm+'_'+mode;claim=claims[key];t=claim['selections']['balanced_exact']['threshold'];cal=F.read(D.OUT/'calibration'/f'{key}.json')
            options=[]
            for row in cal['candidates']:
                ns=nm=0
                for d in c['sources']:
                    ct=counts(select(transform(mean[arm][d],mode),row['threshold']),ys[d]['valid_labels'].astype(bool))
                    ns+=ct['single_correct'];nm+=ct['multi_correct']
                assert (ns,nm)==(row['single_correct'],row['multi_correct']);options.append((row['threshold'],ns,nm))
            transition_counts[key]=len(options)
            for obj in ['balanced_exact','micro_exact']:
                best=max(options,key=lambda z:((z[1]*60+z[2]*1186) if obj=='balanced_exact' else z[1]+z[2],z[0]))
                v=claim['selections'][obj];assert best==(v['threshold'],v['single_correct'],v['multi_correct'])
            for view in ['full','stress_0','stress_1','train_multi']:
                suffix='' if view=='full' else '_'+view
                tot={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};rank=0
                if view=='full':predictions[key]={}
                for d in c['sources']:
                    labels=ys[d]['valid_labels']
                    if view=='train_multi':
                        labels=ys[d]['labels'][ys[d]['train_ids']];labels=labels[(labels>=0).all(1)&((labels==1).sum(1)>1)]
                    truth=labels.astype(bool);p=transform(mean[arm][d+suffix],mode);pred=select(p,t)
                    for k,v in counts(pred,truth).items():tot[k]+=v
                    for pr,tr in zip(p,truth):
                        if tr.sum()>1:rank+=int(pr[tr].min()>(pr[~tr].max() if (~tr).any() else -1))
                    if view=='full':predictions[key][d]=pred
                expected=claim['selections']['balanced_exact'] if view=='full' else claim[view] if view=='train_multi' else claim['stress'][view]
                assert all(expected[k]==v for k,v in tot.items());actual[key+'/'+view]=dict(**tot,attainable_multi=rank)
            full=actual[key+'/full'];g=c['source_gate']
            passed=full['single_correct']>=g['min_single_correct'] and full['multi_correct']>=g['min_multi_correct'] and full['attainable_multi']>=g['min_attainable_multi']
            assert passed==claim['source_gate_passed'] and full['attainable_multi']==claim['attainable_multi']
            for seed,payload in zip(c['seeds'],loaded[arm]):
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
                for d in c['sources']:
                    for k,v in counts(select(transform(payload[d],mode),t),ys[d]['valid_labels'].astype(bool)).items():total[k]+=v
                assert total==claim['seeds'][str(seed)]
            print('SCORE_CHECKED',key,full,flush=True)
    rows=[]
    for d in c['sources']:
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];truth=ys[d]['valid_labels'].astype(bool)
        for i,q in enumerate(ys[d]['valid_ids']):
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(truth[i].sum()>1),
                gold=np.flatnonzero(truth[i]).tolist(),predicted={k:np.flatnonzero(v[d][i]).tolist() for k,v in predictions.items()},
                correct={k:bool((v[d][i]==truth[i]).all()) for k,v in predictions.items()}))
    assert rows==F.read(D.OUT/'paired_source_rows.json')
    for mode in ['relative','absolute']:
        a='vector_residual_'+mode;b='scalar_control_'+mode
        groups=[('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]
        for label,subset in groups:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in subset]
            v=dict(n=len(subset),before_correct=sum(r['correct'][b] for r in subset),after_correct=sum(r['correct'][a] for r in subset),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(subset,delta,seed=20261104,draws=3000))
            assert v==source['paired'][mode][label]
    key='vector_residual_relative';base='scalar_control_relative'
    selected=key if claims[key]['source_gate_passed'] and claims[key]['balanced_exact']>=claims[base]['balanced_exact'] else None
    assert selected==source['selected_target_candidate']
    F.write(dest,dict(all_source_probabilities_replayed=True,all_counts_and_transition_scores_recomputed=True,
        actual=actual,transitions=transition_counts,source_gate_recomputed=True,selected_target_candidate=selected,
        target_predictions_generated=False,independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('AUDIT_COMPLETE',selected,source['paired']['relative'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','score']);globals()[p.parse_args().phase]()
