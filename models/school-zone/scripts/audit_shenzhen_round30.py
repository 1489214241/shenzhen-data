"""Independent numerical replay and full source metric audit for round30."""
import argparse
import pickle
import numpy as np
import torch
import experiment_shenzhen_hierarchy_loss as E
from audit_shenzhen_round16 import select, counts
from audit_shenzhen_round18 import transform
from zoning.evaluate import bootstrap

F,R=E.F,E.R


def replay():
    F.guard();c=E.verify();R.check(E.OUT,'probability_freeze.json')
    dest=E.OUT/'probability_replay_audit.json'
    if dest.exists(): raise FileExistsError('Preserve replay')
    xs,ys=E.X.load(c,'all_multi_training'); start=R.M.LinearRetrieval(next(iter(xs.values())))
    for d in c['targets']:
        with (E.X.TARGET/d/'district.pkl').open('rb') as f: district=pickle.load(f)
        meta=F.read(R.OLD/'inputs'/f'{d}.json')
        assert district.school_ids==[s['school_id'] for s in meta['schools']]
        assert district.answers=={} and district.train_answers=={}
    hist=F.read(E.OUT/'training_history.json');checks={};loss_checks={}
    for arm in c['arms']:
        for seed in c['seeds']:
            key=f'{arm}/{seed}';ck=E.OUT/'checkpoints'/f'{arm}_{seed}.pt';p=E.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(ck)==hist[key]['checkpoint_sha256'] and F.sha(p)==hist[key]['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40
            assert all(np.isfinite(list(v.values())).all() for v in saved['history'])
            if arm=='scalar_control': assert F.sha(ck)==F.sha(E.D.OUT/'checkpoints'/ck.name)
            else:
                assert saved['protocol_hash']==F.sha(E.OUT/'protocol.json')
                assert saved['history']==hist[key]['losses']
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
            model.load_state_dict(saved['model']);model.eval();actual=dict(np.load(p));n=0
            assert set(actual)=={d+s for d in c['sources'] for s in ['', '_train_multi','_stress_0','_stress_1']}
            for d in c['sources']:
                for suffix in ['', '_train_multi','_stress_0','_stress_1']:
                    ids=ys[d]['valid_ids'];x=xs[d]
                    if suffix=='_train_multi':
                        ids=ys[d]['train_ids'];y=ys[d]['labels'][ids];ids=ids[(y>=0).all(1)&((y==1).sum(1)>1)]
                    elif suffix: x=E.S.view(x,d,suffix[1:])
                    expected=R.probabilities(model,x,ids,c['chunk']) if len(ids) else np.zeros((0,x.school_n),np.float32)
                    np.testing.assert_array_equal(actual[d+suffix],expected);n+=len(expected)
            assert n==3832;checks[key]=dict(rows=n,steps=40)
            # Scalar loop in probability space, independent of batched logit
            # masks, for every observed target cell at first final checkpoint.
            if arm=='hierarchy_loss' and seed==c['seeds'][0]:
                for d in c['targets']:
                    ids=[s['school_id'] for s in F.read(R.OLD/'inputs'/f'{d}.json')['schools']]
                    idx={s:i for i,s in enumerate(ids)};children={i:{i} for i in range(len(ids))}
                    for edge in F.read(R.ROOT/'scope_graph_refresh_review/graphs'/f'{d}.json')['edges']:
                        children[idx[edge['implied_school_id']]].add(idx[edge['trigger_school_id']])
                    rel=E.relation_matrix(ids,F.read(R.ROOT/'scope_graph_refresh_review/graphs'/f'{d}.json')['edges']).to(R.DEVICE)
                    rows=0;maximum=0.
                    for t in R.frames(c,xs,ys,'new_references_labels',[d]):
                        with torch.no_grad(): z=model(t).double()
                        # BCE returns the target dtype even with double logits.
                        # Cast BOTH inputs for the independent double formula
                        # check; separately compare production float32 below.
                        y=t['labels'].double();w=t['sample_weight'].double();expected=0.
                        zz=z.cpu().numpy();yy=y.cpu().numpy();ww=w.cpu().numpy()
                        for i in range(len(zz)):
                            for school in range(len(ids)):
                                label=yy[i,school]
                                if label<0: continue
                                options=[j for j in children[school] if label==0 or yy[i,j]==1]
                                best=max(zz[i,j] for j in options)
                                expected+=np.logaddexp(0.,-best if label==1 else best)*ww[i]
                        actual_loss=float(E.masked_max_constraint_loss(z,y,w,rel))
                        assert np.isclose(expected,actual_loss,rtol=1e-12,atol=1e-12)
                        production=float(E.masked_max_constraint_loss(z.float(),t['labels'],t['sample_weight'],rel))
                        assert np.isclose(expected,production,rtol=2e-6,atol=1e-8)
                        maximum=max(maximum,abs(expected-actual_loss));rows+=len(z)
                    loss_checks[d]=dict(rows=rows,max_abs_difference=maximum)
            print('REPLAYED',key,flush=True)
    F.write(dest,dict(models=checks,loss_formula=loss_checks,all_source_probabilities_bitwise_reproduced=True,
                     target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))


def score():
    F.guard();c=E.verify();R.check(E.OUT,'probability_freeze.json')
    dest=R.ROOT/'round30_independent_completion_audit.json'
    if dest.exists(): raise FileExistsError('Preserve audit')
    assert F.read(E.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    report=F.read(E.OUT/'source_report.json');_,ys=E.X.load(c,'all_multi_training')
    predictions={};groups=0
    for arm in c['arms']:
        loaded=[dict(np.load(E.OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in ['relative','absolute']:
            key=arm+'_'+mode;claim=report['reports'][key];t=claim['selections']['balanced_exact']['threshold']
            options=[]
            for v in F.read(E.OUT/'calibration'/f'{key}.json')['candidates']:
                ns=nm=0
                for d in c['sources']:
                    ct=counts(select(transform(mean[d],mode),v['threshold']),ys[d]['valid_labels'].astype(bool))
                    ns+=ct['single_correct'];nm+=ct['multi_correct']
                assert (ns,nm)==(v['single_correct'],v['multi_correct']);options.append((v['threshold'],ns,nm))
            for obj in ['balanced_exact','micro_exact']:
                best=max(options,key=lambda v:((v[1]*60+v[2]*1186) if obj=='balanced_exact' else v[1]+v[2],v[0]))
                v=claim['selections'][obj];assert best==(v['threshold'],v['single_correct'],v['multi_correct'])
            predictions[key]={}
            for view in ['full','stress_0','stress_1']:
                suffix='' if view=='full' else '_'+view;total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};rank=0
                for d in c['sources']:
                    y=ys[d]['valid_labels'].astype(bool);p=transform(mean[d+suffix],mode);pred=select(p,t)
                    for k,v in counts(pred,y).items(): total[k]+=v
                    if view=='full':predictions[key][d]=pred
                    for pr,tr in zip(p,y):
                        if tr.sum()>1:rank+=int(pr[tr].min()>(pr[~tr].max() if (~tr).any() else -1))
                assert total==(claim['full'] if view=='full' else claim['stress'][view]);groups+=1
                if view=='full':
                    g=c['source_gate'];passed=total['single_correct']>=g['min_single_correct'] and total['multi_correct']>=g['min_multi_correct'] and rank>=g['min_attainable_multi']
                    assert rank==claim['attainable_multi'] and passed==claim['source_gate_passed']
                    assert claim['balanced_exact']==total['single_correct']/1186/2+total['multi_correct']/60/2
            for seed,payload in zip(c['seeds'],loaded):
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
                for d in c['sources']:
                    for k,v in counts(select(transform(payload[d],mode),t),ys[d]['valid_labels'].astype(bool)).items():total[k]+=v
                assert total==claim['seeds'][str(seed)];groups+=1
            print('SCORED',key,flush=True)
    rows=[]
    for d in c['sources']:
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];truth=ys[d]['valid_labels'].astype(bool)
        for i,q in enumerate(ys[d]['valid_ids']):
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(truth[i].sum()>1),
                gold=np.flatnonzero(truth[i]).tolist(),predicted={k:np.flatnonzero(v[d][i]).tolist() for k,v in predictions.items()},
                correct={k:bool((v[d][i]==truth[i]).all()) for k,v in predictions.items()}))
    assert rows==F.read(E.OUT/'paired_source_rows.json')
    for mode in ['relative','absolute']:
        a='hierarchy_loss_'+mode;b='scalar_control_'+mode
        for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
            v=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            assert v==report['paired'][mode][name];groups+=1
    a=report['reports']['hierarchy_loss_relative'];b=report['reports']['scalar_control_relative']
    passed=a['source_gate_passed'] and a['balanced_exact']>=b['balanced_exact'] and all(a['stress'][s]['correct']>=b['stress'][s]['correct'] for s in ['stress_0','stress_1'])
    assert report['selected_target_candidate']==('hierarchy_loss_relative' if passed else None)
    F.write(dest,dict(metric_groups=groups,source_rows=len(rows),all_metrics_recomputed=True,source_gate_passed=passed,
                     target_predictions_generated=False,adopted=False,audit_code_sha256=F.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','score']);globals()[p.parse_args().phase]()
