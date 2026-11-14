"""Replay all source checkpoints and audit exact sets, errors, and frozen gate."""
import numpy as np
import torch
import experiment_shenzhen_source_multireferences as X
from zoning.evaluate import bootstrap

F=X.F;R=X.R


def select(p,t):
    p=np.asarray(p,np.float64);v=p>=t;empty=~v.any(1);v[empty,p[empty].argmax(1)]=True
    return v


def counts(pred,truth):
    ok=(pred==truth).all(1);multi=truth.sum(1)>1
    return dict(n=len(truth),correct=int(ok.sum()),single_correct=int((ok&~multi).sum()),multi_correct=int((ok&multi).sum()),
        fp=int((pred&~truth).sum()),fn=int((~pred&truth).sum()))


def replay():
    F.guard();c=X.verify();R.check(X.OUT,'probability_freeze.json')
    if (X.OUT/'probability_replay_audit.json').exists():raise FileExistsError('Preserve replay')
    assert (X.OUT/'independent_input_audit.json').exists()
    ph=F.sha(X.OUT/'protocol.json');history=F.read(X.OUT/'training_history.json');checks={}
    for arm in c['arms']:
        xs,ys=X.load(c,arm);start=R.M.LinearRetrieval(next(iter(xs.values())))
        for seed in c['seeds']:
            key=f'{arm}/{seed}';p=X.OUT/'checkpoints'/f'{arm}_{seed}.pt';npz=X.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(p)==history[key]['checkpoint_sha256'] and F.sha(npz)==history[key]['probabilities_sha256']
            saved=torch.load(p,map_location=R.DEVICE,weights_only=False);parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert saved['step']==c['steps'] and len(saved['history'])==c['steps'] and saved['protocol_hash']==ph
            assert all(np.isfinite(list(row.values())).all() for row in saved['history'])
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(saved['model']);model.eval()
            probs=dict(np.load(npz));assert set(probs)==set(c['sources'])|{d+'_train_multi' for d in c['sources']}
            n=0
            for d in c['sources']:
                ids=ys[d]['valid_ids'];p=R.probabilities(model,xs[d],ids,c['chunk']);np.testing.assert_array_equal(p,probs[d]);n+=len(p)
                ids=ys[d]['train_ids'];y=ys[d]['labels'][ids];ids=ids[(y>=0).all(1)&((y==1).sum(1)>1)]
                p=R.probabilities(model,xs[d],ids,c['chunk']) if len(ids) else np.zeros((0,xs[d].school_n),np.float32)
                np.testing.assert_array_equal(p,probs[d+'_train_multi']);n+=len(p)
            assert n==1340
            checks[key]=dict(replayed_rows=n,steps=saved['step'],same_regularization_anchor=True)
            print('REPLAYED',key,flush=True)
    assert len(checks)==30
    F.write(X.OUT/'probability_replay_audit.json',dict(models=checks,all_source_probabilities_bitwise_reproduced=True,
        source_validation_rows_per_model=1246,source_multi_training_rows_per_model=94,target_prediction_rows=0,target_gold_read=False))


def score():
    F.guard();c=X.verify();R.check(X.OUT,'probability_freeze.json')
    assert F.read(X.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    report=F.read(X.OUT/'source_report.json');claims=report['reports'];_,ys=X.load(c,'control');means={};loaded={};rows=[]
    for arm in c['arms']:
        loaded[arm]=[dict(np.load(X.OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        means[arm]={d:np.stack([p[d] for p in loaded[arm]]).mean(0) for d in c['sources']}
    for d in c['sources']:
        coords=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];truth=ys[d]['valid_labels'].astype(bool)
        pred={a:select(means[a][d],claims[a]['selections']['balanced_exact']['threshold']) for a in c['arms']}
        for i,q in enumerate(ys[d]['valid_ids']):
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in coords[q])),multi=bool(truth[i].sum()>1),
                truth=np.flatnonzero(truth[i]).tolist(),predictions={a:np.flatnonzero(pred[a][i]).tolist() for a in c['arms']},
                correct={a:bool((pred[a][i]==truth[i]).all()) for a in c['arms']}))
    assert len(rows)==1246 and sum(r['multi'] for r in rows)==60
    paired={};seedpaired={}
    for arm in c['arms']:
        chosen=claims[arm]['selections']['balanced_exact'];summary={}
        for objective in ['balanced_exact','micro_exact']:
            t=claims[arm]['selections'][objective]['threshold'];totals={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
            for d in c['sources']:
                counted=counts(select(means[arm][d],t),ys[d]['valid_labels'].astype(bool))
                for k,v in counted.items():totals[k]+=v
            assert all(totals[k]==claims[arm]['selections'][objective][k] for k in totals)
        separable=0
        for d in c['sources']:
            for p,y in zip(means[arm][d],ys[d]['valid_labels'].astype(bool)):
                if y.sum()>1:separable+=int(p[y].min()>(p[~y].max() if (~y).any() else -1))
        assert separable==claims[arm]['attainable_multi']
        g=c['source_gate'];gate=chosen['single_correct']>=g['min_single_correct'] and chosen['multi_correct']>=g['min_multi_correct'] and separable>=g['min_attainable_multi']
        assert bool(gate)==claims[arm]['source_gate_passed']
        for label,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][arm])-int(r['correct']['control']) for r in rr]
            summary[label]=dict(n=len(rr),before_correct=sum(r['correct']['control'] for r in rr),after_correct=sum(r['correct'][arm] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
        paired[arm]=summary;seedpaired[arm]={}
        for j,seed in enumerate(c['seeds']):
            before=[];after=[];nsc=nmc=0
            for d in c['sources']:
                truth=ys[d]['valid_labels'].astype(bool)
                p=select(loaded[arm][j][d],chosen['threshold']);b=select(loaded['control'][j][d],claims['control']['selections']['balanced_exact']['threshold'])
                ok=(p==truth).all(1);bk=(b==truth).all(1);before.extend(bk.tolist());after.extend(ok.tolist())
                nsc+=int((ok&(truth.sum(1)==1)).sum());nmc+=int((ok&(truth.sum(1)>1)).sum())
            delta=np.array(after,int)-np.array(before,int)
            seedpaired[arm][str(seed)]=dict(corrected=int((delta==1).sum()),harmed=int((delta==-1).sum()),correct=sum(after),single_correct=nsc,multi_correct=nmc)
            assert sum(after)==claims[arm]['seeds'][str(seed)]['correct']
        print('AUDITED',arm,summary['all'],summary['multi'],flush=True)
    expected=next((a for a in ['all_multi_training','complete_existing'] if claims[a]['source_gate_passed'] and claims[a]['balanced_exact']>=claims['control']['balanced_exact']),None)
    assert expected==report['selected_target_candidate']
    F.write(X.OUT/'paired_source_rows.json',rows)
    F.write(R.ROOT/'round16_independent_completion_audit.json',dict(models=30,source_rows=1246,source_multi_rows=60,
        independent_input_audit_passed=True,all_source_probabilities_replayed=True,all_source_counts_recomputed=True,source_gate_recomputed=True,
        paired_metrics=paired,seeds=seedpaired,selected_target_candidate=expected,target_predictions_generated=False,adopted=False,independent_validation=False,
        limitation='Reused development holdout; calibration and intervals do not establish independent validation.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','score']);globals()[p.parse_args().phase]()
