"""Replay checkpoints and independently recount full/stress exact-set outcomes."""
import argparse
import numpy as np
import torch
import experiment_shenzhen_school_reference_mask as D
from audit_shenzhen_round16 import select,counts
from zoning.evaluate import bootstrap

F=D.F; R=D.R; X=D.X


def transform(p,mode):
    p=np.array(p,dtype=np.float64,copy=True)
    if mode=='relative':
        maximum=p.max(1); nonzero=maximum>0
        p[nonzero]/=maximum[nonzero,None]
    return p


def replay():
    F.guard(); c=D.verify(); R.check(D.OUT,'probability_freeze.json')
    dest=D.OUT/'probability_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    assert F.read(R.ROOT/'round18_input_replay_audit.json')['full_and_masked_geometry_bitwise_replayed']
    xs,ys=X.load(c,'all_multi_training'); start=R.M.LinearRetrieval(next(iter(xs.values())))
    history=F.read(D.OUT/'training_history.json'); checks={}
    for arm in c['arms']:
        for seed in c['seeds']:
            key=f'{arm}/{seed}'; ck=D.OUT/'checkpoints'/f'{arm}_{seed}.pt'
            path=D.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(ck)==history[key]['checkpoint_sha256']
            assert F.sha(path)==history[key]['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40
            assert saved['history']==history[key]['loss']
            assert all(np.isfinite([v for v in row.values() if isinstance(v,(int,float))]).all() for row in saved['history'])
            if arm=='full_reference_control':
                assert F.sha(ck)==F.sha(X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt')
                assert saved['protocol_hash']==F.sha(X.OUT/'protocol.json')
            else:
                assert saved['protocol_hash']==F.sha(D.OUT/'protocol.json')
                assert [r['view'] for r in saved['history']]==c['schedule']
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
            model.load_state_dict(saved['model']); model.eval(); probs=dict(np.load(path)); n=0
            expected={d+suffix for d in c['sources'] for suffix in ['', '_train_multi','_stress_0','_stress_1']}
            assert set(probs)==expected
            for d in c['sources']:
                for suffix in ['', '_train_multi','_stress_0','_stress_1']:
                    ids=ys[d]['valid_ids']; x=xs[d]
                    if suffix=='_train_multi':
                        ids=ys[d]['train_ids']; z=ys[d]['labels'][ids]
                        ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
                    elif suffix:x=D.view(x,d,suffix[1:])
                    p=R.probabilities(model,x,ids,c['chunk']) if len(ids) else np.zeros((0,x.school_n),np.float32)
                    np.testing.assert_array_equal(p,probs[d+suffix]); n+=len(p)
            assert n==3832
            checks[key]=dict(replayed_rows=n,steps=40,reused_control=arm=='full_reference_control')
            print('REPLAYED',key,flush=True)
    assert len(checks)==20
    F.write(dest,dict(models=checks,all_source_probabilities_bitwise_reproduced=True,
        full_validation_rows=1246,train_multi_rows=94,stress_validation_rows=2492,
        labels_and_inputs_verified=True,target_predictions_generated=False,target_gold_read=False))


def score():
    F.guard(); c=D.verify(); R.check(D.OUT,'probability_freeze.json')
    dest=R.ROOT/'round18_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    assert F.read(D.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    claims=F.read(D.OUT/'source_report.json'); reports=claims['reports']; _,ys=X.load(c,'all_multi_training')
    loaded={a:[dict(np.load(D.OUT/'probabilities'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    means={a:{k:np.stack([p[k] for p in loaded[a]]).mean(0) for k in loaded[a][0]} for a in c['arms']}
    coords={d:np.load(F.REPO/'data'/d/'arrays.npz')['coords'] for d in c['sources']}
    rows={view:[] for view in ['full','stress_0','stress_1']}; measured={}
    for arm in c['arms']:
        for mode in c['decoders']:
            name=arm+'_'+mode; claim=reports[name]; t=claim['selections']['balanced_exact']['threshold']
            cal=F.read(D.OUT/'calibration'/f'{name}.json')
            # Direct predictions at every saved transition, independent of interval-count code.
            options=[]
            for row in cal['candidates']:
                ns=nm=0
                for d in c['sources']:
                    truth=ys[d]['valid_labels'].astype(bool)
                    ct=counts(select(transform(means[arm][d],mode),row['threshold']),truth)
                    ns+=ct['single_correct']; nm+=ct['multi_correct']
                assert (ns,nm)==(row['single_correct'],row['multi_correct'])
                options.append((row['threshold'],ns,nm))
            for objective in ['micro_exact','balanced_exact']:
                best=max(options,key=lambda z:((z[1]+z[2]) if objective=='micro_exact' else (z[1]*60+z[2]*1186),z[0]))
                chosen=claim['selections'][objective]
                assert best==(chosen['threshold'],chosen['single_correct'],chosen['multi_correct'])
            for view in ['full','stress_0','stress_1','train_multi']:
                suffix='' if view=='full' else '_'+view
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}; attainable=0
                for d in c['sources']:
                    labels=ys[d]['valid_labels']
                    if view=='train_multi':
                        labels=ys[d]['labels'][ys[d]['train_ids']]
                        labels=labels[(labels>=0).all(1)&((labels==1).sum(1)>1)]
                    truth=labels.astype(bool); p=transform(means[arm][d+suffix],mode); pred=select(p,t)
                    for k,v in counts(pred,truth).items():total[k]+=v
                    for pr,tr in zip(p,truth):
                        if tr.sum()>1:attainable+=int(pr[tr].min()>(pr[~tr].max() if (~tr).any() else -1))
                expected=claim['selections']['balanced_exact'] if view=='full' else claim['train_multi'] if view=='train_multi' else claim['stress'][view]
                assert all(expected[k]==v for k,v in total.items())
                measured[name+'/'+view]=dict(**total,attainable_multi=attainable)
            total=measured[name+'/full']; gate=c['source_gate']
            passed=total['single_correct']>=gate['min_single_correct'] and total['multi_correct']>=gate['min_multi_correct'] and total['attainable_multi']>=gate['min_attainable_multi']
            assert passed==claim['source_gate_passed'] and total['attainable_multi']==claim['attainable_multi']
            for seed,payload in zip(c['seeds'],loaded[arm]):
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
                for d in c['sources']:
                    for k,v in counts(select(transform(payload[d],mode),t),ys[d]['valid_labels'].astype(bool)).items():total[k]+=v
                assert total==claim['seeds'][str(seed)]
            print('SCORE_CHECKED',name,measured[name+'/full'],flush=True)
    for view in rows:
        suffix='' if view=='full' else '_'+view
        for d in c['sources']:
            truth=ys[d]['valid_labels'].astype(bool)
            pp={a:select(transform(means[a][d+suffix],'relative'),reports[a+'_relative']['selections']['balanced_exact']['threshold']) for a in c['arms']}
            for i,q in enumerate(ys[d]['valid_ids']):
                rows[view].append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in coords[d][q])),multi=bool(truth[i].sum()>1),
                    truth=np.flatnonzero(truth[i]).tolist(),predictions={a:np.flatnonzero(p[i]).tolist() for a,p in pp.items()},
                    correct={a:bool((p[i]==truth[i]).all()) for a,p in pp.items()}))
    before,after=c['arms']; paired={}; seeds={}
    for view,rr in rows.items():
        paired[view]={}
        groups=[('all',rr),('single',[r for r in rr if not r['multi']]),('multi',[r for r in rr if r['multi']])]+[(d,[r for r in rr if r['d']==d]) for d in c['sources']]
        for label,subset in groups:
            delta=[int(r['correct'][after])-int(r['correct'][before]) for r in subset]
            paired[view][label]=dict(n=len(subset),before_correct=sum(r['correct'][before] for r in subset),after_correct=sum(r['correct'][after] for r in subset),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(subset,delta,seed=20261104,draws=3000))
        seeds[view]={}
        suffix='' if view=='full' else '_'+view
        for j,seed in enumerate(c['seeds']):
            delta=[]
            for d in c['sources']:
                truth=ys[d]['valid_labels'].astype(bool)
                ok={a:(select(transform(loaded[a][j][d+suffix],'relative'),reports[a+'_relative']['selections']['balanced_exact']['threshold'])==truth).all(1) for a in c['arms']}
                delta.extend((ok[after].astype(int)-ok[before].astype(int)).tolist())
            seeds[view][str(seed)]=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100))
    candidate=after+'_relative'; control=before+'_relative'
    selected=candidate if reports[candidate]['source_gate_passed'] and reports[candidate]['balanced_exact']>=reports[control]['balanced_exact'] else None
    assert selected==claims['selected_target_candidate']
    F.write(D.OUT/'paired_source_rows.json',rows)
    F.write(dest,dict(models=20,all_source_probabilities_replayed=True,all_counts_and_transition_scores_recomputed=True,
        measured=measured,paired=paired,seeds=seeds,selected_target_candidate=selected,source_gate_recomputed=True,
        target_predictions_generated=False,independent_validation=False,adopted=False,
        limitation='Source validation reused for development; stress partitions are held out reference configurations, not independent labeled samples.',code_sha256=F.sha(__file__)))
    print('AUDIT_COMPLETE',selected,paired,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['replay','score']); globals()[parser.parse_args().phase]()
