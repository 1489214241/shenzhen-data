"""Independent round19 loss identity, checkpoint replay and exact-set audit."""
import argparse
import numpy as np
import torch
import experiment_shenzhen_zlpr as D
from audit_shenzhen_round18 import transform
from audit_shenzhen_round16 import select,counts
from zoning.evaluate import bootstrap

F=D.F;R=D.R;X=D.X


def inputs():
    F.guard();c=D.verify();xs,ys=X.load(c,'all_multi_training');out={}
    dest=D.OUT/'loss_input_audit.json'
    if dest.exists():raise FileExistsError('Preserve loss audit')
    for d in c['sources']+c['targets']:
        ids=ys[d]['train_ids'];labels=ys[d]['labels'][ids]
        # Fixed deterministic scores check all observed/unknown label patterns;
        # this is an algebra/input audit, not a fitted model or synthetic label.
        s=torch.tensor(np.sin(np.arange(labels.size).reshape(labels.shape)/13),dtype=torch.float64,requires_grad=True)
        y=torch.tensor(labels,dtype=s.dtype); w=torch.tensor(np.load(R.OUT/'weights'/f'new_references_labels_{d}.npz')['weights'][ids],dtype=s.dtype)
        full=(labels>=0).all(1);actual=D.source_loss(s,y,w)
        if full.any():
            a=s[torch.tensor(full)];z=y[torch.tensor(full)]
            independent=torch.log1p(torch.exp(-a).masked_fill(z!=1,0).sum(1))+torch.log1p(torch.exp(a).masked_fill(z!=0,0).sum(1))
            expected=(independent*w[torch.tensor(full)]).sum()
        else:expected=s.sum()*0
        partial=~torch.tensor(full)
        expected=expected+R.masked_bce(s[partial],y[partial],w[partial])+R.masked_lsep(s[partial],y[partial],w[partial])
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        grad=torch.autograd.grad(actual,s)[0];assert (grad[y<0]==0).all()
        if d in c['targets']:
            assert not full.any() and not (labels==0).any() and len(ys[d]['valid_ids'])==0
            torch.testing.assert_close(actual,R.masked_bce(s,y,w)+R.masked_lsep(s,y,w),rtol=1e-12,atol=1e-12)
        out[d]=dict(training_rows=len(ids),complete_rows=int(full.sum()),partial_rows=int((~full).sum()),
                    positive_cells=int((labels==1).sum()),negative_cells=int((labels==0).sum()),unknown_cells=int((labels<0).sum()),
                    loss_identity_verified=True,unknown_gradients_zero=True)
    assert sum(v['complete_rows'] for k,v in out.items() if k in c['sources'])==4884
    F.write(dest,dict(districts=out,target_loss_identical=True,all_input_links_verified=True,source_labels_and_weights_unchanged=True,
        target_gold_read=False,code_sha256=F.sha(__file__)))
    print('LOSS_INPUT_AUDIT',out,flush=True)


def replay():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    dest=D.OUT/'probability_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    assert F.read(D.OUT/'loss_input_audit.json')['target_loss_identical']
    xs,ys=X.load(c,'all_multi_training');start=R.M.LinearRetrieval(next(iter(xs.values())))
    history=F.read(D.OUT/'training_history.json');checks={}
    for arm in c['arms']:
        for seed in c['seeds']:
            key=f'{arm}/{seed}';ck=D.OUT/'checkpoints'/f'{arm}_{seed}.pt';path=D.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(path)==history[key]['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40 and saved['history']==history[key]['loss']
            assert all(np.isfinite(list(row.values())).all() for row in saved['history'])
            if arm=='full_reference_control':
                assert F.sha(ck)==F.sha(X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt')
                assert saved['protocol_hash']==F.sha(X.OUT/'protocol.json')
            else:assert saved['protocol_hash']==F.sha(D.OUT/'protocol.json')
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(saved['model']);model.eval()
            ps=dict(np.load(path));expected={d+s for d in c['sources'] for s in ['', '_train_multi','_stress_0','_stress_1']}
            assert set(ps)==expected;n=0
            for d in c['sources']:
                for suffix in ['', '_train_multi','_stress_0','_stress_1']:
                    ids=ys[d]['valid_ids'];x=xs[d]
                    if suffix=='_train_multi':
                        ids=ys[d]['train_ids'];z=ys[d]['labels'][ids];ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
                    elif suffix:x=D.D.view(x,d,suffix[1:])
                    p=R.probabilities(model,x,ids,c['chunk']) if len(ids) else np.zeros((0,x.school_n),np.float32)
                    np.testing.assert_array_equal(p,ps[d+suffix]);n+=len(p)
            assert n==3832;checks[key]=dict(rows=n,steps=40);print('REPLAYED',key,flush=True)
    F.write(dest,dict(models=checks,all_source_probabilities_bitwise_reproduced=True,target_predictions_generated=False,target_gold_read=False))


def score():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    dest=R.ROOT/'round19_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    assert F.read(D.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    source=F.read(D.OUT/'source_report.json');claims=source['reports'];_,ys=X.load(c,'all_multi_training')
    loaded={a:[dict(np.load(D.OUT/'probabilities'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    mean={a:{k:np.stack([p[k] for p in loaded[a]]).mean(0) for k in loaded[a][0]} for a in c['arms']}
    actual={};views=['full','stress_0','stress_1'];predictions={v:{} for v in views}
    for arm in c['arms']:
        for mode in c['decoders']:
            key=arm+'_'+mode;claim=claims[key];t=claim['selections']['balanced_exact']['threshold'];cal=F.read(D.OUT/'calibration'/f'{key}.json')
            options=[]
            for row in cal['candidates']:
                ns=nm=0
                for d in c['sources']:
                    ct=counts(select(transform(mean[arm][d],mode),row['threshold']),ys[d]['valid_labels'].astype(bool))
                    ns+=ct['single_correct'];nm+=ct['multi_correct']
                assert (ns,nm)==(row['single_correct'],row['multi_correct']);options.append((row['threshold'],ns,nm))
            for obj in ['balanced_exact','micro_exact']:
                best=max(options,key=lambda z:((z[1]*60+z[2]*1186) if obj=='balanced_exact' else z[1]+z[2],z[0]))
                v=claim['selections'][obj];assert best==(v['threshold'],v['single_correct'],v['multi_correct'])
            for view in views+['train_multi','fixed_zero']:
                suffix='' if view in ['full','fixed_zero'] else '_'+view
                tot={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};rank=0
                if view in views:predictions[view][key]={}
                for d in c['sources']:
                    labels=ys[d]['valid_labels']
                    if view=='train_multi':
                        labels=ys[d]['labels'][ys[d]['train_ids']];labels=labels[(labels>=0).all(1)&((labels==1).sum(1)>1)]
                    truth=labels.astype(bool);p=transform(mean[arm][d+suffix],mode if view!='fixed_zero' else 'absolute');pred=select(p,t if view!='fixed_zero' else .5)
                    for k,v in counts(pred,truth).items():tot[k]+=v
                    for pr,tr in zip(p,truth):
                        if tr.sum()>1:rank+=int(pr[tr].min()>(pr[~tr].max() if (~tr).any() else -1))
                    if view in views:predictions[view][key][d]=pred
                expected=claim['selections']['balanced_exact'] if view=='full' else claim[view] if view in ['train_multi','fixed_zero'] else claim['stress'][view]
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
    rows={};paired={};candidate='source_zlpr_absolute';baselines=['full_reference_control_absolute','full_reference_control_relative']
    for view in views:
        rows[view]=[];paired[view]={}
        for d in c['sources']:
            xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];truth=ys[d]['valid_labels'].astype(bool)
            for i,q in enumerate(ys[d]['valid_ids']):
                rows[view].append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(truth[i].sum()>1),
                    truth=np.flatnonzero(truth[i]).tolist(),predictions={k:np.flatnonzero(v[d][i]).tolist() for k,v in predictions[view].items()},
                    correct={k:bool((v[d][i]==truth[i]).all()) for k,v in predictions[view].items()}))
        for baseline in baselines:
            paired[view][baseline]={};rr=rows[view]
            groups=[('all',rr),('single',[r for r in rr if not r['multi']]),('multi',[r for r in rr if r['multi']])]+[(d,[r for r in rr if r['d']==d]) for d in c['sources']]
            for label,subset in groups:
                delta=[int(r['correct'][candidate])-int(r['correct'][baseline]) for r in subset]
                paired[view][baseline][label]=dict(n=len(subset),before_correct=sum(r['correct'][baseline] for r in subset),after_correct=sum(r['correct'][candidate] for r in subset),
                    corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(subset,delta,seed=20261104,draws=3000))
    selected=candidate if claims[candidate]['source_gate_passed'] and claims[candidate]['balanced_exact']>=claims['full_reference_control_relative']['balanced_exact'] else None
    assert selected==source['selected_target_candidate']
    F.write(D.OUT/'paired_source_rows.json',rows)
    F.write(dest,dict(all_source_probabilities_replayed=True,all_counts_and_transition_scores_recomputed=True,actual=actual,paired=paired,
        source_gate_recomputed=True,selected_target_candidate=selected,target_predictions_generated=False,
        independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('AUDIT_COMPLETE',selected,paired['full'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inputs','replay','score']);globals()[p.parse_args().phase]()
