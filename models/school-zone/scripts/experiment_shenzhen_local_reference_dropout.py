"""Round38: matched near-reference edge regularization, source training only."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
import experiment_shenzhen_explicit_negative_training as PREV
from shenzhen_local_reference_dropout import masked_batch
F,R,S=PREV.F,PREV.R,PREV.S
OUT=R.ROOT/'round38_local_reference_dropout'
ARMS=['scalar_control','local_dropout']


def load(c,arm=None):return PREV.load(c,'features_labels')


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def prepare():
    F.guard();old=PREV.verify();R.check(PREV.OUT,'probability_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve round38')
    c={k:old[k] for k in ['sources','targets','seeds','steps','lr','reg','width','chunk','ranking_coefficient','source_gate']}
    c.update(arms=ARMS,primary_decoder='relative',primary_objective='balanced_exact',radius_m=500.,
        control='Exact round32 features_labels scorer/probabilities; seven official negatives retained.',
        parent='Same round06 balanced_rank parent per seed, fresh Adam and40steps, same regularization anchor as control.',
        schedule=['full' if s%2==0 else 'local' for s in range(40)],
        intervention='On20source steps, drop all direct retrieval columns of a property if any available coordinate is within500m of the query. Rebuild all six reference context statistics on CPU.',
        radius_basis='Existing native500mcontext scale, fixed before candidate training; no radius/step selection on validation or target.',
        retained_evidence='Original query policy, road crossing, learned boundary and name features are unchanged. Known facts are NOT erased; this is direct-channel regularization, not a complete missing-information simulation.',
        labels='All original complete/partial source and partial target labels and weights unchanged, unknowns remain unknown.',
        target_training='All40steps use unmodified target inputs/labels; no target dropout or target Gold.',
        inference='Full original reference bank at validation and deployment, with existing two source school-absence stress views also reported.',
        prior='Earlier property masks and round18whole-school removal failed; no assertion this localized current-input intervention must work.',
        selection='Only local_dropout_relative: original single/multi/attainable gate, balanced_exact and both stress exact counts >= scalar_control. No secondary-arm fallback.',
        target_rule='Separate target freeze only after all source audits/gates; identical cumulative reviewed supplements for controls and candidate.',
        source_validation='Original1246reused development rows, not independent confirmation.',
        no_target_gold=True,no_target_polygon=True,no_target_cardinality=True,target_predictions_generated=False,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',c)
    links=dict(F.read(PREV.OUT/'input_links.json')['files'])
    for name in ['input_freeze.json','probability_freeze.json','source_report.json']:
        links[str(PREV.OUT/name)]=F.sha(PREV.OUT/name)
    for rel,h in F.read(PREV.OUT/'probability_freeze.json')['files'].items():links[str(PREV.OUT/rel)]=h
    for d in c['sources']+c['targets']:
        p=PREV.OUT/'weights/features_labels'/f'{d}.npz';links[str(p)]=F.sha(p)
    previous=F.read(PREV.OUT/'source_report.json')
    F.write(OUT/'reused_control/source_report.json',dict(reports={
        'scalar_control_'+mode:previous['reports']['features_labels_'+mode] for mode in ['relative','absolute']},origin=str(PREV.OUT/'source_report.json')))
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_local_reference_dropout.py',F.REPO/'tests/test_shenzhen_local_reference_dropout.py',
              F.REPO/'scripts/audit_shenzhen_round38.py',F.REPO/'scripts/shenzhen_source_trial_report.py']:
        links[str(p)]=F.sha(p);dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('PREPARED_ROUND38',flush=True)


def frames(c,xs,ys,districts,local=False):
    result=[]
    for d in districts:
        batches=PREV.frames(c,xs,ys,'features_labels',[d])
        if local:
            assert d in c['sources']
            batches=[masked_batch(t,xs[d].groups[xs[d].ref_sites],c['radius_m']) for t in batches]
        result.extend(batches)
    return result


def inputs():
    """Full training-row independent mask and NumPy context verification."""
    F.guard();c=verify();dest=OUT/'input_audit.json'
    if dest.exists():raise FileExistsError('Preserve input audit')
    xs,ys=load(c);audit={}
    for d in c['sources']:
        x=xs[d];xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];refxy=xy[x.ref_sites];rg=x.groups[x.ref_sites]
        unique=np.unique(rg);n=dropped=empty=0;maxdiff=0.
        for t in frames(c,xs,ys,[d]):
            actual=masked_batch(t,rg,c['radius_m']);original=t['valid'].cpu().numpy();expected=original.copy()
            for i,q in enumerate(t['ids']):
                dist=np.linalg.norm(refxy-xy[q],axis=1)
                for group in unique:
                    js=rg==group
                    if (original[i,js] & (dist[js]<=c['radius_m'])).any():expected[i,js]=False
            np.testing.assert_array_equal(actual['valid'].cpu().numpy(),expected)
            for k in ['pair','anchor','gb','mem','labels','sample_weight','ids','groups']:
                assert actual[k] is t[k]
            pair=t['pair'].cpu().numpy().astype(float);anchor=t['anchor'].cpu().numpy().astype(float)
            gb=t['gb'].cpu().numpy().astype(float);mem=t['mem'][:len(rg)].cpu().numpy().astype(float)
            weights=np.exp(gb)[None]*expected;dist=pair[:,:,t['meta'][0]]*300
            p500=weights*np.exp(-dist/500);p1000=weights*np.exp(-dist/1000)
            m500=p500@mem;m1000=p1000@mem
            cross=(pair[:,:,t['meta'][1]].sum(-1)>0).astype(float);cm=(p500*cross)@mem
            ctx=np.stack([m500/np.maximum(m500.sum(1,keepdims=True),1e-12),m1000/np.maximum(m1000.sum(1,keepdims=True),1e-12),
                np.log1p(m500)/4,np.log1p(m1000)/4,cm/np.maximum(m500,1e-12),(expected@mem>0).astype(float)],axis=-1)
            expected_ax=np.concatenate([anchor,ctx],-1);expected_ax[:,:,0]/=10
            rs=x.ref_schools;expected_px=np.concatenate([pair,anchor[:,rs],ctx[:,rs]],-1)
            expected_px[:,:,0]/=10;expected_px[:,:,41]/=10
            for key,value in [('ax',expected_ax),('px',expected_px)]:
                got=actual[key].cpu().numpy();np.testing.assert_allclose(got,value,atol=2e-6,rtol=2e-6)
                maxdiff=max(maxdiff,float(np.abs(got-value).max()))
            n+=len(expected);dropped+=int((original&~expected).sum());empty+=int((~expected.any(1)).sum())
        audit[d]=dict(training_rows=n,dropped_direct_reference_cells=dropped,rows_without_direct_references=empty,max_context_error=maxdiff)
    for d in c['targets']:assert not len(ys[d]['valid_ids'])
    F.write(dest,dict(districts=audit,all_source_training_masks_independently_checked=True,
        all_contexts_independently_recomputed=True,query_and_boundary_evidence_unchanged=True,labels_weights_unchanged=True,
        target_inputs_unchanged=True,target_gold_read=False,audit_code_sha256=F.sha(__file__)))
    print('LOCAL_INPUT_AUDIT_PASSED',audit,flush=True)


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():R.check(OUT,'probability_freeze.json');return
    assert F.read(OUT/'input_audit.json')['all_contexts_independently_recomputed']
    xs,ys=load(c);start=R.M.LinearRetrieval(next(iter(xs.values())))
    full=frames(c,xs,ys,c['sources']);local=frames(c,xs,ys,c['sources'],True);target=frames(c,xs,ys,c['targets'])
    batches={'full':full+target,'local':local+target}
    history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    ph=F.sha(OUT/'protocol.json')
    for seed in c['seeds']:
        arm='local_dropout';key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'probabilities'/arm/f'{seed}.npz'
        if key in history:
            assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256']
        else:
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
            if ck.exists():
                saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bt=rt=0.;view=c['schedule'][step]
                for t in batches[view]:
                    z=model(t);bce=R.masked_bce(z,t['labels'],t['sample_weight']);rank=R.masked_lsep(z,t['labels'],t['sample_weight'])
                    loss=bce+c['ranking_coefficient']*rank;assert torch.isfinite(loss);loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm,view=view))
                if (step+1)%10==0:
                    R.checkpoint(ck,model,opt,view,step+1,losses,ph);print('LOCAL_TRAIN',seed,step+1,losses[-1],flush=True)
            model.eval();payload=S.source_probabilities(model,xs,ys,c)
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),steps=len(losses),losses=losses)
            F.write(OUT/'training_history.json',history);print('LOCAL_MODEL_COMPLETE',seed,flush=True)
        cp=OUT/'probabilities/scalar_control'/f'{seed}.npz';cp.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PREV.OUT/'probabilities/features_labels'/cp.name,cp)
        cc=OUT/'checkpoints'/f'scalar_control_{seed}.pt';shutil.copyfile(PREV.OUT/'checkpoints'/f'features_labels_{seed}.pt',cc)
        history[f'scalar_control/{seed}']=dict(reused=True,checkpoint_sha256=F.sha(cc),probabilities_sha256=F.sha(cp))
        F.write(OUT/'training_history.json',history)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['checkpoints','probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    from shenzhen_source_trial_report import report_trial
    F.guard();c=verify();R.check(OUT,'probability_freeze.json');_,ys=load(c)
    report_trial(F,R,OUT,c,ys,'local_dropout',['scalar_control'],OUT/'reused_control')


def replay():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json');dest=OUT/'probability_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    xs,ys=load(c);start=R.M.LinearRetrieval(next(iter(xs.values())));history=F.read(OUT/'training_history.json');checks={}
    for seed in c['seeds']:
        parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
        for arm in ARMS:
            ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';path=OUT/'probabilities'/arm/f'{seed}.npz';h=history[f'{arm}/{seed}']
            assert F.sha(ck)==h['checkpoint_sha256'] and F.sha(path)==h['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            if arm=='scalar_control':assert F.sha(ck)==F.sha(PREV.OUT/'checkpoints'/f'features_labels_{seed}.pt')
            else:
                assert saved['protocol_hash']==F.sha(OUT/'protocol.json') and saved['history']==h['losses']
                assert [v['view'] for v in saved['history']]==c['schedule']
                assert all(np.isfinite([v[k] for k in ['bce','ranking','regularizer','grad_norm']]).all() for v in saved['history'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(saved['model']);model.eval()
            actual=dict(np.load(path));reproduced=S.source_probabilities(model,xs,ys,c)
            assert actual.keys()==reproduced.keys()
            for k,v in actual.items():np.testing.assert_array_equal(v,reproduced[k])
            n=sum(len(v) for v in actual.values());assert n==3832
            checks[f'{arm}/{seed}']=dict(rows=n,steps=40);print('LOCAL_REPLAYED',arm,seed,flush=True)
    F.write(dest,dict(models=checks,all_source_probabilities_bitwise_reproduced=True,target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))


def score():
    import audit_shenzhen_round38 as A
    A.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','inputs','train','report','replay','score']);globals()[p.parse_args().phase]()
