"""Round32: observed policy negatives, separated from feature repair effects."""
import argparse
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import torch
import experiment_shenzhen_set_evidence as D
import build_shenzhen_explicit_negative_inputs as B
from shenzhen_reference_retraining import fixed_source_weights

F,R,X,S=D.F,D.R,D.X,D.S
OUT=R.ROOT/'round32_explicit_negative_training'
ARMS=['scalar_control','features_only','features_labels']


def load(c,arm):
    xs,ys=X.load(c,'all_multi_training')
    if arm!='scalar_control':
        for d in c['targets']:
            with (B.OUT/d/'inputs.pkl').open('rb') as f:xs[d]=pickle.load(f)
            if arm=='features_labels':ys[d]=dict(np.load(B.OUT/d/'labels.npz'))
    return xs,ys


def prepare():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json');R.check(B.OUT,'manifest.json')
    if OUT.exists():raise FileExistsError('Preserve frozen trial')
    assert F.read(R.ROOT/'explicit_negative_input_completion_audit.json')['only_declared_cells_changed']
    config={k:c[k] for k in ['sources','targets','seeds','steps','lr','reg','width','chunk','ranking_coefficient','source_gate']}
    config.update(arms=ARMS,primary_decoder='relative',primary_objective='balanced_exact',
        control='Reuse audited round24 scalar_control; identical source full-reference bank, round06 parent,40steps,10seeds,optimizer and model capacity.',
        features_only='Four previously missing negative flags in target school-policy columns; labels unchanged.',
        features_labels='Same repaired flags plus7 source-proven negative labels. Six rows already have observed positive labels; one new negative-only row. No unknown assumed negative.',
        loss='Unchanged masked BCE plus LSEP over observed positive-negative pairs only. Six target rows acquire ranking supervision; negative-only row gets BCE only.',
        weights='Original denominator5211 and10active districts; old source and target weights exactly preserved; new independent property receives10/5211.',
        source_selection='Only features_labels may advance: original source gate AND balanced_exact >= BOTH scalar_control and features_only AND both stress exact counts >= BOTH controls. No fallback to features_only or absolute decoder.',
        target_rule='Only after source pass, separately freeze target predictions with identical round31 factual supplements and exclusions for all comparators; compare matched controls and959/1761 development incumbent.',
        source_validation='Original1246development rows,1186single60multi. Reused, not independent confirmation.',
        no_target_gold=True,no_target_polygon=True,no_target_cardinality=True,target_predictions_generated=False,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',config)
    links=dict(F.read(D.OUT/'input_links.json')['files'])
    for folder,name in [(B.OUT,'manifest.json'),(D.OUT,'probability_freeze.json')]:
        links[str(folder/name)]=F.sha(folder/name)
        links.update({str(folder/p):h for p,h in F.read(folder/name)['files'].items()})
    links.update(F.read(B.OUT/'input_links.json')['files'])
    for p in [R.ROOT/'explicit_negative_input_completion_audit.json',D.OUT/'source_report.json']:
        links[str(p)]=F.sha(p)
    bx,by=load(c,'scalar_control');_,baseline=R.load(c,'continued_rank');audits={}
    for arm in ARMS:
        xs,ys=load(c,arm);weights,wa=fixed_source_weights({d:x.groups for d,x in xs.items()},baseline,ys,c['sources'])
        checks={}
        for d,x in xs.items():
            oldw=np.load(R.OUT/'weights'/f'new_references_labels_{d}.npz')['weights'];ids=by[d]['train_ids']
            np.testing.assert_array_equal(weights[d][ids],oldw[ids])
            if arm!='features_labels' or d in c['sources']:np.testing.assert_array_equal(weights[d],oldw)
            if d in c['sources']:
                for k in by[d]:np.testing.assert_array_equal(ys[d][k],by[d][k])
            else:assert not len(ys[d]['valid_ids'])
            np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
            path=OUT/'weights'/arm/f'{d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,weights=weights[d])
            checks[d]=dict(train_rows=len(ys[d]['train_ids']),positive_cells=int((ys[d]['labels']==1).sum()),negative_cells=int((ys[d]['labels']==0).sum()),
                           old_weights_identical=True,new_weighted_rows=np.flatnonzero((weights[d]>0)&(oldw==0)).tolist())
        audits[arm]=dict(weights=wa,districts=checks)
    F.write(OUT/'input_weight_audit.json',audits)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_source_trial_report.py',F.REPO/'scripts/shenzhen_partial_model.py',F.REPO/'scripts/shenzhen_balanced_ranking.py']:
        cp=OUT/'code'/p.name;cp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,cp);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('PREPARED',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def frames(c,xs,ys,arm,districts):
    result=[]
    for d in districts:
        ids=ys[d]['train_ids'];w=np.load(OUT/'weights'/arm/f'{d}.npz')['weights']
        for lo in range(0,len(ids),c['chunk']):
            q=ids[lo:lo+c['chunk']];t=R.M.batch(xs[d],q)
            t['labels']=torch.from_numpy(ys[d]['labels'][q]).to(R.DEVICE)
            t['sample_weight']=torch.tensor(w[q],dtype=torch.float32,device=R.DEVICE);result.append(t)
    return result


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():R.check(OUT,'probability_freeze.json');return
    inputs={a:load(c,a) for a in ARMS};xs,ys=inputs['scalar_control'];start=R.M.LinearRetrieval(next(iter(xs.values())))
    source=frames(c,xs,ys,'scalar_control',c['sources'])
    batches={a:source+frames(c,*inputs[a],a,c['targets']) for a in ARMS[1:]}
    ph=F.sha(OUT/'protocol.json');history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
        for arm in ARMS[1:]:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256'];continue
            began=time.perf_counter();model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
            if ck.exists():
                saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bt=rt=0.
                for t in batches[arm]:
                    z=model(t);y=t['labels'];w=t['sample_weight'];bce=R.masked_bce(z,y,w);rank=R.masked_lsep(z,y,w)
                    loss=bce+c['ranking_coefficient']*rank;assert torch.isfinite(loss);loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm))
                if (step+1)%10==0:
                    R.checkpoint(ck,model,opt,'full',step+1,losses,ph);print('TRAIN',arm,seed,step+1,losses[-1],flush=True)
            model.eval();payload=S.source_probabilities(model,*inputs[arm],c)
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),steps=len(losses),elapsed_seconds=time.perf_counter()-began,losses=losses)
            F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
        cp=OUT/'probabilities/scalar_control'/f'{seed}.npz';cp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(D.OUT/'probabilities/scalar_control'/cp.name,cp)
        ck=OUT/'checkpoints'/f'scalar_control_{seed}.pt';shutil.copyfile(D.OUT/'checkpoints'/ck.name,ck)
        history[f'scalar_control/{seed}']=dict(reused=True,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(cp));F.write(OUT/'training_history.json',history)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    from shenzhen_source_trial_report import report_trial
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    _,ys=load(c,'scalar_control');report_trial(F,R,OUT,c,ys,'features_labels',['scalar_control','features_only'],D.OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
