"""Matched source-reference intervention; freeze and gate source results before target inference."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import pickle
import shutil
import gc
import numpy as np
import torch
import experiment_shenzhen_reference_retraining as R
import build_shenzhen_source_reference_completion as C
import build_shenzhen_source_multireferences as B
from shenzhen_exact_calibration import exact_thresholds,calibration_intervals,threshold_metrics

F=R.F
OUT=R.ROOT/'round16_source_multireferences'
TARGET=F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
ARMS=['control','complete_existing','all_multi_training']
BANKS={'complete_existing':C.OUT,'all_multi_training':B.OUT}


def load(c,arm):
    xs,ys=R.load(c,'new_references_labels')
    for d in c['sources']:
        if arm!='control':
            with (BANKS[arm]/d/'inputs.pkl').open('rb') as h:xs[d]=pickle.load(h)
            for key,a in dict(np.load(BANKS[arm]/d/'labels.npz')).items():np.testing.assert_array_equal(ys[d][key],a)
    for d in R.TARGETS:
        with (TARGET/d/'inputs.pkl').open('rb') as h:xs[d]=pickle.load(h)
    return xs,ys


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve experiment')
    c=R.verify();links={}
    for bank in [C.OUT,B.OUT,TARGET]:
        R.check(bank,'manifest.json');links[str(bank/'manifest.json')]=F.sha(bank/'manifest.json')
        links.update({str(bank/rel):h for rel,h in F.read(bank/'manifest.json')['files'].items()})
        if (bank/'input_links.json').exists():links.update(F.read(bank/'input_links.json')['files'])
    gate=F.read(R.ROOT/'round11_relation_capacity_probe/protocol.json')['source_gate']
    protocol=dict(sources=c['sources'],targets=R.TARGETS,seeds=c['seeds'],arms=ARMS,
        steps=40,lr=.003,reg=.01,width=32,chunk=64,ranking_coefficient=1.,source_gate=gate,
        parent='Same round06 balanced_rank checkpoint per seed; fresh Adam; same regularization anchor.',
        intervention='Source reference bank only: unchanged control, co-label completion at existing sites, all eligible complete multi training sites.',
        fixed='Identical training and validation labels, district weights, optimization budget, target partial labels and repaired target features.',
        source_metric='Exact complete set; balanced_exact primary, micro_exact secondary. Original1246source validation rows,60multi.',
        target_inference='None during this experiment. Requires source gate AND primary balanced_exact no worse than matched control; separately frozen target test only.',
        next_test='Select all_multi_training if it passes; otherwise complete_existing if it passes. No target-dependent selection.',
        no_target_gold=True,no_target_polygon=True,no_target_cardinality=True,partial_unknown_not_negative=True,
        source_holdout_reused=True,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',protocol)
    bx,by=load(protocol,'control');audit={}
    for arm in ARMS:
        xs,ys=load(protocol,arm);audit[arm]={}
        for d,x in xs.items():
            for k in by[d]:np.testing.assert_array_equal(by[d][k],ys[d][k])
            np.testing.assert_array_equal(x.anchor,bx[d].anchor)
            np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
            if d in R.TARGETS:
                np.testing.assert_array_equal(x.pair,bx[d].pair)
                assert not (ys[d]['labels']==0).any() and not len(ys[d]['valid_ids'])
            audit[arm][d]=dict(reference_pairs=len(x.ref_sites),reference_sites=len(set(x.ref_sites)),school_coverage=len(set(x.ref_schools)),train_rows=len(ys[d]['train_ids']),valid_rows=len(ys[d]['valid_ids']))
    for d in c['sources']+R.TARGETS:
        p=R.OUT/'weights'/f'new_references_labels_{d}.npz';links[str(p)]=F.sha(p)
    for seed in c['seeds']:
        p=R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt';links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_partial_model.py',F.REPO/'scripts/shenzhen_balanced_ranking.py',F.REPO/'scripts/shenzhen_exact_calibration.py',F.REPO/'scripts/experiment_shenzhen_reference_retraining.py',F.REPO/'scripts/experiment_shenzhen_partial_labels.py',F.REPO/'zoning/model.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_audit.json',audit);F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('PREPARED',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():
        R.check(OUT,'probability_freeze.json');print('ALL_COMPLETE');return
    history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    ph=F.sha(OUT/'protocol.json')
    # One arm cached at a time keeps the full-input training under laptop GPU memory.
    for arm in ARMS:
        xs,ys=load(c,arm);batches=R.frames(c,xs,ys,'new_references_labels',c['sources']+R.TARGETS)
        start=R.M.LinearRetrieval(next(iter(xs.values())))
        print('FRAMES',arm,len(batches),'GPU_GB',torch.cuda.memory_allocated()/1e9,flush=True)
        for seed in c['seeds']:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256']
                continue
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
            if ck.exists():
                saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bt=rt=0.
                for t in batches:
                    logits=model(t);bce=R.masked_bce(logits,t['labels'],t['sample_weight']);rank=R.masked_lsep(logits,t['labels'],t['sample_weight'])
                    loss=bce+c['ranking_coefficient']*rank;assert torch.isfinite(loss);loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm))
                if (step+1)%10==0:
                    R.checkpoint(ck,model,opt,'full',step+1,losses,ph);print('TRAIN',arm,seed,step+1,losses[-1],flush=True)
            payload={};model.eval()
            for d in c['sources']:
                payload[d]=R.probabilities(model,xs[d],ys[d]['valid_ids'],c['chunk'])
                ids=ys[d]['train_ids'];z=ys[d]['labels'][ids];ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
                payload[d+'_train_multi']=R.probabilities(model,xs[d],ids,c['chunk']) if len(ids) else np.zeros((0,xs[d].school_n),np.float32)
            assert all(np.isfinite(a).all() for a in payload.values())
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest))
            F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
        del batches,xs,ys,start,model,opt,parent;gc.collect();torch.cuda.empty_cache()
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve source outcomes')
    _,ys=load(c,'control');labels=[ys[d]['valid_labels'] for d in c['sources']];reports={}
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([v[k] for v in loaded]).mean(0) for k in loaded[0]}
        ps=[mean[d] for d in c['sources']];cal=exact_thresholds(ps,labels);g=c['source_gate'];v=cal['selections']['balanced_exact']
        attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
        train_labels=[]
        for d in c['sources']:
            z=ys[d]['labels'][ys[d]['train_ids']];train_labels.append(z[(z>=0).all(1)&((z==1).sum(1)>1)])
        ns,nm=cal['source_single'],cal['source_multi'];balanced=v['single_correct']/ns/2+v['multi_correct']/nm/2
        gate=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
        reports[arm]=dict(selections=cal['selections'],balanced_exact=balanced,attainable_multi=attainable,source_gate_passed=bool(gate),
            train_multi=threshold_metrics([mean[d+'_train_multi'] for d in c['sources']],train_labels,v['threshold']),
            seeds={str(s):threshold_metrics([p[d] for d in c['sources']],labels,v['threshold']) for s,p in zip(c['seeds'],loaded)},
            districts={d:threshold_metrics([mean[d]],[ys[d]['valid_labels']],v['threshold']) for d in c['sources']})
        F.write(OUT/'calibration'/f'{arm}.json',cal)
    for arm in ARMS:
        reports[arm]['eligible_for_target']=bool(arm!='control' and reports[arm]['source_gate_passed'] and reports[arm]['balanced_exact']>=reports['control']['balanced_exact'])
    selected=next((a for a in ['all_multi_training','complete_existing'] if reports[a]['eligible_for_target']),None)
    F.write(OUT/'source_report.json',dict(reports=reports,selected_target_candidate=selected,target_scored=False,adopted=False,independent_validation=False))
    for arm,r in reports.items():print(arm,{k:v for k,v in r.items() if k not in ['seeds','districts']},flush=True)
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','report']);globals()[parser.parse_args().phase]()
