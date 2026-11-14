"""School-wide reference absence training; same labels, parent and40-step budget."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse
import shutil
import pickle
import numpy as np
import torch
import experiment_shenzhen_source_multireferences as X
import build_shenzhen_school_reference_views as V
from shenzhen_school_reference_views import apply_view
from experiment_shenzhen_relative_decode import relative
from shenzhen_exact_calibration import exact_thresholds,calibration_intervals,threshold_metrics

F=X.F;R=X.R
OUT=R.ROOT/'round18_school_reference_mask'
ARMS=['full_reference_control','school_mask_training']
MODES=['absolute','relative']


def view(x,d,name):
    z=dict(np.load(V.OUT/d/f'{name}.npz'));return apply_view(x,z['columns'],z['boundary'])


def prepare():
    F.guard();c=X.verify();R.check(X.OUT,'probability_freeze.json');R.check(V.OUT,'manifest.json')
    if OUT.exists():raise FileExistsError('Preserve experiment')
    links=dict(F.read(V.OUT/'input_links.json')['files'])
    for p,h in links.items():assert F.sha(p)==h,p
    config=dict(sources=c['sources'],targets=c['targets'],seeds=c['seeds'],steps=40,lr=.003,reg=.01,width=32,chunk=64,ranking_coefficient=1.,
        arms=ARMS,decoders=MODES,primary_decoder='relative',primary_objective='balanced_exact',source_gate=c['source_gate'],
        control='Exact round16 all_multi_training checkpoints, source probabilities replayed; same parent,40steps,loss,labels,weights,full-bank source and target inputs.',
        treatment='Same round06 parent;20full steps and20school-masked steps alternating, four fixed training views used5times each.',
        schedule=['full' if s%2==0 else f'train_{(s//2)%4}' for s in range(40)],
        stress='Two held-out source reference partitions, no fitting or threshold selection. Source full validation threshold reused.',
        calibration='Same1246source full-reference validation rows. Same primary relative decoder as round17 candidate; absolute is secondary only.',
        source_selection='Primary relative must pass original source gate and balanced_exact >= matched full_reference_control_relative. No fallback to absolute after failure.',
        target='Unchanged repaired inputs; target positive-only loss unchanged on every step; no target masks.',
        boundary='Recomputed from remaining ORIGINAL rule-known sets; no partial multi-to-single conversion, no added complete training negative boundaries.',
        prior='Random-property and spatial-property masks previously failed adoption; this is school-wide missing support with complete multi source training.',
        target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',config)
    for bank,name in [(X.OUT,'input_freeze.json'),(X.OUT,'probability_freeze.json'),(V.OUT,'manifest.json')]:
        links[str(bank/name)]=F.sha(bank/name);links.update({str(bank/p):h for p,h in F.read(bank/name)['files'].items()})
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_school_reference_views.py',F.REPO/'scripts/experiment_shenzhen_relative_decode.py',F.REPO/'scripts/shenzhen_exact_calibration.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links));F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def source_probabilities(model,xs,ys,c):
    result={}
    for d in c['sources']:
        result[d]=R.probabilities(model,xs[d],ys[d]['valid_ids'],c['chunk'])
        ids=ys[d]['train_ids'];z=ys[d]['labels'][ids];ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
        result[d+'_train_multi']=R.probabilities(model,xs[d],ids,c['chunk']) if len(ids) else np.zeros((0,xs[d].school_n),np.float32)
        for key in ['stress_0','stress_1']:
            result[d+'_'+key]=R.probabilities(model,view(xs[d],d,key),ys[d]['valid_ids'],c['chunk'])
    assert all(np.isfinite(p).all() for p in result.values())
    return result


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():R.check(OUT,'probability_freeze.json');return
    xs,ys=X.load(c,'all_multi_training');start=R.M.LinearRetrieval(next(iter(xs.values())))
    ph=F.sha(OUT/'protocol.json');history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    full=R.frames(c,xs,ys,'new_references_labels',c['sources']);target=R.frames(c,xs,ys,'new_references_labels',R.TARGETS)
    batches={'full':full}
    for key in [f'train_{i}' for i in range(4)]:
        batches[key]=[]
        for d in c['sources']:
            masked={d:view(xs[d],d,key)};batches[key].extend(R.frames(c,masked,ys,'new_references_labels',[d]))
        print('FRAMES',key,len(batches[key]),'GPU_GB',torch.cuda.memory_allocated()/1e9,flush=True)
    for seed in c['seeds']:
        for arm in ARMS:
            key=f'{arm}/{seed}';dest=OUT/'probabilities'/arm/f'{seed}.npz';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt'
            if key in history:
                assert F.sha(dest)==history[key]['probabilities_sha256'] and F.sha(ck)==history[key]['checkpoint_sha256'];continue
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
            if arm=='full_reference_control':
                parent=X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt';saved=torch.load(parent,map_location=R.DEVICE,weights_only=False)
                model.load_state_dict(saved['model']);model.eval();payload=source_probabilities(model,xs,ys,c)
                old=dict(np.load(X.OUT/'probabilities/all_multi_training'/f'{seed}.npz'))
                for k,p in old.items():np.testing.assert_array_equal(p,payload[k])
                ck.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent,ck);losses=saved['history']
            else:
                parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
                model.load_state_dict(parent['model']);opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
                if ck.exists():
                    saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                    model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
                for step in range(begin,c['steps']):
                    model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bt=rt=0.;v=c['schedule'][step]
                    for t in batches[v]+target:
                        logits=model(t);bce=R.masked_bce(logits,t['labels'],t['sample_weight']);rank=R.masked_lsep(logits,t['labels'],t['sample_weight'])
                        loss=bce+c['ranking_coefficient']*rank;assert torch.isfinite(loss);loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                    losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm,view=v))
                    if (step+1)%10==0:
                        R.checkpoint(ck,model,opt,'full',step+1,losses,ph);print('TRAIN',arm,seed,step+1,losses[-1],flush=True)
                model.eval();payload=source_probabilities(model,xs,ys,c)
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),reused_control=arm=='full_reference_control')
            F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['checkpoints','probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve source report')
    _,ys=X.load(c,'all_multi_training');labels=[ys[d]['valid_labels'] for d in c['sources']];reports={}
    old=F.read(R.ROOT/'round17_relative_multireferences/source_report.json')['reports']
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in MODES:
            key=arm+'_'+mode;transform=relative if mode=='relative' else lambda p:p
            ps=[transform(mean[d]) for d in c['sources']];cal=exact_thresholds(ps,labels);v=cal['selections']['balanced_exact'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            passed=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            trainlabels=[]
            for d in c['sources']:
                z=ys[d]['labels'][ys[d]['train_ids']];trainlabels.append(z[(z>=0).all(1)&((z==1).sum(1)>1)])
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,
                train_multi=threshold_metrics([transform(mean[d+'_train_multi']) for d in c['sources']],trainlabels,v['threshold']),
                stress={s:threshold_metrics([transform(mean[d+'_'+s]) for d in c['sources']],labels,v['threshold']) for s in ['stress_0','stress_1']},
                seeds={str(s):threshold_metrics([transform(p[d]) for d in c['sources']],labels,v['threshold']) for s,p in zip(c['seeds'],loaded)})
            if arm=='full_reference_control':assert cal['selections']==old['all_multi_training_'+mode]['selections']
            F.write(OUT/'calibration'/f'{key}.json',cal);print(key,{k:v for k,v in reports[key].items() if k!='seeds'},flush=True)
    candidate='school_mask_training_relative';baseline='full_reference_control_relative'
    selected=candidate if reports[candidate]['source_gate_passed'] and reports[candidate]['balanced_exact']>=reports[baseline]['balanced_exact'] else None
    F.write(OUT/'source_report.json',dict(reports=reports,selected_target_candidate=selected,target_predictions_generated=False,independent_validation=False,adopted=False))
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
