"""Round19: replace complete-source BCE+LSEP by ZLPR, all partial losses fixed."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse
import shutil
import numpy as np
import torch
import experiment_shenzhen_school_reference_mask as D
from shenzhen_zlpr import source_loss
from shenzhen_exact_calibration import exact_thresholds,threshold_metrics,calibration_intervals

F=D.F; R=D.R; X=D.X
OUT=R.ROOT/'round19_source_zlpr'
ARMS=['full_reference_control','source_zlpr']


def prepare():
    F.guard(); c=D.verify(); R.check(D.OUT,'probability_freeze.json')
    assert F.read(R.ROOT/'round18_independent_completion_audit.json')['all_source_probabilities_replayed']
    if OUT.exists():raise FileExistsError('Preserve experiment')
    config={k:c[k] for k in ['sources','targets','seeds','steps','lr','reg','width','chunk','source_gate']}
    config.update(arms=ARMS,decoders=['absolute','relative'],primary_decoder='absolute',primary_objective='balanced_exact',
        intervention='Only complete source rows use ZLPR Equation1. All partial source and Shenzhen positive-only rows retain BCE+LSEP.',
        control='Exact round16 all_multi_training checkpoints and source predictions; all inputs, weights, parent and40steps matched.',
        reference_views='All full during fitting. Same2round18 stress views only for evaluation.',
        source_selection='Absolute calibrated balanced-exact must pass original count/ranking gate AND meet full_reference_control_relative balanced-exact. Relative and fixed-zero are secondary, never fallback candidates.',
        source_scale='Use ZLPR as written, no rescaling, loss-coefficient or learning-rate search. Gradient scale and effective regularization are part of this loss substitution.',
        score_transform='Store sigmoid(logits) for compatible monotone decoding, NOT a claim of calibrated probabilities. Fixed-zero diagnostic corresponds to threshold0.5.',
        research=str(R.ROOT/'zlpr_method_review/阅读与适用性.txt'),target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,
        independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',config)
    links=dict(F.read(D.OUT/'input_links.json')['files'])
    for bank,name in [(D.OUT,'input_freeze.json'),(D.OUT,'probability_freeze.json')]:
        R.check(bank,name);links[str(bank/name)]=F.sha(bank/name)
        links.update({str(bank/p):h for p,h in F.read(bank/name)['files'].items()})
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_zlpr.py',F.REPO/'tests/test_shenzhen_zlpr.py',
              R.ROOT/'zlpr_method_review/2208.02955.pdf',R.ROOT/'zlpr_method_review/阅读与适用性.txt',
              R.ROOT/'round18_independent_completion_audit.json',R.ROOT/'round18_input_replay_audit.json']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('ROUND19_FROZEN',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def train():
    F.guard(); c=verify()
    if (OUT/'probability_freeze.json').exists():R.check(OUT,'probability_freeze.json');return
    xs,ys=X.load(c,'all_multi_training'); start=R.M.LinearRetrieval(next(iter(xs.values())))
    full=R.frames(c,xs,ys,'new_references_labels',c['sources']); target=R.frames(c,xs,ys,'new_references_labels',c['targets'])
    ph=F.sha(OUT/'protocol.json'); history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        for arm in ARMS:
            key=f'{arm}/{seed}'; ck=OUT/'checkpoints'/f'{arm}_{seed}.pt'; dest=OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256'];continue
            if arm=='full_reference_control':
                parent=D.OUT/'checkpoints'/f'full_reference_control_{seed}.pt'
                ck.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent,ck)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(D.OUT/'probabilities/full_reference_control'/f'{seed}.npz',dest)
                losses=torch.load(ck,map_location='cpu',weights_only=False)['history']
            else:
                model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
                model.load_state_dict(torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)['model'])
                opt=torch.optim.Adam(model.parameters(),lr=c['lr']); begin=0;losses=[]
                if ck.exists():
                    saved=torch.load(ck,map_location=R.DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); begin=saved['step'];losses=saved['history']
                for step in range(begin,c['steps']):
                    model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward(); st=tt=0.
                    for t in full:
                        logits=model(t);loss=source_loss(logits,t['labels'],t['sample_weight']);assert torch.isfinite(loss)
                        loss.backward();st+=float(loss.detach())
                    for t in target:
                        logits=model(t);loss=R.masked_bce(logits,t['labels'],t['sample_weight'])+R.masked_lsep(logits,t['labels'],t['sample_weight'])
                        assert torch.isfinite(loss);loss.backward();tt+=float(loss.detach())
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                    losses.append(dict(source_loss=st,target_original_loss=tt,regularizer=float(reg.detach()),grad_norm=norm))
                    if (step+1)%10==0:
                        R.checkpoint(ck,model,opt,'full',step+1,losses,ph); print('TRAIN',arm,seed,step+1,losses[-1],flush=True)
                model.eval(); payload=D.source_probabilities(model,xs,ys,c)
                dest.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest,**payload)
            history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),reused_control=arm=='full_reference_control')
            F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    F.guard(); c=verify(); R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve source report')
    _,ys=X.load(c,'all_multi_training'); labels=[ys[d]['valid_labels'] for d in c['sources']]; reports={}
    previous=F.read(D.OUT/'source_report.json')['reports']; trainlabels=[]
    for d in c['sources']:
        z=ys[d]['labels'][ys[d]['train_ids']]; trainlabels.append(z[(z>=0).all(1)&((z==1).sum(1)>1)])
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in c['decoders']:
            key=arm+'_'+mode; transform=D.relative if mode=='relative' else lambda p:p
            ps=[transform(mean[d]) for d in c['sources']]; cal=exact_thresholds(ps,labels); v=cal['selections']['balanced_exact'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            passed=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,
                train_multi=threshold_metrics([transform(mean[d+'_train_multi']) for d in c['sources']],trainlabels,v['threshold']),
                stress={s:threshold_metrics([transform(mean[d+'_'+s]) for d in c['sources']],labels,v['threshold']) for s in ['stress_0','stress_1']},
                fixed_zero=threshold_metrics([mean[d] for d in c['sources']],labels,.5),
                seeds={str(s):threshold_metrics([transform(p[d]) for d in c['sources']],labels,v['threshold']) for s,p in zip(c['seeds'],loaded)})
            if arm=='full_reference_control':assert cal['selections']==previous[key]['selections']
            F.write(OUT/'calibration'/f'{key}.json',cal);print(key,{k:v for k,v in reports[key].items() if k!='seeds'},flush=True)
    candidate='source_zlpr_absolute'; control='full_reference_control_relative'
    selected=candidate if reports[candidate]['source_gate_passed'] and reports[candidate]['balanced_exact']>=reports[control]['balanced_exact'] else None
    F.write(OUT/'source_report.json',dict(reports=reports,selected_target_candidate=selected,target_predictions_generated=False,independent_validation=False,adopted=False))
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
