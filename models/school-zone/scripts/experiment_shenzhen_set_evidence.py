"""Round24: learned vector pooling vs the matched scalar retrieval control.

Same complete source reference bank, parent, labels, weights, optimizer and steps.
Only a predeclared source-gate pass permits a separately frozen target trial.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import experiment_shenzhen_school_reference_mask as S
from shenzhen_set_evidence import SetEvidenceRetrieval, SCALES
from experiment_shenzhen_relative_decode import relative, select
from shenzhen_exact_calibration import exact_thresholds, calibration_intervals, threshold_metrics
from zoning.evaluate import bootstrap

F = S.F; R = S.R; X = S.X
OUT = R.ROOT / 'round24_set_evidence'
ARMS = ['scalar_control', 'vector_residual']


def prepare():
    F.guard(); c = S.verify(); R.check(S.OUT, 'probability_freeze.json')
    if OUT.exists(): raise FileExistsError('Preserve frozen experiment')
    protocol = dict(sources=c['sources'], targets=c['targets'], seeds=c['seeds'],
        steps=40, lr=.003, reg=.01, width=32, embedding_width=16, chunk=64,
        ranking_coefficient=1., arms=ARMS, primary_decoder='relative',
        primary_objective='balanced_exact', source_gate=c['source_gate'],
        reference_bank='round16 all_multi_training, identical in both arms',
        parent='Same round06 balanced_rank checkpoint per seed; fresh Adam.',
        control='Reuse round18 full_reference_control, identical to round16 all_multi_training; replay separately.',
        intervention='Add a zero-output-initialized residual: phi(existing74d pair-school-context) ->16d, weighted school means plus masses at100/300/1000m, rho with existing33d anchor-context.',
        parameter_matching=False, comparison='Same input and optimization steps, additional representation capacity; not equal FLOPs or equal parameter count.',
        pooling='Original property mass retained; self-property mask retained; no cross-school normalization in new branch. Empty support has zero vector and zero observed mass, not a negative label.',
        loss='Unchanged masked BCE+LSEP, identical complete-source and positive-only target supervision and weights.',
        source_selection='Relative decoder must pass original source gate AND balanced_exact >= scalar_control_relative. Absolute decoder secondary only, no fallback.',
        stress='Two original held-out source support partitions, no training or recalibration on them. Report with full-source selected threshold.',
        validation='Original1246source development rows,1186single and60multi. Reused labels, not independent confirmation.',
        no_target_gold=True, no_target_polygon=True, no_target_cardinality=True,
        target_predictions_generated=False, independent_validation=False, adopted=False,
        research='Deep Sets, NeurIPS2017 main paper and appendix A-I read. Use conditioned invariant representation, not theorem-based guarantee or paper benchmark reproduction.')
    F.write(OUT / 'protocol.json', protocol)
    links = dict(F.read(S.OUT / 'input_links.json')['files'])
    for bank, name in [(S.OUT, 'input_freeze.json'), (S.OUT, 'probability_freeze.json')]:
        links[str(bank/name)] = F.sha(bank/name)
        links.update({str(bank/p): h for p, h in F.read(bank/name)['files'].items()})
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_set_evidence.py',
              F.REPO/'scripts/experiment_shenzhen_school_reference_mask.py',
              F.REPO/'tests/test_shenzhen_set_evidence.py']:
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); links[str(p)] = F.sha(p)
    for p in (R.ROOT/'deepsets_method_review').iterdir():
        if p.is_file(): links[str(p)] = F.sha(p)
    xs, ys = X.load(c, 'all_multi_training'); audit = {}
    for d, x in xs.items():
        np.testing.assert_array_equal(x.self_mask, x.groups[:, None] == x.groups[x.ref_sites][None])
        if d in c['targets']:
            assert not (ys[d]['labels'] == 0).any() and not len(ys[d]['valid_ids'])
        else:
            assert not set(x.groups[ys[d]['train_ids']]) & set(x.groups[ys[d]['valid_ids']])
        audit[d] = dict(reference_pairs=len(x.ref_sites), reference_sites=len(set(x.ref_sites)),
                        train_rows=len(ys[d]['train_ids']), valid_rows=len(ys[d]['valid_ids']))
    F.write(OUT/'input_audit.json', audit)
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'input_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p)
        for p in OUT.rglob('*') if p.is_file()}, target_gold_read=False))
    print('PREPARED', flush=True)


def verify():
    R.check(OUT, 'input_freeze.json')
    for p, h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p) == h, p
    return F.read(OUT/'protocol.json')


def new_model(start, seed, c):
    return SetEvidenceRetrieval(41, 27, start, seed, c['width'], c['embedding_width']).to(R.DEVICE)


def train():
    F.guard(); c = verify()
    if (OUT/'probability_freeze.json').exists():
        R.check(OUT, 'probability_freeze.json'); print('ALL_COMPLETE'); return
    xs, ys = X.load(c, 'all_multi_training')
    start = R.M.LinearRetrieval(next(iter(xs.values())))
    frames = R.frames(c, xs, ys, 'new_references_labels', c['sources']+c['targets'])
    history = F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    ph = F.sha(OUT/'protocol.json')
    print('FRAMES', len(frames), 'DEVICE', R.DEVICE, 'GPU_GB', torch.cuda.memory_allocated()/1e9, flush=True)
    for seed in c['seeds']:
        key = f'vector_residual/{seed}'; dest = OUT/'probabilities/vector_residual'/f'{seed}.npz'
        ck = OUT/'checkpoints'/f'vector_residual_{seed}.pt'
        if key in history:
            assert F.sha(ck) == history[key]['checkpoint_sha256']
            assert F.sha(dest) == history[key]['probabilities_sha256']; continue
        begin_time = time.perf_counter()
        parent = torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt', map_location=R.DEVICE, weights_only=False)
        model = new_model(start, seed, c); model.load_parent(parent['model'])
        control = R.BernoulliRetrieval(41, 27, start, seed, c['width']).to(R.DEVICE)
        control.load_state_dict(parent['model'])
        with torch.no_grad():
            for t in frames: torch.testing.assert_close(model(t), control(t), rtol=0, atol=0)
        print('PARENT_EQUAL_ALL_TRAIN_BATCHES', seed, flush=True)
        base_parameters = sum(p.numel() for p in control.parameters()); del control
        opt = torch.optim.Adam(model.parameters(), lr=c['lr']); begin=0; losses=[]
        if ck.exists():
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False); assert saved['protocol_hash']==ph
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
        for step in range(begin, c['steps']):
            model.train(); opt.zero_grad(); reg=model.regularization(c['reg']);reg.backward();bt=rt=0.
            for t in frames:
                logits=model(t); bce=R.masked_bce(logits,t['labels'],t['sample_weight'])
                rank=R.masked_lsep(logits,t['labels'],t['sample_weight'])
                loss=bce+c['ranking_coefficient']*rank; assert torch.isfinite(loss)
                loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
            losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm))
            if (step+1)%10==0:
                R.checkpoint(ck,model,opt,'full',step+1,losses,ph)
                print('TRAIN',seed,step+1,losses[-1],flush=True)
        model.eval();payload=S.source_probabilities(model,xs,ys,c)
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
        history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),
            elapsed_seconds=time.perf_counter()-begin_time, parent_exact_on_all_training_batches=True,
            base_parameters=base_parameters,candidate_parameters=sum(p.numel() for p in model.parameters()))
        # Matched controls were already replayed by round18; re-audit this round too.
        cp=OUT/'probabilities/scalar_control'/f'{seed}.npz';cp.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(S.OUT/'probabilities/full_reference_control'/f'{seed}.npz',cp)
        ck_control=OUT/'checkpoints'/f'scalar_control_{seed}.pt'
        shutil.copyfile(S.OUT/'checkpoints'/f'full_reference_control_{seed}.pt',ck_control)
        history[f'scalar_control/{seed}']=dict(reused=True,checkpoint_sha256=F.sha(ck_control),probabilities_sha256=F.sha(cp))
        F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve report')
    _,ys=X.load(c,'all_multi_training'); labels=[ys[d]['valid_labels'] for d in c['sources']]
    reports={};predictions={}; rows=[]
    train_labels=[]
    for d in c['sources']:
        z=ys[d]['labels'][ys[d]['train_ids']];train_labels.append(z[(z>=0).all(1)&((z==1).sum(1)>1)])
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in ['relative','absolute']:
            key=arm+'_'+mode; transform=relative if mode=='relative' else lambda p:p
            ps=[transform(mean[d]) for d in c['sources']];cal=exact_thresholds(ps,labels)
            v=cal['selections']['balanced_exact'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            gate=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(gate),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,
                train_multi=threshold_metrics([transform(mean[d+'_train_multi']) for d in c['sources']],train_labels,v['threshold']),
                stress={s:threshold_metrics([transform(mean[d+'_'+s]) for d in c['sources']],labels,v['threshold']) for s in ['stress_0','stress_1']},
                seeds={str(s):threshold_metrics([transform(p[d]) for d in c['sources']],labels,v['threshold']) for s,p in zip(c['seeds'],loaded)})
            predictions[key]=[select(np.asarray(p,np.float64),v['threshold']) for p in ps]
            F.write(OUT/'calibration'/f'{key}.json',cal)
            if arm=='scalar_control':
                old=F.read(S.OUT/'source_report.json')['reports']['full_reference_control_'+mode]
                assert reports[key]['selections']==old['selections']
            print('REPORT',key,{k:v for k,v in reports[key].items() if k!='seeds'},flush=True)
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            y=labels[di][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(y.sum()>1),
                gold=np.flatnonzero(y).tolist(),predicted={k:np.flatnonzero(v[di][i]).tolist() for k,v in predictions.items()},
                correct={k:bool((v[di][i]==y).all()) for k,v in predictions.items()}))
    assert len(rows)==1246 and sum(r['multi'] for r in rows)==60
    paired={}
    for mode in ['relative','absolute']:
        a='vector_residual_'+mode;b='scalar_control_'+mode;paired[mode]={}
        for name,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rr]
            paired[mode][name]=dict(n=len(rr),before_correct=sum(r['correct'][b] for r in rr),after_correct=sum(r['correct'][a] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    candidate='vector_residual_relative';base='scalar_control_relative'
    selected=candidate if reports[candidate]['source_gate_passed'] and reports[candidate]['balanced_exact']>=reports[base]['balanced_exact'] else None
    F.write(OUT/'paired_source_rows.json',rows)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=paired,selected_target_candidate=selected,
        target_predictions_generated=False,independent_validation=False,adopted=False))
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
