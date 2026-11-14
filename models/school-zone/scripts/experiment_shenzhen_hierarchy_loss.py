"""Round30: matched partial-label hierarchy-loss trial with a source gate."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import experiment_shenzhen_set_evidence as D
from shenzhen_hierarchy_loss import masked_max_constraint_loss, relation_matrix
from experiment_shenzhen_relative_decode import relative, select
from shenzhen_exact_calibration import exact_thresholds, calibration_intervals, threshold_metrics
from zoning.evaluate import bootstrap

F, R, X, S = D.F, D.R, D.X, D.S
OUT = R.ROOT / 'round30_hierarchy_loss'
ARMS = ['scalar_control', 'hierarchy_loss']


def prepare():
    F.guard(); c=D.verify(); R.check(D.OUT,'probability_freeze.json')
    if OUT.exists(): raise FileExistsError('Preserve frozen experiment')
    pre=F.read(R.ROOT/'hierarchical_method_review/supervision_preflight.json')
    assert all(not v['exact_clause_implications'] for v in pre['sources'].values())
    config={k:c[k] for k in ['sources','targets','seeds','steps','lr','reg','width','chunk','ranking_coefficient','source_gate']}
    config.update(arms=ARMS,primary_decoder='relative',primary_objective='balanced_exact',
        intervention='Replace target positive-only masked BCE with masked max-constraint loss. Source identity relations keep BCE exactly unchanged. LSEP and regularizer unchanged.',
        mechanism_limit='No target negatives: cannot directly train down an unknown/wrong target-school logit; positive supervision can delegate to another observed positive.',
        hierarchy='77 proved round27 direct scope edges; graph already transitively closed. Self edges included. Equality cycles allowed. No new policy relations or labels.',
        source_hierarchy='Identity only. Conservative full-scope exact-clause probe found no nontrivial edges; not proof that no local shared scopes exist.',
        control='Reuse audited round24 scalar_control, same full-reference inputs, round06 parent, fresh Adam, 40 steps, 10 seeds, labels, weights, capacity.',
        source_gate_rule='Relative decoder must pass original gate, balanced_exact >= matched control, and exact counts on BOTH held-out support stress views >= control. Absolute results secondary, no fallback.',
        target_rule='No target inference in this script. Passing source gate permits a separately frozen trial with max projection, original university restrictions and identical round29 factual supplements.',
        reference_bank='round16 all_multi_training',
        research='C-HMCNN NeurIPS2020 Eq4 adapted to partial observed labels, not exact reproduction. Paper full hierarchy labels/AU(PRC) differ from our setting.',
        source_validation='1246 reused development rows; not independent confirmation.',
        no_target_gold=True,no_target_polygon=True,no_target_cardinality=True,target_predictions_generated=False,adopted=False)
    F.write(OUT/'protocol.json',config)
    links=dict(F.read(D.OUT/'input_links.json')['files'])
    links.update(pre['files'])
    for p in [R.ROOT/'hierarchical_method_review/supervision_preflight.json',Path(__file__),
              D.OUT/'probability_freeze.json']:
        links[str(p)]=F.sha(p)
    for arm in ['probabilities/scalar_control','checkpoints']:
        for p in (D.OUT/arm).glob('*'):
            if arm=='checkpoints' and not p.name.startswith('scalar_control_'): continue
            links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_hierarchy_loss.py',F.REPO/'tests/test_shenzhen_hierarchy_loss.py']:
        dest=OUT/'code'/p.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('PREPARED',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def train():
    F.guard(); c=verify()
    if (OUT/'probability_freeze.json').exists(): R.check(OUT,'probability_freeze.json'); return
    xs,ys=X.load(c,'all_multi_training'); start=R.M.LinearRetrieval(next(iter(xs.values())))
    batches=[]
    for d in c['sources']+c['targets']:
        rel=None
        if d in c['targets']:
            ids=[s['school_id'] for s in F.read(R.OLD/'inputs'/f'{d}.json')['schools']]
            graph=F.read(R.ROOT/'scope_graph_refresh_review/graphs'/f'{d}.json')
            rel=relation_matrix(ids,graph['edges']).to(R.DEVICE)
        batches.extend((t,rel) for t in R.frames(c,xs,ys,'new_references_labels',[d]))
    ph=F.sha(OUT/'protocol.json'); history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        key=f'hierarchy_loss/{seed}'; ck=OUT/'checkpoints'/f'hierarchy_loss_{seed}.pt'
        dest=OUT/'probabilities/hierarchy_loss'/f'{seed}.npz'
        if key in history:
            assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256']; continue
        begin_time=time.perf_counter()
        parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
        model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE); model.load_state_dict(parent['model'])
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']); begin=0; losses=[]
        if ck.exists():
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False); assert saved['protocol_hash']==ph
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
        for step in range(begin,c['steps']):
            model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bt=rt=0.
            for t,rel in batches:
                z=model(t); y=t['labels']; w=t['sample_weight']
                bce=R.masked_bce(z,y,w) if rel is None else masked_max_constraint_loss(z,y,w,rel)
                rank=R.masked_lsep(z,y,w); loss=bce+c['ranking_coefficient']*rank
                assert torch.isfinite(loss);loss.backward();bt+=float(bce.detach());rt+=float(rank.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
            losses.append(dict(bce=bt,ranking=rt,regularizer=float(reg.detach()),grad_norm=norm))
            if (step+1)%10==0:
                R.checkpoint(ck,model,opt,'full',step+1,losses,ph)
                print('TRAIN',seed,step+1,losses[-1],flush=True)
        model.eval();payload=S.source_probabilities(model,xs,ys,c)
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
        history[key]=dict(checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest),steps=len(losses),
                          elapsed_seconds=time.perf_counter()-begin_time,losses=losses)
        cp=OUT/'probabilities/scalar_control'/f'{seed}.npz';cp.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(D.OUT/'probabilities/scalar_control'/f'{seed}.npz',cp)
        cc=OUT/'checkpoints'/f'scalar_control_{seed}.pt';shutil.copyfile(D.OUT/'checkpoints'/cc.name,cc)
        history[f'scalar_control/{seed}']=dict(reused=True,checkpoint_sha256=F.sha(cc),probabilities_sha256=F.sha(cp))
        F.write(OUT/'training_history.json',history);print('MODEL_COMPLETE',key,flush=True)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists(): raise FileExistsError('Preserve report')
    _,ys=X.load(c,'all_multi_training'); labels=[ys[d]['valid_labels'] for d in c['sources']]
    reports={};predictions={};rows=[]
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in ['relative','absolute']:
            key=arm+'_'+mode; transform=relative if mode=='relative' else lambda p:p
            ps=[transform(mean[d]) for d in c['sources']];cal=exact_thresholds(ps,labels)
            v=cal['selections']['balanced_exact'];g=c['source_gate'];threshold=v['threshold']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            gate=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(gate),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,
                full=threshold_metrics(ps,labels,threshold),
                stress={s:threshold_metrics([transform(mean[d+'_'+s]) for d in c['sources']],labels,threshold) for s in ['stress_0','stress_1']},
                seeds={str(s):threshold_metrics([transform(p[d]) for d in c['sources']],labels,threshold) for s,p in zip(c['seeds'],loaded)})
            predictions[key]=[select(np.asarray(p,np.float64),threshold) for p in ps]
            F.write(OUT/'calibration'/f'{key}.json',cal)
            if arm=='scalar_control': assert reports[key]['selections']==F.read(D.OUT/'source_report.json')['reports'][key]['selections']
            print('REPORT',key,reports[key]['full'],reports[key]['stress'],flush=True)
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
        a='hierarchy_loss_'+mode;b='scalar_control_'+mode;paired[mode]={}
        for name,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rr]
            paired[mode][name]=dict(n=len(rr),before_correct=sum(r['correct'][b] for r in rr),after_correct=sum(r['correct'][a] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    a=reports['hierarchy_loss_relative']; b=reports['scalar_control_relative']
    passed=a['source_gate_passed'] and a['balanced_exact']>=b['balanced_exact'] and all(a['stress'][s]['correct']>=b['stress'][s]['correct'] for s in ['stress_0','stress_1'])
    F.write(OUT/'paired_source_rows.json',rows)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=paired,selected_target_candidate='hierarchy_loss_relative' if passed else None,
                                       target_predictions_generated=False,independent_validation=False,adopted=False))
    print('TARGET_GATE',passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
