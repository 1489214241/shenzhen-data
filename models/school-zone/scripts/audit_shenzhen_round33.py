"""Round33 independent positive-gradient, source replay and score checks."""
from pathlib import Path
import argparse
import numpy as np
import torch
import experiment_shenzhen_late_text_training as E
from audit_shenzhen_round16 import select, counts
from audit_shenzhen_round18 import transform
from zoning.evaluate import bootstrap
F, R = E.F, E.R


def gradients():
    F.guard(); c = E.verify(); dest = E.OUT/'positive_gradient_audit.json'
    if dest.exists(): raise FileExistsError('Preserve positive gradient audit')
    bx, by = E.load(c, 'scalar_control'); all_checks = {}; weight_checks = {}
    for arm in E.ARMS[1:]:
        xs, ys = E.load(c, arm)
        for d in c['sources'] + c['targets']:
            old = np.load(E.OUT/'weights/scalar_control'/f'{d}.npz')['weights']
            new = np.load(E.OUT/'weights'/arm/f'{d}.npz')['weights']
            np.testing.assert_array_equal(new[by[d]['train_ids']], old[by[d]['train_ids']])
            added = sorted(set(ys[d]['train_ids']) - set(by[d]['train_ids']))
            for q in added:
                assert old[q] == 0 and new[q] == 10/5211
                assert not (xs[d].groups[by[d]['train_ids']] == xs[d].groups[q]).any()
            weight_checks[arm+'/'+d] = dict(added_rows=list(map(int, added)), old_rows_unchanged=True)
            if d in c['sources']:
                for k in by[d]: np.testing.assert_array_equal(ys[d][k], by[d][k])
                continue
            pairs = [tuple(p) for p in np.argwhere(by[d]['labels'] != ys[d]['labels'])]
            assert len(pairs) == (10 if d == '龙华区' else 9)
            for q, j in pairs: assert by[d]['labels'][q,j] == -1 and ys[d]['labels'][q,j] == 1
            ids = sorted({q for q,j in pairs}); local = {q:i for i,q in enumerate(ids)}
            t = R.M.batch(xs[d], ids); y = torch.from_numpy(ys[d]['labels'][ids]).to(R.DEVICE)
            w = torch.tensor(new[ids], dtype=torch.float32, device=R.DEVICE)
            start = R.M.LinearRetrieval(next(iter(xs.values())))
            for seed in c['seeds']:
                saved = torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt', map_location=R.DEVICE, weights_only=False)
                model = R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
                model.load_state_dict(saved['model']); model.eval()
                with torch.no_grad(): z = model(t)
                z = z.detach().requires_grad_(); loss = R.masked_bce(z,y,w) + R.masked_lsep(z,y,w)
                gradient = torch.autograd.grad(loss,z)[0]
                assert (gradient[y<0] == 0).all()
                new_gradients = [float(gradient[local[q],j]) for q,j in pairs]
                assert all(v < 0 for v in new_gradients)
                assert (gradient[y==0] > 0).all()
                all_checks[f'{arm}/{d}/{seed}'] = dict(new_positive_cells=len(pairs),
                    upward_gradients=len(new_gradients), unknown_nonzero_gradients=int((gradient[y<0] != 0).sum()),
                    most_negative=min(new_gradients), least_negative=max(new_gradients))
    assert sum(v['new_positive_cells'] for v in all_checks.values()) == 19*2*10
    result = dict(all_new_labels_push_up=True, per_seed_target_gradient=all_checks,
        weight_checks=weight_checks, target_gold_read=False, target_polygon_read=False,
        audit_code_sha256=F.sha(__file__))
    F.write(dest,result)
    F.write(E.OUT/'input_gradient_audit.json',result)
    print('TEXTUAL_GRADIENTS_AND_WEIGHTS_VERIFIED',len(all_checks),flush=True)


def replay():
    F.guard();c=E.verify();R.check(E.OUT,'probability_freeze.json');dest=E.OUT/'probability_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve replay')
    history=F.read(E.OUT/'training_history.json');checks={}
    for arm in c['arms']:
        xs,ys=E.load(c,arm);start=R.M.LinearRetrieval(next(iter(xs.values())))
        for seed in c['seeds']:
            key=f'{arm}/{seed}';ck=E.OUT/'checkpoints'/f'{arm}_{seed}.pt';path=E.OUT/'probabilities'/arm/f'{seed}.npz'
            assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(path)==history[key]['probabilities_sha256']
            saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            assert saved['step']==40 and len(saved['history'])==40 and all(np.isfinite(list(v.values())).all() for v in saved['history'])
            if arm=='scalar_control':assert F.sha(ck)==F.sha(E.PREV.OUT/'checkpoints'/f'features_labels_{seed}.pt')
            else:assert saved['protocol_hash']==F.sha(E.OUT/'protocol.json') and saved['history']==history[key]['losses']
            parent=torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=R.DEVICE,weights_only=False)
            assert torch.equal(saved['model']['start'],parent['model']['start'])
            model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE);model.load_state_dict(saved['model']);model.eval()
            actual=dict(np.load(path));n=0
            assert set(actual)=={d+s for d in c['sources'] for s in ['','_train_multi','_stress_0','_stress_1']}
            for d in c['sources']:
                for suffix in ['','_train_multi','_stress_0','_stress_1']:
                    ids=ys[d]['valid_ids'];x=xs[d]
                    if suffix=='_train_multi':
                        ids=ys[d]['train_ids'];y=ys[d]['labels'][ids];ids=ids[(y>=0).all(1)&((y==1).sum(1)>1)]
                    elif suffix:x=E.S.view(x,d,suffix[1:])
                    p=R.probabilities(model,x,ids,c['chunk']) if len(ids) else np.zeros((0,x.school_n),np.float32)
                    np.testing.assert_array_equal(actual[d+suffix],p);n+=len(p)
            assert n==3832;checks[key]=dict(rows=n,steps=40);print('REPLAYED',key,flush=True)
    F.write(dest,dict(models=checks,all_source_probabilities_bitwise_reproduced=True,target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))


def score():
    F.guard();c=E.verify();R.check(E.OUT,'probability_freeze.json');dest=R.ROOT/'round33_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    assert F.read(E.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    assert F.read(E.OUT/'input_gradient_audit.json')['per_seed_target_gradient']
    report=F.read(E.OUT/'source_report.json');_,ys=E.load(c,'scalar_control');predictions={};sp={};groups=0
    for arm in c['arms']:
        loaded=[dict(np.load(E.OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in ['relative','absolute']:
            key=arm+'_'+mode;v=report['reports'][key];t=v['selections']['balanced_exact']['threshold'];options=[]
            for candidate in F.read(E.OUT/'calibration'/f'{key}.json')['candidates']:
                ns=nm=0
                for d in c['sources']:
                    ct=counts(select(transform(mean[d],mode),candidate['threshold']),ys[d]['valid_labels'].astype(bool));ns+=ct['single_correct'];nm+=ct['multi_correct']
                assert (ns,nm)==(candidate['single_correct'],candidate['multi_correct']);options.append((candidate['threshold'],ns,nm))
            for obj in ['balanced_exact','micro_exact']:
                best=max(options,key=lambda v:((v[1]*60+v[2]*1186) if obj=='balanced_exact' else v[1]+v[2],v[0]))
                s=v['selections'][obj];assert best==(s['threshold'],s['single_correct'],s['multi_correct'])
            predictions[key]={};sp[key]={}
            for view in ['full','stress_0','stress_1']:
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};rank=0
                for d in c['sources']:
                    y=ys[d]['valid_labels'].astype(bool);p=transform(mean[d+('' if view=='full' else '_'+view)],mode);pr=select(p,t)
                    for k,n in counts(pr,y).items():total[k]+=n
                    if view=='full':predictions[key][d]=pr
                    for scores,truth in zip(p,y):
                        if truth.sum()>1:rank+=int(scores[truth].min()>(scores[~truth].max() if (~truth).any() else -1))
                assert total==(v['full'] if view=='full' else v['stress'][view]);groups+=1
                if view=='full':
                    g=c['source_gate'];passed=total['single_correct']>=g['min_single_correct'] and total['multi_correct']>=g['min_multi_correct'] and rank>=g['min_attainable_multi']
                    assert passed==v['source_gate_passed'] and rank==v['attainable_multi']
                    assert v['balanced_exact']==total['single_correct']/1186/2+total['multi_correct']/60/2
            for seed,payload in zip(map(str,c['seeds']),loaded):
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};sp[key][seed]={}
                for d in c['sources']:
                    pr=select(transform(payload[d],mode),t);sp[key][seed][d]=pr
                    for k,n in counts(pr,ys[d]['valid_labels'].astype(bool)).items():total[k]+=n
                assert total==v['seeds'][seed];groups+=1
            print('SCORED',key,flush=True)
    rows=[]
    for d in c['sources']:
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];y=ys[d]['valid_labels'].astype(bool)
        for i,q in enumerate(ys[d]['valid_ids']):
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(y[i].sum()>1),gold=np.flatnonzero(y[i]).tolist(),
                predicted={k:np.flatnonzero(v[d][i]).tolist() for k,v in predictions.items()},correct={k:bool((v[d][i]==y[i]).all()) for k,v in predictions.items()}))
    assert rows==F.read(E.OUT/'paired_source_rows.json')
    seed_report=F.read(E.OUT/'seed_paired_source.json')
    comparisons=[('textual_labels','scalar_control'),('textual_references','scalar_control'),('textual_references','textual_labels')]
    for mode in ['relative','absolute']:
        for arm,base in comparisons:
            key=arm+'_vs_'+base;a=arm+'_'+mode;b=base+'_'+mode
            for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
                delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
                v=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),corrected=delta.count(1),harmed=delta.count(-1),
                       delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
                assert v==report['paired'][mode][key][name];groups+=1
            for seed in map(str,c['seeds']):
                sr=[]
                for d in c['sources']:
                    y=ys[d]['valid_labels'].astype(bool);ap=sp[a][seed][d];bp=sp[b][seed][d]
                    for i in range(len(y)):
                        sr.append(dict(d=d,multi=bool(y[i].sum()>1),after=bool((ap[i]==y[i]).all()),before=bool((bp[i]==y[i]).all()),
                            fp_before=int((bp[i]&~y[i]).sum()),fp_after=int((ap[i]&~y[i]).sum()),fn_before=int((~bp[i]&y[i]).sum()),fn_after=int((~ap[i]&y[i]).sum())))
                for name,rs in [('all',sr),('single',[r for r in sr if not r['multi']]),('multi',[r for r in sr if r['multi']])]+[(d,[r for r in sr if r['d']==d]) for d in c['sources']]:
                    v=dict(n=len(rs),before_correct=sum(r['before'] for r in rs),after_correct=sum(r['after'] for r in rs),
                        corrected=sum(r['after'] and not r['before'] for r in rs),harmed=sum(r['before'] and not r['after'] for r in rs),
                        **{k:sum(r[k] for r in rs) for k in ['fp_before','fp_after','fn_before','fn_after']})
                    assert v==seed_report[mode][key][seed][name];groups+=1
    a=report['reports']['textual_references_relative'];checks={}
    for base in ['scalar_control','textual_labels']:
        b=report['reports'][base+'_relative'];checks[base]=dict(balanced_noninferior=a['balanced_exact']>=b['balanced_exact'],
            stress_noninferior={s:a['stress'][s]['correct']>=b['stress'][s]['correct'] for s in ['stress_0','stress_1']})
    passed=a['source_gate_passed'] and all(v['balanced_noninferior'] and all(v['stress_noninferior'].values()) for v in checks.values())
    assert checks==report['noninferiority_checks'] and report['selected_target_candidate']==('textual_references_relative' if passed else None)
    F.write(dest,dict(metric_groups=groups,source_rows=len(rows),all_metrics_recomputed=True,source_gate_passed=passed,
        target_predictions_generated=False,adopted=False,audit_code_sha256=F.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['gradients','replay','score']);globals()[p.parse_args().phase]()
