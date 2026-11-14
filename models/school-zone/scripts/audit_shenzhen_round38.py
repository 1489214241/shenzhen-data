"""Round38 independent calibration, paired harm, seed and source gate audit."""
import numpy as np
import experiment_shenzhen_local_reference_dropout as E
from audit_shenzhen_round16 import select,counts
from audit_shenzhen_round18 import transform
from zoning.evaluate import bootstrap
F,R=E.F,E.R

def score():
    F.guard();c=E.verify();R.check(E.OUT,'probability_freeze.json');dest=R.ROOT/'round38_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    assert F.read(E.OUT/'probability_replay_audit.json')['all_source_probabilities_bitwise_reproduced']
    assert F.read(E.OUT/'input_audit.json')['all_contexts_independently_recomputed']
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
    comparisons=[('local_dropout','scalar_control')]
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
    a=report['reports']['local_dropout_relative'];checks={}
    for base in ['scalar_control']:
        b=report['reports'][base+'_relative'];checks[base]=dict(balanced_noninferior=a['balanced_exact']>=b['balanced_exact'],
            stress_noninferior={s:a['stress'][s]['correct']>=b['stress'][s]['correct'] for s in ['stress_0','stress_1']})
    passed=a['source_gate_passed'] and all(v['balanced_noninferior'] and all(v['stress_noninferior'].values()) for v in checks.values())
    assert checks==report['noninferiority_checks'] and report['selected_target_candidate']==('local_dropout_relative' if passed else None)
    F.write(dest,dict(metric_groups=groups,source_rows=len(rows),all_metrics_recomputed=True,source_gate_passed=passed,
        target_predictions_generated=False,adopted=False,audit_code_sha256=F.sha(__file__)))


if __name__=='__main__': score()
