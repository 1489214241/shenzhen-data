"""Source-only exact-set reports, fixed controls and per-seed paired effects."""
import numpy as np
from experiment_shenzhen_relative_decode import relative,select
from shenzhen_exact_calibration import exact_thresholds,calibration_intervals,threshold_metrics
from zoning.evaluate import bootstrap


def report_trial(F,R,out,c,ys,candidate,controls,old_control):
    if (out/'source_report.json').exists():raise FileExistsError('Preserve source report')
    labels=[ys[d]['valid_labels'] for d in c['sources']];reports={};predictions={};seed_predictions={}
    for arm in c['arms']:
        loaded=[dict(np.load(out/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in loaded]).mean(0) for k in loaded[0]}
        for mode in ['relative','absolute']:
            key=arm+'_'+mode;fn=relative if mode=='relative' else lambda p:p
            ps=[fn(mean[d]) for d in c['sources']];cal=exact_thresholds(ps,labels);v=cal['selections']['balanced_exact'];t=v['threshold'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            passed=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,full=threshold_metrics(ps,labels,t),
                stress={s:threshold_metrics([fn(mean[d+'_'+s]) for d in c['sources']],labels,t) for s in ['stress_0','stress_1']},
                seeds={str(s):threshold_metrics([fn(p[d]) for d in c['sources']],labels,t) for s,p in zip(c['seeds'],loaded)})
            predictions[key]=[select(np.asarray(p,np.float64),t) for p in ps]
            seed_predictions[key]={str(s):[select(np.asarray(fn(p[d]),np.float64),t) for d in c['sources']] for s,p in zip(c['seeds'],loaded)}
            F.write(out/'calibration'/f'{key}.json',cal)
            if arm=='scalar_control':assert cal['selections']==F.read(old_control/'source_report.json')['reports'][key]['selections']
            print('SOURCE',key,reports[key]['full'],reports[key]['stress'],flush=True)
    rows=[]
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            y=labels[di][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(y.sum()>1),gold=np.flatnonzero(y).tolist(),
                predicted={k:np.flatnonzero(v[di][i]).tolist() for k,v in predictions.items()},correct={k:bool((v[di][i]==y).all()) for k,v in predictions.items()}))
    assert len(rows)==1246 and sum(r['multi'] for r in rows)==60
    paired={};seed_pairs={}
    comparisons=[(a,'scalar_control') for a in c['arms'] if a!='scalar_control']+[(candidate,a) for a in controls if a!='scalar_control']
    for mode in ['relative','absolute']:
        paired[mode]={};seed_pairs[mode]={}
        for arm,base in comparisons:
            key=arm+'_vs_'+base;a=arm+'_'+mode;b=base+'_'+mode;paired[mode][key]={};seed_pairs[mode][key]={}
            for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
                delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
                paired[mode][key][name]=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),
                    corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            for seed in map(str,c['seeds']):
                sr=[]
                for di,d in enumerate(c['sources']):
                    y=labels[di].astype(bool);ap=seed_predictions[a][seed][di];bp=seed_predictions[b][seed][di]
                    for i in range(len(y)):
                        sr.append(dict(d=d,multi=bool(y[i].sum()>1),after=bool((ap[i]==y[i]).all()),before=bool((bp[i]==y[i]).all()),
                            fp_before=int((bp[i]&~y[i]).sum()),fp_after=int((ap[i]&~y[i]).sum()),fn_before=int((~bp[i]&y[i]).sum()),fn_after=int((~ap[i]&y[i]).sum())))
                groups={}
                for name,rs in [('all',sr),('single',[r for r in sr if not r['multi']]),('multi',[r for r in sr if r['multi']])]+[(d,[r for r in sr if r['d']==d]) for d in c['sources']]:
                    groups[name]=dict(n=len(rs),before_correct=sum(r['before'] for r in rs),after_correct=sum(r['after'] for r in rs),
                        corrected=sum(r['after'] and not r['before'] for r in rs),harmed=sum(r['before'] and not r['after'] for r in rs),
                        **{k:sum(r[k] for r in rs) for k in ['fp_before','fp_after','fn_before','fn_after']})
                seed_pairs[mode][key][seed]=groups
    a=reports[candidate+'_relative'];checks={}
    for base in controls:
        b=reports[base+'_relative'];checks[base]=dict(balanced_noninferior=a['balanced_exact']>=b['balanced_exact'],
            stress_noninferior={s:a['stress'][s]['correct']>=b['stress'][s]['correct'] for s in ['stress_0','stress_1']})
    passed=a['source_gate_passed'] and all(v['balanced_noninferior'] and all(v['stress_noninferior'].values()) for v in checks.values())
    F.write(out/'paired_source_rows.json',rows);F.write(out/'seed_paired_source.json',seed_pairs)
    F.write(out/'source_report.json',dict(reports=reports,paired=paired,noninferiority_checks=checks,
        selected_target_candidate=candidate+'_relative' if passed else None,target_predictions_generated=False,independent_validation=False,adopted=False))
    print('TARGET_GATE',passed,checks,flush=True)
