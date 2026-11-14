"""Round37 independent metrics; copied unchanged round34 equations, new output."""
import numpy as np
import experiment_shenzhen_adaptive_threshold as T
from zoning.evaluate import bootstrap
F,R,E,S=T.F,T.R,T.E,T.S

def decisions(a,threshold=0.,inclusive=False):
    p=a>=threshold if inclusive else a>threshold
    for i in range(len(p)):
        if not p[i].any():p[i,np.argmax(a[i])]=True
    return p


def metric(pred,gold):
    n=correct=single=multi=fp=fn=0
    for ps,ys in zip(pred,gold):
        for p,y in zip(ps,ys):
            a=set(np.flatnonzero(p));b=set(np.flatnonzero(y));good=a==b
            n+=1;correct+=good;single+=good and len(b)==1;multi+=good and len(b)>1
            fp+=len(a-b);fn+=len(b-a)
    return dict(n=n,correct=correct,single_correct=single,multi_correct=multi,fp=fp,fn=fn)


def score():
    F.guard();c=T.verify();R.check(T.OUT,'prediction_freeze.json')
    assert (T.OUT/'independent_replay_audit.json').exists()
    dest=R.ROOT/'round37_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve score audit')
    _,ys=E.load(c,'features_labels');gold=[ys[d]['valid_labels'] for d in c['sources']]
    report=F.read(T.OUT/'source_report.json');seed_report=F.read(T.OUT/'source_seed_metrics.json');rows=F.read(T.OUT/'paired_source_rows.json')
    old=F.read(E.OUT/'source_report.json')['reports']['features_labels_relative'];threshold=old['selections']['balanced_exact']['threshold']
    preds={};allmetrics={};groups=0;attainable=0
    for arm in ['original_relative']+T.ARMS:
        raw=[dict(np.load(E.OUT/'probabilities/features_labels'/f'{s}.npz' if arm=='original_relative' else T.OUT/'margins'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([r[k] for r in raw]).mean(0) for k in raw[0]}
        def predict(r,view):
            ans=[]
            for d in c['sources']:
                a=r[d if view=='valid' else d+'_'+view] if arm=='original_relative' else r[d+'_'+view]
                a=np.asarray(a,np.float64)
                ans.append(decisions(a/a.max(1,keepdims=True),threshold,True) if arm=='original_relative' else decisions(a))
            return ans
        preds[arm]=predict(mean,'valid');full=metric(preds[arm],gold)
        stress={v:metric(predict(mean,v),gold) for v in ['stress_0','stress_1']}
        result=dict(full=full,stress=stress,balanced_exact=full['single_correct']/1186/2+full['multi_correct']/60/2)
        assert result==report['reports'][arm];allmetrics[arm]=result;groups+=3
        for seed,r in zip(c['seeds'],raw):
            ps=predict(r,'valid');expected=dict(all=metric(ps,gold),districts={d:metric([p],[g]) for d,p,g in zip(c['sources'],ps,gold)})
            assert expected==seed_report[arm][str(seed)];groups+=9
        if arm=='adaptive_head':
            for d,y in zip(c['sources'],gold):
                for z,g in zip(mean[d+'_valid'],y):
                    g=g.astype(bool)
                    attainable+=g.sum()>1 and min(z[g])>max(z[~g])
    k=0
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            row=rows[k];k+=1
            assert row['d']==d and row['q']==q and row['gold']==np.flatnonzero(gold[di][i]).tolist()
            assert row['block']==str(tuple(int(v//1000) for v in xy[q]))
            assert row['multi']==(gold[di][i].sum()>1)
            for a,p in preds.items():
                assert row['predicted'][a]==np.flatnonzero(p[di][i]).tolist()
                assert row['correct'][a]==bool((p[di][i]==gold[di][i]).all())
    assert k==1246 and sum(r['multi'] for r in rows)==60
    for a,b in [('adaptive_head','original_relative'),('adaptive_head','constant_head'),('constant_head','original_relative')]:
        for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
            value=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            assert value==report['paired'][a+'_vs_'+b][name];groups+=1
    a=allmetrics['adaptive_head'];g=c['source_gate']
    checks=dict(single=a['full']['single_correct']>=g['min_single_correct'],multi=a['full']['multi_correct']>=g['min_multi_correct'],attainable=bool(attainable>=g['min_attainable_multi']))
    for b in ['original_relative','constant_head']:
        checks['balanced_vs_'+b]=a['balanced_exact']>=allmetrics[b]['balanced_exact']
        for v in ['stress_0','stress_1']:checks[v+'_vs_'+b]=a['stress'][v]['correct']>=allmetrics[b]['stress'][v]['correct']
    assert checks==report['checks'] and attainable==report['attainable_multi']
    assert report['selected_target_candidate']==('adaptive_head' if all(checks.values()) else None)
    F.write(dest,dict(source_rows=k,metric_groups=groups,all_metrics_independently_recomputed=True,source_gate_passed=all(checks.values()),target_predictions_generated=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('SCORE_AUDIT_PASS',groups,'GATE',all(checks.values()),flush=True)


