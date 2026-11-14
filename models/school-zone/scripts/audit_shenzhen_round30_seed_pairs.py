"""Retain corrected/harmed and FP/FN for each seed and source district."""
import numpy as np
import experiment_shenzhen_hierarchy_loss as E
from audit_shenzhen_round16 import select, counts
from audit_shenzhen_round18 import transform


def main():
    E.F.guard();c=E.verify();E.R.check(E.OUT,'probability_freeze.json')
    dest=E.OUT/'seed_paired_audit.json'
    if dest.exists(): raise FileExistsError('Preserve audit')
    report=E.F.read(E.OUT/'source_report.json');_,ys=E.X.load(c,'all_multi_training');result={}
    for mode in ['relative','absolute']:
        result[mode]={}
        for seed in c['seeds']:
            payload={a:dict(np.load(E.OUT/'probabilities'/a/f'{seed}.npz')) for a in c['arms']}
            rows=[];by_d={}
            for d in c['sources']:
                y=ys[d]['valid_labels'].astype(bool);pred={}
                for arm in c['arms']:
                    t=report['reports'][arm+'_'+mode]['selections']['balanced_exact']['threshold']
                    pred[arm]=select(transform(payload[arm][d],mode),t)
                a=pred['scalar_control'];b=pred['hierarchy_loss'];ca=(a==y).all(1);cb=(b==y).all(1)
                for i in range(len(y)):
                    rows.append(dict(d=d,multi=bool(y[i].sum()>1),before=bool(ca[i]),after=bool(cb[i]),
                        fp_before=int((a[i]&~y[i]).sum()),fp_after=int((b[i]&~y[i]).sum()),
                        fn_before=int((~a[i]&y[i]).sum()),fn_after=int((~b[i]&y[i]).sum())))
            for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
                by_d[name]=dict(n=len(rs),before_correct=sum(r['before'] for r in rs),after_correct=sum(r['after'] for r in rs),
                    corrected=sum(r['after'] and not r['before'] for r in rs),harmed=sum(r['before'] and not r['after'] for r in rs),
                    **{k:sum(r[k] for r in rs) for k in ['fp_before','fp_after','fn_before','fn_after']})
            for arm,field in [('scalar_control','before_correct'),('hierarchy_loss','after_correct')]:
                assert by_d['all'][field]==report['reports'][arm+'_'+mode]['seeds'][str(seed)]['correct']
            result[mode][str(seed)]=by_d
    E.F.write(dest,dict(results=result,ensemble_threshold_per_arm_unchanged=True,
                       audit_code_sha256=E.F.sha(__file__),target_gold_read=False))
    print('SEED_PAIRS',[(s,x['all']['corrected'],x['all']['harmed']) for s,x in result['relative'].items()],flush=True)


if __name__=='__main__': main()
