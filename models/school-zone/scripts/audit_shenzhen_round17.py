"""Recompute the relative decoder without using its transformation helper."""
import numpy as np
import experiment_shenzhen_relative_multireferences as D
from audit_shenzhen_round16 import counts

F=D.F;R=D.R;X=D.X


def main():
    F.guard();c=D.verify();_,ys=X.load(c,'control');report=F.read(D.OUT/'source_report.json');claims=report['reports'];rows=F.read(D.OUT/'paired_source_rows.json')
    assert F.read(R.ROOT/'round16_independent_completion_audit.json')['all_source_probabilities_replayed']
    by={(r['d'],r['q']):r for r in rows};assert len(by)==1246
    actual={}
    for bank in c['banks']:
        loaded=[dict(np.load(X.OUT/'probabilities'/bank/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([p[d] for p in loaded]).mean(0) for d in c['sources']}
        for mode in c['decoders']:
            key=bank+'_'+mode;totals={};attainable=0
            for label,ps in [('ensemble',mean)]+[(str(s),p) for s,p in zip(c['seeds'],loaded)]:
                count={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
                for d in c['sources']:
                    p=np.array(ps[d],dtype=np.float64,copy=True)
                    if mode=='relative':
                        maxima=p.max(1);positive=maxima>0;p[positive]/=maxima[positive,None]
                    truth=ys[d]['valid_labels'].astype(bool);selected=p>=claims[key]['selections']['balanced_exact']['threshold']
                    for i in range(len(p)):
                        if not selected[i].any():selected[i,p[i].argmax()]=True
                        if label=='ensemble':
                            row=by[d,int(ys[d]['valid_ids'][i])]
                            assert row['gold']==np.flatnonzero(truth[i]).tolist()
                            assert row['predicted'][key]==np.flatnonzero(selected[i]).tolist()
                            assert row['correct'][key]==bool((selected[i]==truth[i]).all())
                            if truth[i].sum()>1:attainable+=int(p[i,truth[i]].min()>(p[i,~truth[i]].max() if (~truth[i]).any() else -1))
                    measured=counts(selected,truth)
                    for k,v in measured.items():count[k]+=v
                claimed=claims[key]['selections']['balanced_exact'] if label=='ensemble' else claims[key]['seeds'][label]
                assert all(claimed[k]==v for k,v in count.items())
                if label=='ensemble':totals=count
            assert attainable==claims[key]['attainable_multi']
            g=c['source_gate'];passed=totals['single_correct']>=g['min_single_correct'] and totals['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            assert passed==claims[key]['source_gate_passed']
            balanced=totals['single_correct']/1186/2+totals['multi_correct']/60/2;assert balanced==claims[key]['balanced_exact']
            actual[key]=dict(**totals,attainable_multi=attainable,gate_passed=bool(passed),balanced_exact=balanced)
    expected=next((k for k in ['all_multi_training_relative','complete_existing_relative'] if actual[k]['gate_passed'] and actual[k]['balanced_exact']>=actual['control_absolute']['balanced_exact']),None)
    assert expected==report['selected_target_candidate']
    for bank in c['banks']:
        for label,claim in report['paired'][bank].items():
            rr=rows if label=='all' else [r for r in rows if r['multi']] if label=='multi' else [r for r in rows if not r['multi']] if label=='single' else [r for r in rows if r['d']==label]
            delta=[int(r['correct'][bank+'_relative'])-int(r['correct'][bank+'_absolute']) for r in rr]
            assert claim['corrected']==delta.count(1) and claim['harmed']==delta.count(-1)
    dest=R.ROOT/'round17_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    F.write(dest,dict(source_checkpoint_replay_inherited_from_round16=True,all_6_source_decoders_recomputed=True,all_10_seeds_recomputed=True,
        actual=actual,selected_target_candidate=expected,source_gate_recomputed=True,corrected_harmed_recomputed=True,target_predictions_generated=False,
        independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('AUDIT_COMPLETE',actual,'NEXT_TARGET',expected,flush=True)


if __name__=='__main__':main()
