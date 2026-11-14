"""Post-score source-only diagnosis; never a new selector or target input."""
import numpy as np
import experiment_shenzhen_set_evidence as D
from audit_shenzhen_round18 import transform
from audit_shenzhen_round16 import select

F=D.F;R=D.R


def main():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    dest=D.OUT/'diagnostics_not_training/ranking_and_threshold.json'
    if dest.exists():raise FileExistsError('Preserve diagnosis')
    assert F.read(R.ROOT/'round24_independent_completion_audit.json')['all_counts_and_transition_scores_recomputed']
    _,ys=D.X.load(c,'all_multi_training');report=F.read(D.OUT/'source_report.json')['reports']
    loaded={a:[dict(np.load(D.OUT/'probabilities'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    mean={a:{k:np.stack([p[k] for p in loaded[a]]).mean(0) for k in loaded[a][0]} for a in c['arms']}
    views={}
    for view in ['full','stress_0','stress_1']:
        rows=[];suffix='' if view=='full' else '_'+view
        for d in c['sources']:
            truth=ys[d]['valid_labels'].astype(bool)
            ps={a:transform(mean[a][d+suffix],'relative') for a in c['arms']}
            pred={a:select(p,report[a+'_relative']['selections']['balanced_exact']['threshold']) for a,p in ps.items()}
            for i,q in enumerate(ys[d]['valid_ids']):
                y=truth[i];single=y.sum()==1
                info={}
                for a in c['arms']:
                    p=ps[a][i]
                    info[a]=dict(exact=bool((pred[a][i]==y).all()),top1_in_gold=bool(y[p.argmax()]),
                        rank_separable=bool(p[y].min()>(p[~y].max() if (~y).any() else -1)),
                        fp=int((pred[a][i]&~y).sum()),fn=int((~pred[a][i]&y).sum()),predicted_count=int(pred[a][i].sum()))
                rows.append(dict(district=d,source_row_id=int(q),single=bool(single),models=info))
        b='scalar_control';a='vector_residual';harmed=[r for r in rows if r['models'][b]['exact'] and not r['models'][a]['exact']]
        single_harm=[r for r in harmed if r['single']]
        views[view]=dict(rows=rows,harmed=len(harmed),single_harmed=len(single_harm),
            harmed_single_top1_still_correct=sum(r['models'][a]['top1_in_gold'] for r in single_harm),
            harmed_single_top1_now_wrong=sum(not r['models'][a]['top1_in_gold'] for r in single_harm),
            counts={arm:dict(single_top1_correct=sum(r['single'] and r['models'][arm]['top1_in_gold'] for r in rows),
                multi_rank_separable=sum(not r['single'] and r['models'][arm]['rank_separable'] for r in rows)) for arm in c['arms']})
    frontier={}
    for a in c['arms']:
        for mode in ['relative','absolute']:
            key=a+'_'+mode;cal=F.read(D.OUT/'calibration'/f'{key}.json')
            eligible=[r for r in cal['candidates'] if r['single_correct']>=c['source_gate']['min_single_correct']]
            frontier[key]=dict(max_multi_with_original_single_floor=max([r['multi_correct'] for r in eligible],default=None),
                deployable_selection=False,used_to_change_this_round=False)
    # Algebraic illustration only, not fitted model results or label-based tuning.
    logit=np.array([2.,0.]);shifted=logit+5
    sigmoid=lambda z:1/(1+np.exp(-z))
    sig_ratio=lambda z:sigmoid(z)/sigmoid(z).max()
    odds_ratio=lambda z:np.exp(z-z.max())
    example=dict(logits=logit.tolist(),shifted_logits=shifted.tolist(),
        sigmoid_ratios_before=sig_ratio(logit).tolist(),sigmoid_ratios_after=sig_ratio(shifted).tolist(),
        logit_mass_ratios_before=odds_ratio(logit).tolist(),logit_mass_ratios_after=odds_ratio(shifted).tolist(),
        meaning='Shared shifts leave ranking and native retrieval mass ratios unchanged, but alter sigmoid-ratio decisions. This alone does not prove such shifts caused real model errors.')
    F.write(dest,dict(views=views,threshold_frontier=frontier,analytic_offset_example=example,
        target_gold_read=False,target_predictions_generated=False,post_score_source_diagnosis=True,
        not_for_training=True,original_source_failure_unchanged=True,adopted=False,code_sha256=F.sha(__file__)))
    print('DIAGNOSIS',{v:{k:x for k,x in data.items() if k!='rows'} for v,data in views.items()},frontier,flush=True)


if __name__=='__main__':main()
