"""Post-score attribution, diagnostic only; no candidate selection or tuning."""
from collections import Counter
import shenzhen_full_inputs as F


def main():
    root=F.REPO/'outputs/shenzhen_resolution_20261112/round06_balanced_ranking'
    out=root/'decoding_attribution_not_training.json'
    if out.exists():raise FileExistsError('Preserve post-score audit')
    reports={}
    for arm in ['continued_bce','balanced_bce','balanced_rank']:
        reports[arm]={}
        for obj in ['micro_exact','balanced_exact']:
            rows=F.read(root/'diagnostics_not_training'/f'{arm}_{obj}.json')
            counts=Counter();sizes=Counter()
            for r in rows:
                raw=set(r['decision']['raw_school_ids']);final=set(r['after']);gold=set(r['gold'])
                counts['n']+=1;sizes[len(raw)]+=1
                counts['raw_exact']+=raw==gold
                counts['final_exact']+=final==gold
                if len(gold)>1:
                    counts['multi_gold']+=1
                    counts['multi_raw_exact']+=raw==gold
                    counts['multi_final_exact']+=final==gold
                    counts['multi_final_correct_from_raw_single']+=len(raw)==1 and final==gold
                counts['raw_changed_by_policy_or_lower_bound']+=raw!=final
            reports[arm][obj]=dict(counts=counts,raw_cardinality_distribution=dict(sizes))
            print(arm,obj,dict(counts),dict(sizes),flush=True)
    F.write(out,dict(reports=reports,purpose='Mechanism diagnostic after frozen scoring; not an independent causal ablation.',
        adopted=False,used_for_training=False,used_for_calibration=False))


if __name__=='__main__':main()
