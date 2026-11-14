"""Matched weights for reference/partial-label input interventions."""
import numpy as np
from shenzhen_balanced_ranking import balanced_weights


def fixed_source_weights(groups, baseline_labels, current_labels, sources):
    active=[d for d,y in baseline_labels.items() if len(y['train_ids'])]
    denominator=sum(len(set(groups[d][baseline_labels[d]['train_ids']])) for d in active)
    if not denominator:raise ValueError('Empty baseline training population')
    weights={}
    for d,g in groups.items():
        ids=current_labels[d]['train_ids'];w=np.zeros(len(g))
        if len(ids):
            _,inv,count=np.unique(g[ids],return_inverse=True,return_counts=True)
            w[ids]=len(active)/denominator/count[inv]
        weights[d]=w
    out,audit=balanced_weights({d:y['labels'] for d,y in current_labels.items()},weights,sources)
    audit.update(baseline_group_denominator=denominator,baseline_active_districts=len(active),
        per_district_mass={d:float(w.sum()) for d,w in out.items()},
        treatment='New target-positive groups receive original per-property weight; source mass is fixed, not diluted.')
    return out,audit


def direct_positive_union(decision, positives):
    """Same late supplement as round07; no extra policy expansion from new facts."""
    return sorted(set(decision['predicted_school_ids'])|set(positives))
