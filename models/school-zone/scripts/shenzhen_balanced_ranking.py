"""Source-only complete-set balancing and masked LSEP ranking loss."""
import numpy as np
import torch
from torch.nn import functional as TF


def balanced_weights(labels,weights,sources):
    """Equalize single/multi complete-source row mass; preserve all other mass.

Input weights already average rows within each property. Balancing operates on
that weighted population, not raw counts. Partial target references stay partial.
"""
    groups={};mass=np.zeros(2)
    for d in sources:
        y=labels[d];full=(y>=0).all(1);multi=(y==1).sum(1)>1
        groups[d]=(full,multi)
        for k in [0,1]:mass[k]+=weights[d][full&(multi==k)].sum()
    if (mass<=0).any():raise ValueError('Both complete source strata required')
    factors=mass.sum()/(2*mass);out={d:w.copy() for d,w in weights.items()}
    for d,(full,multi) in groups.items():
        for k in [0,1]:out[d][full&(multi==k)]*=factors[k]
    return out,dict(before_mass=mass.tolist(),factors=factors.tolist(),after_mass=[float(mass.sum()/2)]*2,
        target_labels_used=False,partial_source_weights_unchanged=True)


def masked_lsep(logits,labels,weights):
    """Eq7 LSEP, all observed positive-negative pairs; unknown labels excluded.

Exact factorization: log(1 + sum(exp(negative))*sum(exp(-positive))).
No pair sampling is needed at our vocabulary size. No observed negative means
no ranking supervision, which includes every Shenzhen partial-label row.
"""
    pos=labels==1;neg=labels==0;eligible=pos.any(1)&neg.any(1)
    if not eligible.any():return logits.sum()*0.
    s=logits[eligible];p=pos[eligible];n=neg[eligible]
    log_pairs=torch.logsumexp(s.masked_fill(~n,-torch.inf),1)+torch.logsumexp((-s).masked_fill(~p,-torch.inf),1)
    return (TF.softplus(log_pairs)*weights[eligible]).sum()
