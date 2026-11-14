"""Observed-label ZLPR; complete-source intervention leaves partial losses intact.

Su et al., arXiv:2208.02955, Equation 1. Unknown cells are excluded from both
terms. This masked extension is not the paper's fully observed benchmark.
"""
import torch


def zlpr_rows(logits,labels):
    if logits.ndim!=2 or logits.shape!=labels.shape:raise ValueError('Label shape mismatch')
    if not torch.all((labels==-1)|(labels==0)|(labels==1)):raise ValueError('Invalid label state')
    zero=logits.new_zeros((len(logits),1))
    positive=torch.logsumexp(torch.cat([(-logits).masked_fill(labels!=1,-torch.inf),zero],1),1)
    negative=torch.logsumexp(torch.cat([logits.masked_fill(labels!=0,-torch.inf),zero],1),1)
    return positive+negative


def source_loss(logits,labels,weights):
    from shenzhen_partial_model import masked_bce
    from shenzhen_balanced_ranking import masked_lsep
    full=(labels>=0).all(1)
    a=(zlpr_rows(logits[full],labels[full])*weights[full]).sum()
    b=masked_bce(logits[~full],labels[~full],weights[~full])+masked_lsep(logits[~full],labels[~full],weights[~full])
    return a+b
