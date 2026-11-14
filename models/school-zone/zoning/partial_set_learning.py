"""Marginal set supervision: known positives required, unknown labels stay free."""
import torch


def compatible_sets(labels,membership):
    if not ((labels==-1)|(labels==0)|(labels==1)).all():raise ValueError('Invalid label state')
    # A candidate violates evidence only if it omits an observed positive or
    # includes an observed negative. Unknown cells impose no condition.
    pos=labels==1;neg=labels==0;members=membership.bool()
    missing=(pos[:,None,:]&~members[None,:,:]).any(-1)
    extra=(neg[:,None,:]&members[None,:,:]).any(-1)
    allowed=~(missing|extra)
    if not allowed.any(1).all():raise ValueError('Observed facts have no compatible set hypothesis')
    return allowed


def marginal_set_loss(logits,allowed,weights):
    if logits.shape!=allowed.shape or not allowed.any(1).all():raise ValueError('Invalid compatibility mask')
    loss=logits.logsumexp(1)-logits.masked_fill(~allowed,-torch.inf).logsumexp(1)
    return (loss*weights).sum()
