"""Propagate reviewed scope-state changes through every dependent feature block."""
from copy import deepcopy
import numpy as np
from zoning.evidence import policy


def dependent_blocks(x,statuses,negative):
    q=statuses[:,x.ref_schools];r=statuses[x.ref_sites,x.ref_schools]
    return dict(policy=policy(q,negative[:,x.ref_schools]),
                agreement=((q>=0)&(r[None]>=0)&(q==r[None])).astype(np.float32),
                disagreement=((q>=0)&(r[None]>=0)&(q!=r[None])).astype(np.float32)),policy(statuses,negative)


def patch_features(x,before,after,negative):
    if before.shape!=after.shape:raise ValueError('Scope shape changed')
    old,old_anchor=dependent_blocks(x,before,negative)
    new,new_anchor=dependent_blocks(x,after,negative)
    for name,v in old.items():np.testing.assert_array_equal(x.pair[:,:,x.pair_blocks[name]],v)
    np.testing.assert_array_equal(x.anchor[:,:,x.anchor_blocks['policy']],old_anchor)
    out=deepcopy(x)
    for name,v in new.items():out.pair[:,:,out.pair_blocks[name]]=v
    out.anchor[:,:,out.anchor_blocks['policy']]=new_anchor
    return out
