"""School-wide missing reference views, preserving labels and independent query policy."""
from copy import deepcopy
import numpy as np
import torch


def plans(school_n,seed):
    rng=np.random.default_rng(seed);out={}
    for partition in range(3):
        order=rng.permutation(school_n);half=school_n//2
        hidden=np.zeros(school_n,bool);hidden[order[:half]]=True
        prefix='train' if partition<2 else 'stress'
        index=partition*2 if partition<2 else 0
        out[f'{prefix}_{index}']=hidden;out[f'{prefix}_{index+1}']=~hidden
    return out


def boundary_support(known,hidden):
    # Removing one member of a partial multi set must never create a false singleton.
    return {q:list(ss) for q,ss in known.items() if not any(hidden[s] for s in ss)}


def apply_view(x,columns,boundary):
    columns=np.asarray(columns,int);out=deepcopy(x)
    assert columns.ndim==1 and len(set(columns))==len(columns)
    out.pair=x.pair[:,columns].copy()
    assert boundary.shape==out.pair[:,:,x.pair_blocks['learned_boundary']].shape
    out.pair[:,:,x.pair_blocks['learned_boundary']]=boundary
    out.ref_sites=x.ref_sites[columns].copy();out.ref_schools=x.ref_schools[columns].copy()
    _,inv,count=np.unique(out.groups[out.ref_sites],return_inverse=True,return_counts=True)
    out.group_bias=-np.log(count[inv]).astype(np.float32)
    out.self_mask=out.groups[:,None]==out.groups[out.ref_sites][None]
    out.membership=torch.nn.functional.one_hot(torch.tensor(np.r_[out.ref_schools,np.arange(out.school_n)],dtype=torch.long),out.school_n).float()
    assert np.isfinite(out.pair).all()
    return out
