"""Drop near-property retrieval edges, retaining independently supplied facts.

This regularizes the direct retrieval channel; it is NOT a simulation in which
all knowledge about those properties disappears. Existing policy/road-boundary
features remain available and unchanged. Only validity and its six context
statistics are rebuilt. No labels enter this transformation.
"""
import numpy as np
import torch
from zoning import model as M


def local_validity(valid, distances, reference_groups, radius=500.):
    valid=np.asarray(valid,bool);distances=np.asarray(distances,float)
    if distances.shape!=valid.shape or valid.shape[1]!=len(reference_groups):
        raise ValueError('Mismatched reference arrays')
    if radius<0 or not np.isfinite(distances).all() or (distances<0).any():
        raise ValueError('Invalid distance or radius')
    _,inv=np.unique(reference_groups,return_inverse=True)
    hit=np.zeros((len(valid),int(inv.max())+1 if len(inv) else 0),bool)
    np.logical_or.at(hit,(np.arange(len(valid))[:,None],inv[None,:]),valid&(distances<=radius))
    return valid & ~hit[:,inv]


def masked_batch(t, reference_groups, radius=500.):
    """M.batch computes context on CPU; preserve that numerical convention."""
    pair=t['pair'].detach().cpu();anchor=t['anchor'].detach().cpu()
    valid=local_validity(t['valid'].detach().cpu().numpy(),pair[:,:,t['meta'][0]].numpy()*300,reference_groups,radius)
    vt=torch.from_numpy(valid)
    px,ax=M.context(pair,anchor,vt,t['gb'].detach().cpu(),t['mem'][:pair.shape[1]].detach().cpu(),
        t['rs'].detach().cpu().numpy(),*t['meta'])
    out=dict(t);device=t['pair'].device
    out.update(valid=vt.to(device),px=px.to(device),ax=ax.to(device))
    return out
