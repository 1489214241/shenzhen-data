"""School-independent feature probe using only the frozen retrieval evidence.

Pool positive-reference features separately for each school and distance scale.
No school IDs, location IDs, cardinality labels, polygons or Gold enter vectors.
"""
import numpy as np
import torch
from zoning import model as M

SCALES = (100.,300.,1000.)


def pool_references(pair,valid,bias,schools,school_n,distance_column):
    blocks=[]
    for scale in SCALES:
        pooled=np.full((len(pair),school_n,pair.shape[-1]+1),np.nan,np.float32)
        for school in range(school_n):
            indices=np.flatnonzero(schools==school)
            if not len(indices):
                pooled[:,school,-1]=0
                continue
            local=pair[:,indices]
            w=np.exp(bias[indices][None]-local[:,:,distance_column]*300/scale)*valid[:,indices]
            total=w.sum(1)
            pooled[:,school,:-1]=np.divide(np.einsum('qr,qrf->qf',w,local),total[:,None],
                out=np.full((len(pair),pair.shape[-1]),np.nan,np.float32),where=total[:,None]>0)
            pooled[:,school,-1]=np.log1p(total)
        blocks.append(pooled)
    return np.concatenate(blocks,axis=-1)


def school_vectors(x,ids,chunk=64):
    result=[]
    for lo in range(0,len(ids),chunk):
        q=ids[lo:lo+chunk];pair=x.pair[q];anchor=x.anchor[q];valid=~x.self_mask[q]
        _, ax=M.context(torch.from_numpy(pair),torch.from_numpy(anchor),torch.from_numpy(valid),
            torch.from_numpy(x.group_bias),x.membership[:len(x.ref_sites)],x.ref_schools,
            x.pair_blocks['distance'].start,x.crossing_cols,x.np_)
        pooled=pool_references(pair,valid,x.group_bias,x.ref_schools,x.school_n,x.pair_blocks['distance'].start)
        result.append(np.concatenate([ax.numpy(),pooled],axis=-1))
    if not result:return np.empty((0,x.school_n,x.na+6+len(SCALES)*(x.np_+1)),np.float32)
    out=np.concatenate(result);assert not np.isinf(out).any()
    return out
