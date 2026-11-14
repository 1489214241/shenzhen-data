"""Schoolwise property deduplication for independent Bernoulli retrieval.

The native categorical model divides one property's reference mass across all
observed school relations. This experimental alternative deduplicates repeated
observations within (property, school), without dividing by other school labels.
It uses observed references only, never target labels or target cardinality.
"""
import numpy as np
import torch
from zoning import model as M


def schoolwise_bias(groups, ref_sites, ref_schools):
    groups=np.asarray(groups);ri=np.asarray(ref_sites,dtype=int);rs=np.asarray(ref_schools,dtype=int)
    if len(ri)!=len(rs):raise ValueError('Reference identity lengths differ')
    if not len(ri):return np.empty(0,np.float32)
    # np.unique handles the native numeric group IDs without string coercion.
    keys=np.column_stack([groups[ri],rs])
    _,inverse,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    return -np.log(count[inverse]).astype(np.float32)


def batch(x,ids,mode):
    t=M.batch(x,ids)
    if mode=='property_mass':return t
    if mode not in ['school_mass','school_mass_context']:raise ValueError(mode)
    t['gb']=torch.from_numpy(schoolwise_bias(x.groups,x.ref_sites,x.ref_schools)).to(t['gb'].device)
    if mode=='school_mass_context':
        px,ax=M.context(t['pair'],t['anchor'],t['valid'],t['gb'],t['mem'][:len(x.ref_sites)],
                        t['rs'],*t['meta'])
        t['px']=px;t['ax']=ax
    return t
