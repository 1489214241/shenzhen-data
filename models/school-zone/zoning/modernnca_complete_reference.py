"""Experimental complete-reference branch for the existing ModernNCA-LN.

Complete reference sets are supplied explicitly. Partial policy positives never
become complete labels. School logits retain the native aggregation; the extra
branch combines whole-set evidence mass with a spatial nearest-set prior.
"""
import numpy as np
import torch
from torch import nn
from .modernnca_multilabel import MultiLabelModernNCA
from .full_evidence import tokens


def coverage_select(coords, budget, seed, scale=500.):
    """Greedy facility location using coordinates only; deterministic tie order."""
    xy=np.asarray(coords, float)
    if not 0 < budget <= len(xy): raise ValueError('Invalid budget')
    distance=np.linalg.norm(xy[:,None]-xy[None],axis=-1)
    affinity=np.exp(-distance/scale)
    order=np.random.default_rng(seed).permutation(len(xy))
    best=np.zeros(len(xy)); selected=[]; available=np.ones(len(xy),bool)
    for _ in range(budget):
        gain=np.maximum(affinity-best[:,None],0).sum(0)
        gain[~available]=-np.inf
        q=int(order[np.argmax(gain[order])]);selected.append(q);available[q]=False
        best=np.maximum(best,affinity[:,q])
    return np.array(selected,int)


def catalogue(complete_sets, school_n):
    sets={tuple(sorted(set(s))) for s in complete_sets}
    if () in sets: raise ValueError('Empty complete set')
    sets.update((s,) for s in range(school_n))
    return sorted(sets,key=lambda s:(len(s),s))


class CompleteReferenceModernNCA(MultiLabelModernNCA):
    def __init__(self,np_,na,seed,dim=32):
        super().__init__(np_,na,seed,dim)
        self.set_log_weight=nn.Parameter(torch.tensor(-2.302585093))

    def forward(self,t):
        x,valid=tokens(t,self.np);b,n,_=x.shape
        r=t['px'].shape[1];s=t['ax'].shape[1]
        ev=self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e=ev.new_zeros(b,n,self.dim);e[valid]=ev
        mass=torch.cat([t['gb'].exp(),t['gb'].new_ones(s)])[None]*valid
        count=mass@t['mem']
        proto=((e*mass[...,None]).transpose(1,2)@t['mem']).transpose(1,2)/count[...,None].clamp_min(1e-12)
        query=self.query(torch.cat([proto.mean(1),e[:,r:].mean(1)],-1))
        z=-torch.linalg.vector_norm(query[:,None]-e,dim=-1)*self.log_temperature.exp().clamp(max=100.)
        z=(z+mass.clamp_min(1e-30).log()).masked_fill(~valid,-torch.inf)
        offset=z.max(1,keepdim=True).values
        exp=torch.exp(z-offset)
        school_logits=(exp@t['mem']).clamp_min(1e-30).log()+offset+self.binary_bias
        set_mass=exp@t['set_mem']
        # Missing complete-set evidence is not invented from partial positives.
        set_log_mass=set_mass.clamp_min(1e-30).log()
        relative=set_log_mass-set_log_mass.max(1,keepdim=True).values
        scores=t['geo_prior']+self.set_log_weight.exp().clamp(max=10.)*relative
        scores=scores.masked_fill(~t['set_available'],-1e9)
        return school_logits,scores


def complete_targets(labels,candidates):
    labels=np.asarray(labels)
    if not np.isin(labels,[0,1]).all():
        raise ValueError('Partial observations cannot be complete-set targets')
    mapping={s:i for i,s in enumerate(candidates)}
    return np.array([mapping.get(tuple(np.flatnonzero(y)), -1) for y in labels],int)
