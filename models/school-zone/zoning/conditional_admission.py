"""School-shared supplement learner; frozen top-1 is never trained away.

ZLPR follows Su et al. (2022), eq.1 with an observed-only mask, which is our
partial-label adaptation, not a claim that the paper validated this setting.
"""
import numpy as np
import torch
from torch import nn


class Admission(nn.Module):
    def __init__(self, dim, seed):
        super().__init__(); torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias,-3.)

    def forward(self,x):return self.net(x).squeeze(-1)


def partial_loss(z,y,weight,kind):
    observed=y>=0
    if kind=='bce':
        values=torch.nn.functional.binary_cross_entropy_with_logits(z,y.clamp_min(0),reduction='none')
        row=values.masked_fill(~observed,0).sum(1)/observed.sum(1).clamp_min(1)
    elif kind=='zlpr':
        zero=torch.zeros_like(z[:,:1])
        pos=torch.cat([zero,(-z).masked_fill(y!=1,-torch.inf)],1)
        neg=torch.cat([zero,z.masked_fill(y!=0,-torch.inf)],1)
        row=pos.logsumexp(1)+neg.logsumexp(1)
    else:raise ValueError(kind)
    return (row*weight).sum()


def combine(p,admission):
    """Encode a mandatory native top-1 without changing other admissions."""
    result=np.asarray(admission,dtype=np.float64).copy()
    result[np.arange(len(p)),np.asarray(p).argmax(1)]=1.
    return result


def features(x,p,lo,hi):
    """All references enter means; exclude query property; no school IDs."""
    a=np.asarray(x.anchor[lo:hi],np.float32)
    top=np.asarray(p[lo:hi]).argmax(1)
    aa=a[np.arange(len(a)),top][:,None,:]
    mem=x.membership.detach().cpu().numpy()[:len(x.ref_sites)]
    mass=(~x.self_mask[lo:hi]).astype(np.float32)*np.exp(x.group_bias)[None,:]
    count=mass@mem
    means=np.einsum('brf,br,rs->bsf',x.pair[lo:hi],mass,mem,optimize=True)/np.maximum(count[...,None],1e-12)
    pp=np.asarray(p[lo:hi],np.float32);mx=pp.max(1,keepdims=True)
    rel=np.log(np.maximum(pp,1e-30))-np.log(np.maximum(mx,1e-30))
    entropy=-(pp*np.log(np.maximum(pp,1e-30))).sum(1)/np.log(pp.shape[1])
    extra=np.stack([np.clip(rel,-30,0)/10,pp,np.broadcast_to(mx,pp.shape),
        np.broadcast_to(entropy[:,None],pp.shape),np.log1p(count)/5],-1)
    return np.concatenate([a,np.broadcast_to(aa,a.shape),a-aa,means,extra],-1).astype(np.float32)
