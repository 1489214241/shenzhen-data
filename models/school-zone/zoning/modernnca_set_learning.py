"""Learned permutation-invariant set readout of frozen ModernNCA-LN evidence.

The evidence encoder/query/matching weights are frozen in this experiment.
The new school transform and joint-set readout are trained on complete source
sets, not on thresholded predictions or partial target positives.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .full_evidence import tokens


@torch.no_grad()
def evidence_latents(model,t):
    if model.training:raise ValueError('Frozen evidence must use eval mode')
    x,valid=tokens(t,model.np);b,n,_=x.shape;r=t['px'].shape[1];s=t['ax'].shape[1]
    ev=model.encoder.post_encoder(model.encoder.encoder(x[valid]))
    e=ev.new_zeros(b,n,model.dim);e[valid]=ev
    mass=torch.cat([t['gb'].exp(),t['gb'].new_ones(s)])[None]*valid
    count=mass@t['mem']
    proto=((e*mass[...,None]).transpose(1,2)@t['mem']).transpose(1,2)/count[...,None].clamp_min(1e-12)
    query=model.query(torch.cat([proto.mean(1),e[:,r:].mean(1)],-1))
    score=-torch.linalg.vector_norm(query[:,None]-e,dim=-1)*model.log_temperature.exp().clamp(max=100.)
    score=(score+mass.clamp_min(1e-30).log()).masked_fill(~valid,-torch.inf)
    offset=score.max(1,keepdim=True).values
    logits=(torch.exp(score-offset)@t['mem']).clamp_min(1e-30).log()+offset+model.binary_bias
    p=logits.softmax(1)
    features=torch.cat([proto,e[:,r:],query[:,None].expand(-1,s,-1),
        p.clamp_min(1e-30).log()[...,None],count.log1p()[...,None],
        torch.linalg.vector_norm(query[:,None]-proto,dim=-1)[...,None]],-1)
    return features,p


class LearnedSetReadout(nn.Module):
    def __init__(self,seed,dim=32,width=32):
        super().__init__();torch.manual_seed(seed);self.dim=dim
        self.school=nn.Sequential(nn.LayerNorm(dim*3+3),nn.Linear(dim*3+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.joint=nn.Sequential(nn.Linear(width*2+4,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.joint[-1].weight);nn.init.zeros_(self.joint[-1].bias)

    def forward(self,features,membership):
        """membership is [candidate,school], or [batch,candidate,school]."""
        if membership.ndim==2:membership=membership[None].expand(len(features),-1,-1)
        size=membership.sum(-1)
        if not (size>0).all():raise ValueError('Empty school set')
        norm=membership/size[...,None]
        h=self.school(features)
        mean=norm@h;var=((norm@h.square())-mean.square()).clamp_min(0)
        lp=features[...,3*self.dim]
        avg=(norm@lp[...,None]).squeeze(-1)
        variance=((norm@lp.square()[...,None]).squeeze(-1)-avg.square()).clamp_min(0)
        p_sum=(membership@lp.exp()[...,None]).squeeze(-1)
        stats=torch.stack([avg,variance.sqrt(),size.log(),p_sum],-1)
        # Exact zero-parameter control matches the geometric mean decoder.
        base=avg+size.log()
        return base+self.joint(torch.cat([mean,var,stats],-1)).squeeze(-1)


def complete_set_targets(labels,membership):
    if not ((labels==0)|(labels==1)).all():
        raise ValueError('Partial positives must not become complete set targets')
    matches=(labels[:,None,:]==membership[None,:,:]).all(-1)
    if not (matches.sum(1)==1).all():raise ValueError('Complete target needs one candidate')
    return matches.long().argmax(1)


def training_loss(logits,target,teacher_p,singleton_columns,is_single,weights):
    ce=F.cross_entropy(logits,target,reduction='none')
    conditional=logits[:,singleton_columns].log_softmax(1)
    kl=F.kl_div(conditional,teacher_p,reduction='none').sum(1)
    return ((ce+.25*kl*is_single)*weights).sum()
