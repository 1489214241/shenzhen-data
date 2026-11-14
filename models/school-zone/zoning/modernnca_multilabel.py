"""Experimental multi-label readout of the selected ModernNCA-LN adapter.

Retains its relation tokens, encoder, query, distances and school aggregation.
Exposes school log masses before categorical normalization, with one binary
intercept shared across schools. No RetrievalScorer parameters are inherited.
"""
import torch
from torch import nn
from .modernnca_normalization import LayerNormModernNCA
from .full_evidence import tokens


class MultiLabelModernNCA(LayerNormModernNCA):
    def __init__(self, np_, na, seed, dim=32):
        super().__init__(np_, na, seed, dim)
        self.binary_bias = nn.Parameter(torch.tensor(0.))

    def forward(self, t):
        x, valid = tokens(t, self.np)
        b,n,_ = x.shape; r=t['px'].shape[1]; s=t['ax'].shape[1]
        ev=self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e=ev.new_zeros(b,n,self.dim);e[valid]=ev
        mass=torch.cat([t['gb'].exp(),t['gb'].new_ones(s)])[None]*valid
        mem=t['mem'];sums=(e*mass[...,None]).transpose(1,2)@mem
        count=mass@mem;proto=sums.transpose(1,2)/count[...,None].clamp_min(1e-12)
        query=self.query(torch.cat([proto.mean(1),e[:,r:].mean(1)],-1))
        scores=-torch.linalg.vector_norm(query[:,None]-e,dim=-1)*self.log_temperature.exp().clamp(max=100.)
        scores=(scores+mass.clamp_min(1e-30).log()).masked_fill(~valid,-torch.inf)
        offset=scores.max(1,keepdim=True).values
        school_mass=torch.exp(scores-offset)@mem
        return school_mass.clamp_min(1e-30).log()+offset+self.binary_bias


def categorical_positive_loss(logits,labels,weight):
    """Identical native NLL for singleton labels; mean NLL over observed positives.

    This is equivalent to repeated single-label questions with split weight.
    It exposes the same positive relations to the categorical control; unknowns
    are not explicitly converted into negative labels, although softmax still
    creates competition. This remains a single-output control, not native-task
    historical training reproduction.
    """
    pos=(labels==1).to(logits.dtype)
    if not (pos.sum(1)>0).all():raise ValueError('Positive supervision required')
    return (-(logits.log_softmax(1)*pos).sum(1)/pos.sum(1)*weight).sum()


def observed_bce_loss(logits,labels,weight):
    mask=labels>=0
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels.clamp_min(0),reduction='none')
    return (loss.masked_fill(~mask,0).sum(1)*weight).sum()
