"""Expose already available reference provenance to both comparator families.

No reference or Gold labels change. Extra source columns start at zero effect;
all-zero source columns reproduce the original function and regularization.
"""
import torch
from torch import nn
from .modernnca_normalization import LayerNormModernNCA
from .full_evidence import tokens
from .model import RetrievalScorer,_school_mask

SOURCE_TYPES=('verified','boundary','community')  # named already has a shared input flag

def source_matrix(rule,x):
    kinds=[rule.kind[int(q)] for q in x.ref_sites]
    assert set(kinds)<=set(SOURCE_TYPES)|{'named'}
    return torch.tensor([[float(k==c) for c in SOURCE_TYPES] for k in kinds],dtype=torch.float32).reshape(-1,3)

def zero_projection(width):
    with torch.random.fork_rng(devices=[]):
        layer=nn.Linear(3,width,bias=False)
        nn.init.zeros_(layer.weight)
    return layer

class ProvenanceModernNCA(LayerNormModernNCA):
    def __init__(self,np_,na,seed):
        super().__init__(np_,na,seed)
        self.source_projection=zero_projection(self.dim)

    def forward(self,t):
        x,valid=tokens(t,self.np);b,n,_=x.shape;r=t['px'].shape[1];s=t['ax'].shape[1]
        source=torch.cat([t['source_type'],t['source_type'].new_zeros(s,3)],0)[None].expand(b,-1,-1)
        ev=self.encoder.post_encoder(self.encoder.encoder(x[valid])+self.source_projection(source[valid]))
        e=ev.new_zeros(b,n,self.dim);e[valid]=ev
        mass=torch.cat([t['gb'].exp(),t['gb'].new_ones(s)])[None]*valid;mem=t['mem']
        sums=(e*mass[...,None]).transpose(1,2)@mem;count=mass@mem
        proto=sums.transpose(1,2)/count[...,None].clamp_min(1e-12)
        query=self.query(torch.cat([proto.mean(1),e[:,r:].mean(1)],-1))
        scale=self.log_temperature.exp().clamp(max=100.)
        logits=-torch.linalg.vector_norm(query[:,None]-e,dim=-1)*scale+mass.clamp_min(1e-30).log()
        return logits.masked_fill(~valid,-torch.inf).softmax(-1)@mem

class ProvenanceRetrieval(RetrievalScorer):
    def __init__(self,np_,na,start,seed,width=32):
        super().__init__(np_,na,start,seed,width)
        self.source_projection=zero_projection(width)
        self.source_linear=nn.Parameter(torch.zeros(3))

    def forward(self,t):
        hidden=self.pnet[0](t['px'])+self.source_projection(t['source_type'])[None]
        for layer in list(self.pnet.children())[1:]:hidden=layer(hidden)
        pl=t['pair']@self.p+t['gb'][None]+t['px'][:,:,self.np:]@self.pc+hidden.squeeze(-1)
        pl=pl+(t['source_type']@self.source_linear)[None]
        al=t['anchor']@self.a+self.offset+t['ax'][:,:,self.na:]@self.ac+self.anet(t['ax']).squeeze(-1)
        pl=pl.masked_fill(~t['valid'],-torch.inf);al=_school_mask(self,al,t['valid'])
        return torch.softmax(torch.cat([pl,al],1),1)@t['mem']

    def regularization(self,reg):
        return super().regularization(reg)+reg*self.source_projection.weight.square().mean()+.01*self.source_linear.square().sum()
