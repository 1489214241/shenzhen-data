"""Numerical-embedding completeness audit for the relation-input ModernNCA.

Research adapter: same lossless tokens, same query, LN, Euclidean readout and
school vote. Numerical columns are explicit feature-schema metadata. Original
PLR-lite implementation is imported without reimplementing its mathematics.
"""
import torch
from torch import nn
from .modernnca_normalization import LayerNormModernNCA
from TALENT.model.lib.tabr.utils import PLREmbeddings, LREmbeddings


def numeric_columns(x):
    pair=list(range(x.pair_blocks['distance'].start,x.pair_blocks['distance'].stop))
    b=x.pair_blocks['learned_boundary'].start
    pair += [b,b+1,b+2,b+4,b+5]  # column3 is an observed/not-observed indicator
    school=[x.np_+i for i in range(x.anchor_blocks['distance'].start,x.anchor_blocks['distance'].stop)]
    context=list(range(x.np_+x.na,x.np_+x.na+5))  # sixth is binary availability
    return sorted(set(pair+school+context+[x.np_+x.na+6]))  # log project mass


class NumericInput(nn.Module):
    def __init__(self,nin,dim,cols,mean,std,kind,frequency_scale=.1):
        super().__init__()
        assert kind in ['standard','lr','plr']
        categorical=[i for i in range(nin) if i not in cols]
        self.register_buffer('cols',torch.tensor(cols,dtype=torch.long))
        self.register_buffer('categorical',torch.tensor(categorical,dtype=torch.long))
        self.register_buffer('mean',mean.detach().float().cpu())
        self.register_buffer('std',std.detach().float().cpu())
        assert self.mean.shape==self.std.shape==(len(cols),)
        assert (self.std>0).all()
        self.kind=kind
        if kind=='plr':
            self.numeric=PLREmbeddings(n_features=len(cols),n_frequencies=16,
                frequency_scale=frequency_scale,d_embedding=16,lite=True)
        elif kind=='lr':self.numeric=LREmbeddings(n_features=len(cols),d_embedding=16)
        else:self.numeric=nn.Identity()
        nout=len(categorical)+len(cols)*(1 if kind=='standard' else 16)
        self.projection=nn.Linear(nout,dim)

    def forward(self,x):
        z=(x[:,self.cols]-self.mean)/self.std
        n=self.numeric(z).flatten(1)
        return self.projection(torch.cat([n,x[:,self.categorical]],1))


class NumericModernNCA(LayerNormModernNCA):
    def __init__(self,np_,na,seed,cols,mean,std,kind,frequency_scale=.1,dropout=.1):
        super().__init__(np_,na,seed)
        self.numeric_kind=kind
        if kind!='raw':
            # Preserve common post-encoder/query initialization and dropout RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+731)
                module=NumericInput(np_+na+8,self.dim,cols,mean,std,kind,frequency_scale)
            if kind=='standard':
                # Same original affine map at initialization (up to float error),
                # then optimize the standardized parameterization.
                original=self.encoder.encoder
                order=torch.cat([module.cols,module.categorical])
                with torch.no_grad():
                    module.projection.weight.copy_(original.weight[:,order])
                    module.projection.weight[:,:len(cols)] *= module.std[None]
                    module.projection.bias.copy_(original.bias+original.weight[:,cols]@module.mean)
            self.encoder.encoder=module
        for mod in self.encoder.post_encoder.modules():
            if isinstance(mod,nn.Dropout):mod.p=dropout

    def regularization(self,reg):
        if self.numeric_kind!='standard':return super().regularization(reg)
        # Preserve the raw affine function's penalty under standardization.
        module=self.encoder.encoder
        w=module.projection.weight
        effective=torch.cat([w[:,:len(module.cols)]/module.std[None],w[:,len(module.cols):]],1)
        bias=module.projection.bias-effective[:,:len(module.cols)]@module.mean
        rest=[p for name,p in self.named_parameters() if not name.startswith('encoder.encoder.')]
        return reg*(sum(p.square().mean() for p in rest)+effective.square().mean()+bias.square().mean())
