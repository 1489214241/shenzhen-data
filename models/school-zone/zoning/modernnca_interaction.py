"""Full-evidence residual adapters. ISAB/TAML inspired, not full paper replicas."""
import math
import torch
from torch import nn
from .full_evidence import tokens
from .modernnca_normalization import LayerNormModernNCA


class WeightedMAB(nn.Module):
    """Official Set Transformer MAB arithmetic, with key mass/mask added."""
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_o = nn.Linear(dim, dim)
        self.ln0, self.ln1 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, q, k, mass=None):
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)
        width = self.dim // self.heads
        qq, kk, vv = [torch.cat(x.split(width, 2), 0) for x in (q, k, v)]
        logits = qq.bmm(kk.transpose(1, 2)) / math.sqrt(self.dim)
        if mass is not None:
            bias = mass.clamp_min(1e-30).log().masked_fill(mass <= 0, -torch.inf)
            logits = logits + bias.repeat(self.heads, 1)[:, None]
        attn = logits.softmax(-1)
        out = torch.cat((qq + attn.bmm(vv)).split(q.shape[0], 0), 2)
        out = self.ln0(out)
        return self.ln1(out + torch.relu(self.fc_o(out)))


class InteractionModernNCA(LayerNormModernNCA):
    def __init__(self, np_, na, seed, variant, dim=32):
        super().__init__(np_, na, seed, dim)
        assert variant in ('identity', 'rowmlp', 'statistics', 'isab')
        self.variant = variant
        # Preserve baseline CPU/CUDA dropout streams. New modules initialize on CPU.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 173)
            if variant == 'isab':
                self.inducing = nn.Parameter(torch.empty(1, 8, dim))
                nn.init.xavier_uniform_(self.inducing)
                self.mab0, self.mab1 = WeightedMAB(dim), WeightedMAB(dim)
                self.output = nn.Linear(dim, dim)
            elif variant in ('rowmlp', 'statistics'):
                nin, hidden = (dim, 154) if variant == 'rowmlp' else (5*dim+2, 51)
                self.hidden = nn.Sequential(nn.Linear(nin, hidden), nn.SiLU())
                self.output = nn.Linear(hidden, dim)
            if variant != 'identity':
                nn.init.zeros_(self.output.weight)
                nn.init.zeros_(self.output.bias)

    def encode(self, t):
        x, valid = tokens(t, self.np)
        ev = self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e = ev.new_zeros(*x.shape[:2], self.dim)
        e[valid] = ev
        s = t['ax'].shape[1]
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
        return e, mass, valid

    def statistics(self, e, mass, t):
        r = t['px'].shape[1]
        ref, w, mem = e[:, :r], mass[:, :r], t['mem'][:r]
        count = w @ mem
        denom = count[..., None].clamp_min(1e-12)
        mean = ((ref*w[..., None]).transpose(1, 2) @ mem).transpose(1, 2) / denom
        second = ((ref.square()*w[..., None]).transpose(1, 2) @ mem).transpose(1, 2) / denom
        var = (second-mean.square()).clamp_min(0)
        local = torch.cat([mean, var, count.log1p()[..., None]], -1)
        # Shared per-school statistics, no permanent class-ID parameters.
        assigned = torch.einsum('ns,bsd->bnd', t['mem'], local)
        glob = local.mean(1)[:, None].expand(-1, e.shape[1], -1)
        return torch.cat([e, assigned, glob], -1)

    def contextualize(self, e, mass, valid, t):
        if self.variant == 'identity':
            return e
        if self.variant == 'isab':
            h = self.mab0(self.inducing.expand(e.shape[0], -1, -1), e, mass)
            z = self.mab1(e, h)
        elif self.variant == 'statistics':
            z = self.hidden(self.statistics(e, mass, t))
        else:
            z = self.hidden(e)
        return (e + self.output(z)) * valid[..., None]

    def readout(self, e, mass, valid, t):
        r = t['px'].shape[1]
        count = mass @ t['mem']
        proto = ((e*mass[..., None]).transpose(1, 2) @ t['mem']).transpose(1, 2) / count[..., None].clamp_min(1e-12)
        query = self.query(torch.cat([proto.mean(1), e[:, r:].mean(1)], -1))
        distance = torch.linalg.vector_norm(query[:, None]-e, dim=-1)
        logits = -self.log_temperature.exp().clamp(max=100.)*distance + mass.clamp_min(1e-30).log()
        return logits.masked_fill(~valid, -torch.inf).softmax(-1) @ t['mem']

    def forward(self, t, bypass=False):
        e, mass, valid = self.encode(t)
        if not bypass:
            e = self.contextualize(e, mass, valid, t)
        return self.readout(e, mass, valid, t)
