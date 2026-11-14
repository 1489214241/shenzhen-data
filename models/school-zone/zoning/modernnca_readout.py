"""Experimental school-content readout; not a TabR reproduction or default model.

The original query, encoder and Euclidean vote stay intact. A zero-initialized
shared school head corrects log vote mass. The summary control receives the same
input through the encoder, but deliberately omits the new reference-content
pooling path. This is an ablation, not a claim that it lacks access to evidence.
"""
import torch
from torch import nn
from .full_evidence import tokens
from .modernnca_normalization import LayerNormModernNCA


class ReadoutModernNCA(LayerNormModernNCA):
    def __init__(self, np_, na, seed, variant='content', dim=32):
        super().__init__(np_, na, seed, dim)
        assert variant in ('identity', 'summary', 'content')
        self.variant = variant
        self.na = na
        # Common: direct school evidence/context plus seven scalar summaries.
        nin = na + 6 + 7
        if variant == 'content':
            # Reference-only weighted raw evidence mean/variance + embedding mean.
            nin += 2 * (np_ + na + 6) + dim
        self.head_input = nin
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 419)
            if variant != 'identity':
                hidden = max(1, round(10000 / (nin + 2)))
                self.head = nn.Sequential(nn.Linear(nin, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def pool(value, weights, membership, denominator):
        return ((value * weights[..., None]).transpose(1, 2) @ membership).transpose(1, 2) / denominator[..., None].clamp_min(1e-30)

    def components(self, t):
        x, valid = tokens(t, self.np)
        ev = self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e = ev.new_zeros(*x.shape[:2], self.dim)
        e[valid] = ev
        r, s = t['px'].shape[1], t['ax'].shape[1]
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
        count = mass @ t['mem']
        proto = self.pool(e, mass, t['mem'], count)
        query = self.query(torch.cat([proto.mean(1), e[:, r:].mean(1)], -1))
        distance = torch.linalg.vector_norm(query[:, None] - e, dim=-1)
        logits = -self.log_temperature.exp().clamp(max=100.) * distance + mass.clamp_min(1e-30).log()
        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        p = weights @ t['mem']
        return p, e, mass, weights, distance

    def school_features(self, t, components):
        p, e, mass, weights, distance = components
        r = t['px'].shape[1]
        mem = t['mem'][:r]
        rw = weights[:, :r]
        ref_vote = rw @ mem
        ref_count = mass[:, :r] @ mem
        # School identity rows are in the same order as candidate schools.
        anchor_vote = weights[:, r:]
        dist_mean = self.pool(distance[:, :r, None], rw, mem, ref_vote).squeeze(-1)
        dist_second = self.pool(distance[:, :r, None].square(), rw, mem, ref_vote).squeeze(-1)
        dist_var = (dist_second - dist_mean.square()).clamp_min(0)
        scalar = torch.stack([p.clamp_min(1e-30).log(), ref_vote / p.clamp_min(1e-30),
                              anchor_vote / p.clamp_min(1e-30), ref_count.log1p(),
                              (ref_count == 0).to(p.dtype), dist_mean, dist_var], -1)
        common = torch.cat([t['ax'], scalar], -1)
        if self.variant != 'content':
            return common
        # Mask before squaring: excluded extreme/NaN values must have no effect.
        raw = t['px'].masked_fill(~t['valid'][..., None], 0.)
        mean = self.pool(raw, rw, mem, ref_vote)
        second = self.pool(raw.square(), rw, mem, ref_vote)
        variance = (second - mean.square()).clamp_min(0)
        emean = self.pool(e[:, :r], rw, mem, ref_vote)
        return torch.cat([common, mean, variance, emean], -1)

    def forward(self, t, bypass=False):
        parts = self.components(t)
        p = parts[0]
        if self.variant == 'identity' or bypass:
            return p
        correction = self.head(self.school_features(t, parts)).squeeze(-1)
        return (p.clamp_min(1e-30).log() + correction).softmax(-1)
