"""Targeted residual variants of the independent full-evidence ModernNCA-LN.

These are custom task adaptations, not FEAT/GMU reproductions. All reference
tokens and school anchors remain in the Euclidean-distance label aggregation.
"""
import math
import torch
from torch import nn
from .full_evidence import tokens
from .modernnca_normalization import LayerNormModernNCA


class TargetedModernNCA(LayerNormModernNCA):
    def __init__(self, np_, na, seed, variant, dim=32):
        super().__init__(np_, na, seed, dim)
        assert variant in ('schoolquery', 'adaptivegate', 'constantgate')
        self.variant = variant
        # Preserve the baseline RNG stream for dropout as well as initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 11013)
            if variant == 'schoolquery':
                self.residual = nn.Sequential(nn.Linear(dim * 3 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
            elif variant == 'adaptivegate':
                self.residual = nn.Sequential(nn.Linear(dim * 3 + 7, dim), nn.SiLU(), nn.Linear(dim, 1))
            else:
                self.offset = nn.Parameter(torch.zeros(()))
            if variant != 'constantgate':
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)

    def forward(self, t, details=False):
        x, valid = tokens(t, self.np)
        b, n, _ = x.shape
        r, s = t['px'].shape[1], t['ax'].shape[1]
        ev = self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e = ev.new_zeros(b, n, self.dim); e[valid] = ev
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
        mem = t['mem']
        count = mass @ mem
        proto = ((e * mass[..., None]).transpose(1, 2) @ mem).transpose(1, 2) / count[..., None].clamp_min(1e-12)
        anchors = e[:, r:]
        context = torch.cat([proto.mean(1), anchors.mean(1)], -1)
        query = self.query(context)
        scale = self.log_temperature.exp().clamp(max=100.)
        if self.variant == 'schoolquery':
            # Shared function over schools: no school identity parameters.
            z = torch.cat([query[:, None].expand(-1, s, -1), proto, anchors, count.log1p()[..., None]], -1)
            delta = self.residual(z)
            token_delta = torch.einsum('ns,bsd->bnd', mem, delta)
            distance = torch.linalg.vector_norm(query[:, None] + token_delta - e, dim=-1)
        else:
            distance = torch.linalg.vector_norm(query[:, None] - e, dim=-1)
        logits = (-distance * scale + mass.clamp_min(1e-30).log()).masked_fill(~valid, -torch.inf)
        base_logits = logits
        if self.variant == 'constantgate':
            delta = self.offset.expand(b, 1)
        elif self.variant == 'adaptivegate':
            # Finite channel distributions even when every reference is masked.
            ref_logits = logits[:, :r]
            available = valid[:, :r].any(-1)
            if r:
                safe = ref_logits.masked_fill(~available[:, None], 0.)
                rp = safe.softmax(-1) * valid[:, :r]
                rp = rp @ mem[:r]
                logzr = torch.logsumexp(safe, -1)
            else:
                rp = logits.new_zeros(b, s); logzr = logits.new_zeros(b)
            ap = logits[:, r:].softmax(-1)
            logza = torch.logsumexp(logits[:, r:], -1)
            denom = max(math.log(s), 1.)
            entropy = lambda p: -(p * p.clamp_min(1e-30).log()).sum(-1) / denom
            rm = mass[:, :r].sum(-1)
            scalars = torch.stack([entropy(rp), entropy(ap), rp.max(-1).values, ap.max(-1).values,
                (rp * ap).sum(-1), rm.log1p(), torch.where(available, logzr-logza, torch.zeros_like(logza))], -1)
            delta = self.residual(torch.cat([query, context, scalars], -1))
        if self.variant != 'schoolquery':
            # Equivalent to sigmoid(log Zref - log Zanchor + delta) channel gate.
            # delta=0 is exactly the original joint softmax, including count mass.
            logits = torch.cat([logits[:, :r] + delta, logits[:, r:]], -1)
        P = logits.softmax(-1) @ mem
        if details:
            return P, dict(delta=delta, base_logits=base_logits, logits=logits, valid=valid)
        return P
