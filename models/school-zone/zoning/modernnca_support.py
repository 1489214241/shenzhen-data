"""Support-conditioned research heads; CNAPS/MetaOptNet inspired, not replicas.

Preserves lossless tokens, LN encoder, query generator, memberships and anchors.
Metric statistics use references only (school anchors are a different token type).
Reference weights are the existing inverse building-group multiplicities.
"""
import math
import torch
from .full_evidence import tokens
from .modernnca_normalization import LayerNormModernNCA


def support_covariance(e, weight):
    """Population covariance, equal total mass per building group.

No Bessel correction based on the number of duplicated rows. Zero/one effective
group yields identity metric via reliability=0, including multiple points in one
building. All legal rows still participate in the final prediction.
"""
    count = weight.sum(-1)
    w = weight / count[:, None].clamp_min(1e-12)
    mean = (e * w[..., None]).sum(1)
    z = e - mean[:, None]
    cov = (z * w[..., None]).transpose(1, 2) @ z
    reliability = torch.where(count > 1.00001, count / (count + 1), 0.)
    return cov, reliability, count


def ridge_scores(e, query, weight, membership, penalty=.1):
    """Weighted mean square loss + penalty*||W||^2, unpenalized intercept.

All anchors/references included; soft multi-school membership is the target.
Primal d x d solve, differentiable through support and query representations.
"""
    e = e / math.sqrt(e.shape[-1]); query = query / math.sqrt(query.shape[-1])
    w = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    xm = (e * w[..., None]).sum(1)
    ym = w @ membership
    z = e - xm[:, None]
    gram = (z * w[..., None]).transpose(1, 2) @ z
    cross = (z * w[..., None]).transpose(1, 2) @ membership
    eye = torch.eye(e.shape[-1], device=e.device, dtype=e.dtype)
    coefficients = torch.linalg.solve(gram + penalty * eye, cross)
    return ((query-xm)[:, None] @ coefficients).squeeze(1) + ym


class SupportModernNCA(LayerNormModernNCA):
    def __init__(self, np_, na, seed, variant, dim=32, strength=.5, ridge_penalty=.1):
        super().__init__(np_, na, seed, dim)
        assert variant in ('identity', 'scalar', 'diagonal', 'covariance', 'ridge')
        self.variant, self.strength, self.ridge_penalty = variant, strength, ridge_penalty

    def encode_support(self, t):
        x, valid = tokens(t, self.np)
        b, n, _ = x.shape
        r, s = t['px'].shape[1], t['ax'].shape[1]
        ev = self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e = ev.new_zeros(b, n, self.dim); e[valid] = ev
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
        count = mass @ t['mem']
        proto = ((e * mass[..., None]).transpose(1, 2) @ t['mem']).transpose(1, 2) / count[..., None].clamp_min(1e-12)
        query = self.query(torch.cat([proto.mean(1), e[:, r:].mean(1)], -1))
        return e, query, mass, valid

    def readout(self, e, query, mass, valid, t, variant=None, details=False):
        kind = variant or self.variant
        r = t['px'].shape[1]
        scale = self.log_temperature.exp().clamp(max=100.)
        info = {}
        if kind == 'ridge':
            logits = ridge_scores(e, query, mass, t['mem'], self.ridge_penalty)
            # Membership regression has scores around [0,1], unlike distances.
            # Fixed multiplier 10, common learned temperature; declared a priori.
            probability = (10 * scale * logits).softmax(-1)
        else:
            delta = query[:, None] - e
            if kind == 'identity' or self.strength == 0:
                distance = torch.linalg.vector_norm(delta, dim=-1)
            else:
                cov, reliability, count = support_covariance(e[:, :r], mass[:, :r])
                alpha = self.strength * reliability
                diagonal = cov.diagonal(dim1=-2, dim2=-1)
                if kind == 'scalar':
                    diagonal = diagonal.mean(-1, keepdim=True).expand_as(diagonal)
                if kind in ('scalar', 'diagonal'):
                    variance = (1-alpha[:, None]) + alpha[:, None] * diagonal
                    distance = torch.linalg.vector_norm(delta / variance.sqrt()[:, None], dim=-1)
                else:
                    eye = torch.eye(self.dim, device=e.device, dtype=e.dtype)
                    regularized = alpha[:, None, None] * cov + (1-alpha[:, None, None]) * eye
                    chol = torch.linalg.cholesky(regularized)
                    white = torch.linalg.solve_triangular(chol, delta.transpose(1, 2), upper=False)
                    distance = torch.linalg.vector_norm(white, dim=1)
                info = dict(group_count=count, reliability=reliability, covariance=cov)
            logits = (-scale * distance + mass.clamp_min(1e-30).log()).masked_fill(~valid, -torch.inf)
            probability = logits.softmax(-1) @ t['mem']
        return (probability, info) if details else probability

    def forward(self, t, details=False):
        return self.readout(*self.encode_support(t), t, details=details)
