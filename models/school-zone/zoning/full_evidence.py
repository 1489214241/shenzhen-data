"""Independent full-relation-input adapters, not official unmodified models.

Every model receives all legal reference rows, all school rows, memberships,
project multiplicity and the same context features as RetrievalScorer.
No RetrievalScorer scores, weights or embeddings enter these adapters.

Relation input is query-conditioned. Thus learned distances below are between
query-conditioned evidence tokens, not unconditional community coordinates.
School tokens are explicit labelled candidate anchors, including zero-support
schools. These two adaptations must be disclosed when naming paper families.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

vendor = Path(__file__).resolve().parents[1] / 'third_party'
if str(vendor) not in sys.path:
    sys.path.insert(0, str(vendor))
from TALENT.model.models.modernNCA import ModernNCA


def tokens(t, np_):
    """Lossless packing of px/ax/group bias/type; no top-k or statistics-only replacement."""
    b, r, _ = t['px'].shape
    s = t['ax'].shape[1]
    p = torch.cat([t['px'], t['gb'][None, :, None].expand(b, -1, -1),
                   torch.zeros(b, r, 1, device=t['px'].device)], -1)
    a = torch.cat([torch.zeros(b, s, np_, device=p.device), t['ax'],
                   torch.zeros(b, s, 1, device=p.device),
                   torch.ones(b, s, 1, device=p.device)], -1)
    v = torch.cat([t['valid'], torch.ones(b, s, device=p.device, dtype=torch.bool)], 1)
    return torch.cat([p, a], 1), v


class FullEvidenceModel(nn.Module):
    def __init__(self, method, np_, na, seed, dim=32):
        super().__init__()
        torch.manual_seed(seed)
        assert method in ('matching', 'modernnca', 'proto', 'deepsets')
        self.method, self.np, self.dim = method, np_, dim
        # Same trainable input adapter across families; official ModernNCA encoder.
        self.encoder = ModernNCA(d_in=np_ + na + 8, d_num=np_ + na + 8,
                                 d_out=2, dim=dim, dropout=.1, d_block=dim * 2,
                                 n_blocks=1, num_embeddings=None, sample_rate=1.)
        self.query = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.log_temperature = nn.Parameter(torch.tensor(2.302585093 if method == 'matching' else 0.))
        if method == 'deepsets':
            self.readout = nn.Sequential(nn.Linear(dim * 3 + 1, dim * 2), nn.SiLU(),
                                         nn.Linear(dim * 2, 1))

    def forward(self, t):
        x, valid = tokens(t, self.np)
        b, n, _ = x.shape
        r = t['px'].shape[1]
        s = t['ax'].shape[1]
        # Excluded own-project references must not affect BatchNorm or context.
        xv = x[valid]
        ev = self.encoder.post_encoder(self.encoder.encoder(xv))
        e = ev.new_zeros(b, n, self.dim)
        e[valid] = ev
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
        mem = t['mem']
        sums = (e * mass[..., None]).transpose(1, 2) @ mem
        count = mass @ mem
        proto = sums.transpose(1, 2) / count[..., None].clamp_min(1e-12)
        # Every school contributes equally to query context; raw references are
        # all encoded before this learned aggregation, never dropped beforehand.
        query = self.query(torch.cat([proto.mean(1), e[:, r:].mean(1)], -1))
        scale = self.log_temperature.exp().clamp(max=100.)
        if self.method == 'proto':
            logits = -(query[:, None] - proto).square().sum(-1) * scale
            return logits.softmax(-1)
        if self.method == 'deepsets':
            z = torch.cat([proto, e[:, r:], query[:, None].expand(-1, s, -1),
                           count.log1p()[..., None]], -1)
            return self.readout(z).squeeze(-1).softmax(-1)
        if self.method == 'modernnca':
            logits = -torch.linalg.vector_norm(query[:, None] - e, dim=-1) * scale
        else:
            logits = (F.normalize(query, dim=-1)[:, None] * F.normalize(e, dim=-1)).sum(-1) * scale
        # Common same-project mass convention, supplied to all models.
        logits = logits + mass.clamp_min(1e-30).log()
        logits = logits.masked_fill(~valid, -torch.inf)
        return logits.softmax(-1) @ mem

    def regularization(self, reg):
        return reg * sum(p.square().mean() for p in self.parameters())
