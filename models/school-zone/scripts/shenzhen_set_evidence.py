"""School-conditioned vector pooling, added residually to frozen-parent retrieval.

Deep Sets inspired, not a reproduction of the paper's set expansion experiment.
The input is the existing reference evidence; no IDs or unknown negative labels
are added. Weighted means and observed masses retain vector information before
the school decision, unlike scalar log-mass pooling alone.
"""
import torch
from torch import nn
from shenzhen_partial_model import BernoulliRetrieval

SCALES = (100., 300., 1000.)


def school_pool(embedding, weights, membership):
    """Reference-order invariant mean plus observed mass, including empty sets."""
    mass = weights @ membership
    summed = (embedding * weights.unsqueeze(-1)).transpose(1, 2) @ membership
    mean = summed.transpose(1, 2) / mass.clamp_min(1e-12).unsqueeze(-1)
    # A zero mass is an explicit absence of support, never a negative label.
    return torch.cat([mean, torch.log1p(mass).unsqueeze(-1)], dim=-1)


class SetEvidenceRetrieval(BernoulliRetrieval):
    def __init__(self, np_, na, start, seed, width=32, embedding_width=16):
        super().__init__(np_, na, start, seed, width)
        self.set_phi = nn.Sequential(nn.Linear(np_ + na + 6, width), nn.SiLU(),
                                     nn.Linear(width, embedding_width), nn.Tanh())
        self.set_rho = nn.Sequential(
            nn.Linear(na + 6 + len(SCALES) * (embedding_width + 1), width),
            nn.SiLU(), nn.Linear(width, 1))
        # Exact parent output at initialization; learn the residual rather than
        # replacing the established retrieval path with a random classifier.
        nn.init.zeros_(self.set_rho[-1].weight)
        nn.init.zeros_(self.set_rho[-1].bias)

    def residual(self, t):
        embedding = self.set_phi(t['px'])
        dist = t['pair'][:, :, t['meta'][0]] * 300
        mem = t['mem'][:t['pair'].shape[1]]
        blocks = []
        for scale in SCALES:
            weights = torch.exp(t['gb'][None] - dist / scale) * t['valid']
            blocks.append(school_pool(embedding, weights, mem))
        return self.set_rho(torch.cat([t['ax']] + blocks, dim=-1)).squeeze(-1)

    def forward(self, t):
        return super().forward(t) + self.residual(t)

    def regularization(self, reg):
        return super().regularization(reg) + reg * sum(
            p.square().mean() for net in [self.set_phi, self.set_rho] for p in net.parameters())

    def load_parent(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        expected = {k for k in self.state_dict() if k.startswith(('set_phi.', 'set_rho.'))}
        if set(missing) != expected or extra:
            raise ValueError(('Unexpected parent state', missing, extra))
