"""A frozen-ranker threshold probe, inspired by ATLOP's virtual TH label.

This is not the ATLOP architecture or its benchmark reproduction. The school
ranker is fixed; only the stopping boundary is learned from complete source
training labels. Partial label sets cannot supervise a complete stopping rule.
"""
import math
import numpy as np
import torch
from torch import nn


def query_features(logits, anchor_context):
    """School-permutation invariant, no identifiers or observed label counts."""
    if logits.shape[1] < 2:
        raise ValueError('At least two school candidates required')
    p = torch.softmax(logits, dim=1)
    pooled = (p[:, :, None] * anchor_context).sum(1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(1) / math.log(p.shape[1])
    best = logits.topk(2, dim=1).values
    gap = best[:, 0] - best[:, 1]
    return torch.cat([pooled, entropy[:, None], gap[:, None]], 1)


def at_loss(logits, threshold, labels, weights):
    """Paper's two CE terms, restricted to explicitly observed labels.

For this trial all training labels are complete source labels. Unknown entries
are still explicitly masked, so this helper cannot silently invent negatives.
Thresholds are separate from class logits; no in-place mutation of labels.
"""
    assert logits.shape == labels.shape
    assert threshold.shape == logits.shape[:1]
    assert ((labels == -1) | (labels == 0) | (labels == 1)).all()
    pos, neg = labels == 1, labels == 0
    pden = torch.logsumexp(torch.cat([threshold[:, None], logits.masked_fill(~pos, -torch.inf)], 1), 1)
    nden = torch.logsumexp(torch.cat([threshold[:, None], logits.masked_fill(~neg, -torch.inf)], 1), 1)
    positive = pos.sum(1) * pden - logits.masked_fill(~pos, 0).sum(1)
    return ((positive + nden - threshold) * weights).sum()


class ThresholdHead(nn.Module):
    def __init__(self, mean, scale, adaptive, seed, width=32):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.offset = nn.Parameter(torch.tensor(-1.))
        self.adaptive = adaptive
        if adaptive:
            self.net = nn.Sequential(nn.Linear(len(mean), width), nn.SiLU(), nn.Linear(width, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, logits, features):
        delta = self.offset.expand(logits.shape[0])
        if self.adaptive:
            delta = delta + self.net((features - self.mean) / self.scale).squeeze(1)
        return logits.max(1).values + delta

    def penalty(self, coefficient):
        value = (self.offset + 1).square()
        if self.adaptive:
            value = value + sum(p.square().mean() for p in self.net.parameters())
        return coefficient * value


def choose(margins):
    pred = np.asarray(margins) > 0
    empty = ~pred.any(1)
    pred[empty, np.asarray(margins)[empty].argmax(1)] = True
    return pred
