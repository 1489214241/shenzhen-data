"""Experimental ModernNCA + relation evidence; not the default model.

Retains the official ModernNCA encoder, Euclidean distance, optional SNS and
label-weighted readout. Adds the existing reference scorer's relation evidence
and school anchors. This is explicitly a hybrid, not unmodified ModernNCA.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from .model import RetrievalScorer

VENDOR = Path(__file__).resolve().parents[1] / 'third_party'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from TALENT.model.models.modernNCA import ModernNCA


class ModernNCARelation(RetrievalScorer):
    def __init__(self, np_, na, start, seed, sample_rate=1.0, encoding='none'):
        super().__init__(np_, na, start, seed, width=32)
        emb = None if encoding == 'none' else dict(
            type='PLREmbeddings', n_frequencies=4, frequency_scale=.1,
            d_embedding=4, lite=True)
        self.metric = ModernNCA(d_in=2, d_num=2, d_out=2, dim=64,
                               dropout=.1, d_block=128, n_blocks=1,
                               num_embeddings=emb, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.log_metric_scale = nn.Parameter(torch.tensor(-2.302585093))

    def embed(self, coords):
        if self.metric.num_embeddings is not None:
            coords = self.metric.num_embeddings(coords).flatten(1)
        return self.metric.post_encoder(self.metric.encoder(coords))

    def forward(self, t):
        # Encode all district locations jointly. Inputs contain coordinates only,
        # no query labels; the same coordinate normalization is used at inference.
        z = self.embed(t['coords'])
        distance = torch.cdist(z[t['qids']], z[t['refids']])
        pl = (t['pair'] @ self.p + t['gb'][None]
              + t['px'][:, :, self.np:] @ self.pc
              + self.pnet(t['px']).squeeze(-1)
              - self.log_metric_scale.exp() * distance)
        al = (t['anchor'] @ self.a + self.offset
              + t['ax'][:, :, self.na:] @ self.ac
              + self.anet(t['ax']).squeeze(-1))
        valid = t['valid']
        if self.training and self.sample_rate < 1:
            keep = torch.rand(pl.shape[1], device=pl.device) < self.sample_rate
            valid = valid & keep[None]
        pl = pl.masked_fill(~valid, -torch.inf)
        return torch.softmax(torch.cat([pl, al], 1), 1) @ t['mem']

    def regularization(self, reg):
        return super().regularization(reg) + reg * sum(
            p.square().mean() for p in self.metric.parameters())
