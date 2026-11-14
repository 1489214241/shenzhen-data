"""Normalization-only changes to the independent relation-input ModernNCA.

Not a new matching/readout architecture: all tokens, candidate anchors, query
encoder, Euclidean distance and school aggregation are inherited unchanged.
"""
from torch import nn
from .full_evidence import FullEvidenceModel


class LayerNormModernNCA(FullEvidenceModel):
    def __init__(self, np_, na, seed, dim=32):
        super().__init__('modernnca', np_, na, seed, dim)
        self._replace(self.encoder.post_encoder)

    @classmethod
    def _replace(cls, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                layer = nn.LayerNorm(child.num_features, eps=child.eps)
                layer.weight.data.copy_(child.weight.data)
                layer.bias.data.copy_(child.bias.data)
                setattr(module, name, layer)
            else:
                cls._replace(child)
