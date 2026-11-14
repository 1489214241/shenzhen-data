"""Warm-start the full Modern encoder with lossless extra evidence channels."""
import torch

from .modernnca_multilabel import MultiLabelModernNCA


def expanded_model(state, seed, device='cpu'):
    model = MultiLabelModernNCA(44, 30, seed, 32).to(device)
    mapped = {k: v.detach().clone().to(device) for k, v in state.items()}
    old = mapped['encoder.encoder.weight']
    if old.shape != (32, 76):
        raise ValueError('Expected the frozen 41/27 Modern teacher')
    new = torch.zeros((32, 82), dtype=old.dtype, device=device)
    # Original token: pair41, anchor27, context6, group_mass1, token_type1.
    new[:, :41] = old[:, :41]
    new[:, 44:71] = old[:, 41:68]
    new[:, 74:] = old[:, 68:]
    mapped['encoder.encoder.weight'] = new
    model.load_state_dict(mapped)
    return model


def observed_mean_loss(logits, labels, weight):
    """Equal property-row objective; unknown entries have exactly zero gradient.

    Complete source singletons include every observed negative school, so
    their loss penalizes extra outputs, not just relative singleton ranking.
    """
    observed = labels >= 0
    if not observed.any(1).all():
        raise ValueError('Training rows need at least one observed relation')
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp_min(0), reduction='none')
    row = loss.masked_fill(~observed, 0).sum(1) / observed.sum(1)
    return (row * weight).sum()
