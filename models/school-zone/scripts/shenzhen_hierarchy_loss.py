"""Partial-observation adaptation of C-HMCNN Eq. 4 (not a reproduction).

R[parent, child] means child implies parent. Self edges are mandatory.
Only observed positives can explain a positive parent; an unknown is never
assigned a negative label. A known negative parent can penalize descendants.
"""
import torch
from torch.nn import functional as TF


def masked_max_constraint_loss(logits, labels, weight, relation):
    if logits.ndim != 2 or labels.shape != logits.shape:
        raise ValueError('Expected matching batch-school arrays')
    n = logits.shape[1]
    if relation.shape != (n, n) or relation.dtype != torch.bool:
        raise ValueError('Expected Boolean parent-child relation matrix')
    if not relation.diagonal().all():
        raise ValueError('Every school must imply itself')
    if not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
        raise ValueError('Invalid observed label state')
    if weight.shape != (len(logits),):
        raise ValueError('Expected one weight per row')
    # Unknown/negative parents use the full max, but unknown-parent loss is
    # subsequently masked. Positive parents always have their own positive
    # self-edge, so no empty max or artificial finite sentinel is necessary.
    allowed = relation[None] & ((labels[:, :, None] != 1) | (labels[:, None, :] == 1))
    selected = logits[:, None, :].expand(-1, n, -1).masked_fill(~allowed, -torch.inf).max(2).values
    cells = TF.binary_cross_entropy_with_logits(selected, labels.clamp_min(0), reduction='none')
    return (cells.masked_fill(labels < 0, 0).sum(1) * weight).sum()


def relation_matrix(school_ids, edges):
    index = {s: i for i, s in enumerate(school_ids)}
    if len(index) != len(school_ids):
        raise ValueError('Duplicate school identity')
    r = torch.eye(len(index), dtype=torch.bool)
    for e in edges:
        r[index[e['implied_school_id']], index[e['trigger_school_id']]] = True
    return r
