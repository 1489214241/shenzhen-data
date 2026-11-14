"""Extend a target reference bank using observed positive relations only."""
from copy import deepcopy
import numpy as np
import torch
from zoning import evidence as E


def extend(x, dist, rules, added_pairs, boundary_known, old_boundary_known):
    old_pairs = set(zip(map(int, x.ref_sites), map(int, x.ref_schools)))
    added = sorted(set(added_pairs) - old_pairs)
    held = [(q, s) for q, s in added if q in dist.not_reference]
    added = [(q, s) for q, s in added if q not in dist.not_reference]
    out = deepcopy(x)
    if not added: return out, added, held
    oldcol = {int(q): i for i, q in enumerate(x.ref_sites)}
    new_sites = sorted({q for q, s in added} - set(oldcol))
    ls = E.lines(dist.lines_key or dist.city)
    boundary = E.learned_boundary(ls, dist.coords, dist.groups, old_boundary_known, x.ref_sites)
    np.testing.assert_array_equal(boundary, x.pair[:, :, x.pair_blocks['learned_boundary']])
    if old_boundary_known != boundary_known:
        out.pair[:, :, out.pair_blocks['learned_boundary']] = E.learned_boundary(
            ls, dist.coords, dist.groups, boundary_known, x.ref_sites)
    geometry = {}
    if new_sites:
        ri = np.asarray(new_sites, int)
        cross, within = E.crossing(ls, dist.coords, dist.coords[ri])
        border = E.learned_boundary(ls, dist.coords, dist.groups, boundary_known, ri)
        delta = E.geometry(dist.coords[ri][None] - dist.coords[:, None], 300)
        names = dist.community_name; strict = dist.community_strict
        same = (names[:, None] == names[ri][None]) & (names[:, None] != '')
        both = strict[:, None] & strict[ri][None]
        comm = np.stack([same & both, (~same) & both], -1).astype(np.float32)
        for i, q in enumerate(new_sites):
            a = np.zeros((x.n, x.np_), np.float32)
            for key, values in [('distance', delta), ('crossing', cross), ('within_2km', within),
                                ('learned_boundary', border), ('same_community', comm)]:
                a[:, x.pair_blocks[key]] = values[:, i]
            geometry[q] = a
    cols = []; states = rules.statuses
    for q, s in added:
        a = out.pair[:, oldcol[q]].copy() if q in oldcol else geometry[q].copy()
        a[:, x.pair_blocks['named_reference']] = 0
        a[:, x.pair_blocks['policy']] = x.anchor[:, s, x.anchor_blocks['policy']]
        a[:, x.pair_blocks['agreement']] = ((states[:, s] >= 0) & (states[q, s] >= 0) & (states[:, s] == states[q, s])).astype(np.float32)
        a[:, x.pair_blocks['disagreement']] = ((states[:, s] >= 0) & (states[q, s] >= 0) & (states[:, s] != states[q, s])).astype(np.float32)
        cols.append(a)
    out.pair = np.concatenate([out.pair, np.stack(cols, axis=1)], axis=1)
    out.ref_sites = np.r_[x.ref_sites, [q for q, s in added]].astype(int)
    out.ref_schools = np.r_[x.ref_schools, [s for q, s in added]].astype(int)
    _, inv, count = np.unique(out.groups[out.ref_sites], return_inverse=True, return_counts=True)
    out.group_bias = -np.log(count[inv]).astype(np.float32)
    out.self_mask = out.groups[:, None] == out.groups[out.ref_sites][None]
    out.membership = torch.nn.functional.one_hot(torch.tensor(
        np.r_[out.ref_schools, np.arange(out.school_n)], dtype=torch.long), out.school_n).float()
    assert np.isfinite(out.pair).all() and np.isfinite(out.anchor).all()
    return out, added, held
