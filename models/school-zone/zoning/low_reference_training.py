"""Research-only support scarcity views and school-conditioned Modern readout.

Source episodes select at most one reference relation per school. This is a
training information mask, NOT a claim that a multi-school acquisition reveals
only one relation. Shenzhen retains the frozen at-least-one acquisition pool.
"""
import copy
import numpy as np
import torch
from . import evidence as EV
from .modernnca_multilabel import MultiLabelModernNCA
from .full_evidence import tokens


def choose_relations(x, eligible, seed, sparse):
    eligible = np.asarray(eligible, int)
    if not sparse:
        return eligible.copy()
    rng = np.random.default_rng(seed)
    result = []
    for school in range(x.school_n):
        ix = eligible[x.ref_schools[eligible] == school]
        if not len(ix):
            continue
        # Uniform over distinct projects, then their coordinate rows.
        groups = x.groups[x.ref_sites[ix]]
        group = rng.choice(np.unique(groups))
        result.append(int(rng.choice(ix[groups == group])))
    return np.array(sorted(result), int)


def rebuild_view(x, indices, coords, roads, raw_name):
    """Rebuild every support-derived field, including learned boundary evidence.

Only original singleton observations can contribute singleton boundary labels;
masking one member of a multi-positive observation does not make it singleton.
Static official policy, road geometry and community observations are retained.
"""
    indices = np.asarray(indices, int)
    v = copy.copy(x)
    v.ref_sites = x.ref_sites[indices].copy()
    v.ref_schools = x.ref_schools[indices].copy()
    v.pair = x.pair[:, indices].copy()
    original = {}
    for q, s in zip(x.ref_sites, x.ref_schools):
        original.setdefault(int(q), set()).add(int(s))
    known = {int(q): [int(s)] for q, s in zip(v.ref_sites, v.ref_schools)
             if len(original[int(q)]) == 1}
    v.pair[:, :, x.pair_blocks['learned_boundary']] = EV.learned_boundary(
        roads, coords, x.groups, known, v.ref_sites)
    v.anchor = x.anchor.copy()
    observed = np.zeros(x.n, bool)
    observed[v.ref_sites] = True
    v.anchor[:, :, x.anchor_blocks['name']] = np.where(
        observed[:, None, None], 0., raw_name)
    _, inv, counts = np.unique(x.groups[v.ref_sites], return_inverse=True, return_counts=True)
    v.group_bias = -np.log(counts[inv]).astype('float32')
    v.self_mask = x.groups[:, None] == x.groups[v.ref_sites][None]
    v.membership = torch.nn.functional.one_hot(torch.tensor(
        np.r_[v.ref_schools, np.arange(x.school_n)], dtype=torch.long), x.school_n).float()
    # No downstream accessor can retrieve hidden reference labels from y.
    v.y = np.full(x.n, -1, dtype=int)
    for q, ss in known.items():
        v.y[q] = ss[0]
    v.supervised = np.flatnonzero(v.y >= 0)
    return v


class SchoolLocalModernNCA(MultiLabelModernNCA):
    """Same parameters/loss as global model; query now conditioned on each school.

No school ID embedding or per-school learned parameters. Still shares static
policy/context features; 'local' describes the query construction, not an
assertion that no other school's evidence can affect context features.
"""
    def forward(self, t):
        x, valid = tokens(t, self.np)
        b, n, _ = x.shape
        r = t['px'].shape[1]
        ev = self.encoder.post_encoder(self.encoder.encoder(x[valid]))
        e = ev.new_zeros(b, n, self.dim)
        e[valid] = ev
        mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(t['ax'].shape[1])])[None] * valid
        mem = t['mem']
        assert bool((mem.sum(1) == 1).all())
        count = mass @ mem
        proto = ((e * mass[..., None]).transpose(1, 2) @ mem).transpose(1, 2)
        proto = proto / count[..., None].clamp_min(1e-12)
        query = self.query(torch.cat([proto, e[:, r:]], -1))
        school = mem.argmax(1)
        scores = -torch.linalg.vector_norm(query[:, school] - e, dim=-1)
        scores = scores * self.log_temperature.exp().clamp(max=100.)
        scores = (scores + mass.clamp_min(1e-30).log()).masked_fill(~valid, -torch.inf)
        offset = scores.max(1, keepdim=True).values
        school_mass = torch.exp(scores - offset) @ mem
        return school_mass.clamp_min(1e-30).log() + offset + self.binary_bias
