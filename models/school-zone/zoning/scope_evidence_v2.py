"""Versioned, condition-preserving evidence routes; no dataset/Gold access."""
import copy
import hashlib
import json

import numpy as np

from .conditional_school_sets import exact_identity, named_sufficient
from .conditional_scope_cells import sufficient_branches


TEXTUAL_KINDS = frozenset({'community_round07', 'residential_community', 'named_region',
                           'explicit_positive_repair', 'reviewed_administrative_identity',
                           'exact_sufficient_named_scope'})
ESTIMATE_KINDS = frozenset({'legacy_named_match', 'legacy_boundary', 'road_corridor', 'community_road'})
CHANNELS = ['scope_unknown', 'scope_outside_estimate', 'scope_inside_estimate']


def evidence_routes(kinds, positive=True):
    """Observed positive is partial. Estimated evidence is never a hard fact."""
    kinds = set(kinds)
    if not kinds or not kinds <= TEXTUAL_KINDS | ESTIMATE_KINDS | {'explicit_negative'}:
        raise ValueError('Unknown evidence type')
    textual = bool(kinds & TEXTUAL_KINDS) if positive else 'explicit_negative' in kinds
    if positive and 'explicit_negative' in kinds:
        raise ValueError('Contradictory polarity')
    return dict(feature=True, reference=positive, training=textual,
                hard_constraint=textual, evidence_type='textual_proof' if textual else 'estimate',
                complete_school_set=False, enrollment_eligibility=False)


def compile_base_branches(observations, policy):
    """Compile independently stated base scopes, never a university member list.

    A pending household/integrated-school statement stays pending. An explicitly
    separated mixed statement may retain supported unconditional OR branches.
    No AND child/exclusion is dropped to manufacture a sufficient condition.
    """
    tags = {r['school_id']: sorted({p['kind'] for p in policy['relations']
                                  if r['school_id'] in p['school_ids']}) for r in observations}
    allowed = {'individual_scope_requires_site_applicability', 'mixed_qualification_partial_scope',
               'reviewed_street_geographic_scope'}
    records = []
    for row in observations:
        active = row['scope_status'] in allowed
        branches = sufficient_branches(row['ast']) if active else []
        for expression in branches:
            encoded = json.dumps([row['school_id'], expression], ensure_ascii=False, separators=(',', ':'))
            records.append(dict(branch_id=hashlib.sha256(encoded.encode()).hexdigest()[:20],
                school_id=row['school_id'], expression=expression, source_text=row['source_text'],
                evidence=copy.deepcopy(row['evidence']), year=policy['year'], stage=policy['stage'],
                relation_tags=tags[row['school_id']], applicability='unknown',
                role='independently_stated_base_scope', complete_school_set=False,
                enrollment_eligibility=False, university_member_expansion=False))
    return records


def exact_named_relations(observations, sites):
    rows = []
    allowed = {'individual_scope_requires_site_applicability', 'mixed_qualification_partial_scope',
               'reviewed_street_geographic_scope'}
    for obs in observations:
        if obs['scope_status'] not in allowed:
            continue
        names = named_sufficient(obs['ast'])
        for q, site in enumerate(sites):
            if exact_identity(site, names):
                rows.append(dict(q=q, school_id=obs['school_id'], site_id=site['coordinate_group_id'],
                    kind='exact_sufficient_named_scope', source_text=obs['source_text'], evidence=obs['evidence']))
    return rows


def append_scope_features(x, states):
    """The same three-valued whole-scope channel for all cities and schools."""
    if states.shape != (x.n, x.school_n) or not np.isin(states, [-1, 0, 1]).all():
        raise ValueError('Malformed three-valued scope observation')
    if 'whole_scope' in x.anchor_blocks or 'whole_scope' in x.pair_blocks:
        raise ValueError('Scope block already appended')
    out = copy.deepcopy(x)
    encoded = (states[..., None] == np.array([-1, 0, 1])).astype(np.float32)
    out.pair_blocks = dict(x.pair_blocks, whole_scope=slice(x.np_, x.np_ + 3))
    out.anchor_blocks = dict(x.anchor_blocks, whole_scope=slice(x.na, x.na + 3))
    out.pair = np.concatenate([x.pair, encoded[:, x.ref_schools]], -1)
    out.anchor = np.concatenate([x.anchor, encoded], -1)
    return out


def apply_verified_constraints(prediction, positive, negative):
    """No closure; no estimated memberships; no invented singleton fallback."""
    pos, neg = set(positive), set(negative)
    if pos & neg:
        raise ValueError('Conflicting verified evidence requires review')
    return sorted((set(prediction) | pos) - neg)
