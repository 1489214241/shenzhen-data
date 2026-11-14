"""School-set hypotheses with explicit provenance, not certified complete labels.

No graph closure, geographic lookup, target labels, or model training occurs here.
Unknown applicability can be ranked as a hypothesis but is never a verified fact.
"""
import math
import re
import numpy as np


def normalize(value):
    return re.sub(r'\s+', '', value).replace('（', '(').replace('）', ')')


def exact_identity(site, names):
    identities = {normalize(n) for n in site.get('coordinate_group_building_names', [site['building_name']])}
    return len(identities) == 1 and bool(identities & {normalize(n) for n in names})


def catalogue(policy, graphs):
    """Every multi-school member must have a direct proof from the SAME root.

    A->B and B->C cannot produce A,B,C. A root with a directly proved
    implication uses that set in place of a contradictory singleton. Scope
    inclusions are lower bounds, not a claim of exhaustive eligibility.
    """
    schools = policy['school_ids']
    scopes = {r['school_id']: r for r in policy['school_scope_inventory']}
    records = []
    for school in schools:
        records.append(dict(candidate_id='single:' + school, school_ids=[school],
            root_school_ids=[school], kind='singleton', exact_site_names=[],
            evidence=scopes.get(school, {}).get('evidence', []),
            condition=scopes.get(school, {}).get('base_scope_text', ''),
            complete_school_set=False))
    for r in policy['relations']:
        if r['kind'] != 'shared' or len(r['school_ids']) < 2:
            continue
        records.append(dict(candidate_id='rule:' + r['rule_id'], school_ids=sorted(r['school_ids']),
            root_school_ids=r.get('trigger_school_ids', []), kind='shared_clause',
            exact_site_names=r.get('exact_site_names', []), evidence=r['evidence'],
            condition=r['reason'], complete_school_set=False))
    for root in schools:
        members = {root}; proofs = []
        for graph in graphs:
            for edge in graph['edges']:
                if edge['trigger_school_id'] == root:
                    members.add(edge['implied_school_id']); proofs.append(edge)
        for r in policy['relations']:
            if r['kind'] == 'shared' and root in r.get('trigger_school_ids', []):
                members.update(r['school_ids']); proofs.append(r)
        if len(members) > 1:
            records.append(dict(candidate_id='direct:' + root, school_ids=sorted(members),
                root_school_ids=[root], kind='direct_root_scope', exact_site_names=[],
                evidence=proofs, condition=scopes.get(root, {}).get('base_scope_text', ''),
                complete_school_set=False))
    direct_roots = {r['root_school_ids'][0] for r in records if r['kind'] == 'direct_root_scope'}
    records = [r for r in records if r['kind'] != 'singleton' or r['school_ids'][0] not in direct_roots]
    for r in records:
        assert set(r['school_ids']) <= set(schools)
        r.update(year=policy['year'], stage=policy['stage'],
            scope_applicability='unknown', enrollment_eligibility=False)
    return records


def site_candidates(records, site, lower_bound=(), excluded=()):
    """Union only input-positive relations, never recursively generated members.

    lower_bound may include existing estimated positives. Preserve that status
    in the caller; the union is an output hypothesis, not complete supervision.
    An explicitly excluded school invalidates its candidate, rather than
    silently trimming a shared set into a different unsupported set.
    """
    lower = set(lower_bound); excluded = set(excluded)
    if lower & excluded:
        raise ValueError('Positive/exclusion conflict must be reviewed')
    merged = {}; audit = []
    for r in records:
        schools = set(r['school_ids']) | lower
        named = exact_identity(site, r.get('exact_site_names', []))
        # A named clause does not certify extra lower-bound members supplied
        # by estimated administrative/road evidence elsewhere in the pipeline.
        status = 'not_satisfied' if schools & excluded else ('satisfied' if named and not (lower-set(r['school_ids'])) else 'unknown')
        audit.append(dict(candidate_id=r['candidate_id'], applicability=status,
            reason='explicit_school_exclusion' if status == 'not_satisfied' else
                   ('exact_official_site_identity' if named else 'residential_scope_not_independently_confirmed')))
        if status == 'not_satisfied':
            continue
        key = tuple(sorted(schools))
        if key not in merged:
            merged[key] = dict(school_ids=list(key), candidate_ids=[], applicability=status,
                complete_school_set=False, lower_bound=sorted(lower), enrollment_eligibility=False)
        merged[key]['candidate_ids'].append(r['candidate_id'])
        if status == 'satisfied':
            merged[key]['applicability'] = 'satisfied'
    # Positive-only evidence can be a minimal output hypothesis, but not a
    # declaration that unobserved schools are negative or that this is complete.
    if lower:
        key = tuple(sorted(lower))
        merged.setdefault(key, dict(school_ids=list(key), candidate_ids=['positive_lower_bound_only'],
            applicability='unknown', complete_school_set=False, lower_bound=sorted(lower),
            enrollment_eligibility=False))
    candidates = sorted(merged.values(), key=lambda c: (len(c['school_ids']), c['school_ids']))
    if not candidates:
        raise ValueError('No admissible candidate; do not manufacture a fallback')
    return candidates, audit


def scores(probability, candidates, school_ids, beta):
    """Geometric mean school support times |S|**beta, in log space.

    beta is selected once using source validation, never target cardinalities.
    This is a structured decoder of frozen Modern probabilities, not a newly
    trained set encoder. Every member must be supported; sums cannot hide a
    near-zero-probability extra school behind a strong member.
    """
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probability')
    byid = {s: j for j, s in enumerate(school_ids)}
    lp = np.log(np.maximum(p, 1e-30))
    return np.array([np.mean([lp[byid[s]] for s in c['school_ids']]) +
                     beta * math.log(len(c['school_ids'])) for c in candidates])


def select(probability, candidates, school_ids, beta):
    values = scores(probability, candidates, school_ids, beta)
    # Explicit ties prefer smaller sets, then the stable school IDs.
    order = sorted(range(len(candidates)), key=lambda j: (-values[j], len(candidates[j]['school_ids']), candidates[j]['school_ids']))
    winner = order[0]
    return dict(candidates[winner], score=float(values[winner]),
        margin=float(values[winner]-values[order[1]]) if len(order)>1 else None,
        status='model_selected_set_hypothesis')


def named_sufficient(node):
    """Exact named leaves sufficient on their own; never drop AND/exclusions.

    Only an OR branch with a named leaf certifies this syntactic condition.
    Household qualifiers, minus, roads, admin points and school references stay
    unresolved; this deliberately sacrifices coverage for auditable identity.
    """
    if node['op'] == 'named':
        return {node['text']}
    if node['op'] == 'or':
        return set().union(*(named_sufficient(c) for c in node['children']))
    return set()
