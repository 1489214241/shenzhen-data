"""Shared community estimation primitives; estimates are never certified labels."""
import re
import numpy as np


def nearest_separated(coords, points, identities):
    """Exact Hangzhou distance/second-distinct-unit criterion in metric CRS."""
    coords, points = np.asarray(coords), np.asarray(points)
    identities = np.asarray(identities, str)
    if not len(points) or len(points) != len(identities):
        raise ValueError('A nonempty identified point inventory is required')
    distances = np.linalg.norm(coords[:, None] - points[None], axis=-1)
    order = np.argsort(distances, axis=1)
    first = order[:, 0]
    d1 = distances[np.arange(len(coords)), first]
    d2 = np.array([next((distances[q, j] for j in order[q, 1:]
                        if identities[j] != identities[first[q]]), np.inf)
                   for q in range(len(coords))])
    strict = (d1 <= 1500) & (d2 >= 2*d1)
    loose = (d1 <= 3000) & (d2 >= 1.2*d1) & ~strict
    # Coincident *different* identities are ambiguous, including d1=d2=0.
    # Original Hangzhou formula is retained; caller audits such coincidences.
    return dict(point=first, distance=d1, second_distance=d2, strict=strict, loose=loose)


def identity(point):
    """Exact known policy identity, else a role-stripped POI name; no fuzzy merge.

    Multiple policy identities (e.g. parent community and subcommittee) remain
    distinct via their original UID rather than silently collapsing levels.
    """
    names = sorted(set(point.get('policy_names', [])))
    if len(names) == 1:
        return 'unit:' + names[0]
    if len(names) > 1:
        return 'ambiguous-poi:' + point['uid']
    name = point['name'].split('街道')[-1]
    name = re.sub(r'^(深圳市)?(龙华区|龙岗区)', '', name)
    name = re.sub(r'(居民委员会|居委会|党群服务中心|委员会|工作站|服务中心)$', '', name)
    return 'unit:' + name if name else 'poi:' + point['uid']


def membership_terms(ast, partial=False):
    """Positive community conditions only; exclusion objects never become positives.

    Only a bare membership or an OR branch can cover an entire unit. A conjunction
    or exclusion needs additional facts, so its membership is a partial feature.
    References, unknown and conditional clauses are not inferred here.
    """
    op = ast['op']
    if op == 'membership':
        yield ast['text'], partial
    elif op in ('or', 'and'):
        for child in ast['children']:
            yield from membership_terms(child, partial or op == 'and')
    elif op == 'minus':
        yield from membership_terms(ast['children'][0], True)


def community_features(points, assignments, observations):
    n, m = len(assignments['point']), len(observations)
    whole, part = {}, {}
    for j, row in enumerate(observations):
        if row['scope_status'].endswith('_pending'):
            continue
        for unit, partial in membership_terms(row['ast']):
            (part if partial else whole).setdefault(unit, set()).add(j)
    cs = np.zeros((n, m, 4), np.float32)
    names = np.full(n, '', dtype=object)
    for q, p in enumerate(assignments['point']):
        if p < 0 or not (assignments['strict'][q] or assignments['loose'][q]):
            continue
        names[q] = identity(points[p])
        w, pp = set(), set()
        for unit in points[p]['policy_names']:
            w |= whole.get(unit, set())
            pp |= part.get(unit, set())
        if w:
            cs[q, sorted(w), 0 if assignments['strict'][q] else 1] = 1
            cs[q, [s for s in range(m) if s not in w], 3] = 1
        if pp:
            cs[q, sorted(pp), 2] = 1
    return names.astype(str), np.asarray(assignments['strict'], bool), cs


def direct_estimates(cs, strict, known, negative):
    """Hangzhou's singleton whole-unit rule, still a partial positive in Shenzhen."""
    result = {}
    for q in range(len(cs)):
        schools = np.flatnonzero(cs[q, :, 0] > 0)
        if q not in known and strict[q] and len(schools) == 1 and not negative[q, schools[0]]:
            result[q] = [int(schools[0])]
    return result
