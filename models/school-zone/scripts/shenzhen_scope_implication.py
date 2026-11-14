"""Conservative source-syntax containment; no geometry, answers or graph closure.

Each edge is proved directly between two complete, unconditional source ASTs.
This proves a relationship between parsed scopes, not the parsing or enrollment
eligibility of a particular household. Unknown syntax is never a wildcard.
"""
import re


class Unsupported(ValueError):
    pass


def text(value):
    return re.sub(r'\s+', '', value).replace('（', '(').replace('）', ')')


def canonical(node):
    op = node['op']
    if op in ('or', 'and'):
        children = []
        for child in node['children']:
            c = canonical(child)
            children.extend(c[1:] if c[0] == op else [c])
        if not children:
            raise Unsupported('Empty Boolean expression')
        return (op, *sorted(set(children))) if len(set(children)) > 1 else children[0]
    if op == 'minus':
        if len(node['children']) != 2:
            raise Unsupported('Malformed exclusion')
        return (op, *(canonical(c) for c in node['children']))
    if op == 'road':
        # Explicit unresolved old/current names must not be silently discarded.
        if node.get('explicit_current_names') or node.get('excluded_notes'):
            raise Unsupported('Qualified road identity')
        atoms = sorted(set(('side', text(name), side) for name, side in node['pairs']))
        if not atoms or any(a[2] not in '东西南北' for a in atoms):
            raise Unsupported('Malformed side')
        return ('and', *atoms) if len(atoms) > 1 else atoms[0]
    if op in ('membership', 'admin', 'named'):
        if not text(node['text']):
            raise Unsupported('Empty identity')
        return (op, text(node['text']))
    if op == 'ring':
        # Order and identity preserved: no guessed street connectivity.
        return (op, *(text(n) for n in node['names']))
    # School references, addresses, partial/conditional/unknown clauses remain
    # unresolved here. Do not turn them into universal or empty predicates.
    raise Unsupported(op)


def implies(a, b):
    """Sufficient Boolean implication, deliberately incomplete."""
    if a == b:
        return True
    if a[0] == 'or':
        return all(implies(c, b) for c in a[1:])
    if b[0] == 'and':
        return all(implies(a, c) for c in b[1:])
    if b[0] == 'or':
        return any(implies(a, c) for c in b[1:])
    if a[0] == 'and':
        return any(implies(c, b) for c in a[1:])
    if a[0] == 'minus':
        if b[0] == 'minus':
            # A\X subset B\Y when A subset B and Y subset X.
            return implies(a[1], b[1]) and implies(b[2], a[2])
        return implies(a[1], b)
    return False


def build(observations, policy):
    restricted = {sid for r in policy['relations']
                  if r['kind'] in ('university', 'eligibility', 'pending')
                  for sid in r['school_ids']}
    expressions, audit = {}, []
    for row in observations:
        sid = row['school_id']
        reason = None
        if sid in restricted:
            reason = 'university_eligibility_or_pending_member'
        elif row['scope_status'] != 'individual_scope_requires_site_applicability':
            reason = row['scope_status']
        else:
            try:
                expressions[sid] = canonical(row['ast'])
            except Unsupported as exc:
                reason = str(exc)
        audit.append(dict(school_id=sid, school_name=row['school_name'],
            admitted=sid in expressions, skip_reason=reason,
            source_text=row['source_text'], evidence=row['evidence']))
    byid = {r['school_id']: r for r in observations}
    edges = []
    for a, x in expressions.items():
        for b, y in expressions.items():
            if a != b and implies(x, y):
                edges.append(dict(trigger_school_id=a, implied_school_id=b,
                    trigger_school_name=byid[a]['school_name'], implied_school_name=byid[b]['school_name'],
                    proof='direct_boolean_scope_implication',
                    trigger_expression=x, implied_expression=y,
                    trigger_evidence=byid[a]['evidence'], implied_evidence=byid[b]['evidence']))
    return dict(edges=edges, inventory=audit, transitive_closure=False,
        target_gold_read=False, meaning='Parsed base scope inclusion only; original school prediction remains uncertain.')


def expand_original(roots, graph):
    """One pass on original roots. Added members never trigger another edge."""
    roots = set(roots)
    return sorted(roots | {e['implied_school_id'] for e in graph['edges'] if e['trigger_school_id'] in roots})
