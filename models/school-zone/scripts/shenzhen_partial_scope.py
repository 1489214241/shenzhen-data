"""Prove scope containment without assigning truth to unresolved subclauses.

An unsupported subtree is a fresh atom, scoped to its school and AST path.
It is neither true nor false and cannot match unknown text in another school.
Boolean structure and exclusions remain intact. These proofs say nothing about
the correctness of a model's original school selection or household eligibility.
"""
from sympy import Symbol, And, Or, Not
from sympy.logic.inference import satisfiable
from shenzhen_scope_implication import canonical, Unsupported, implies


def partial(node, school_id, path=()):
    op = node['op']
    if op in ('and', 'or', 'minus'):
        children = [partial(c, school_id, path + (i,))
                    for i, c in enumerate(node['children'])]
        if not children or (op == 'minus' and len(children) != 2):
            raise ValueError('Malformed Boolean AST')
        if op == 'minus':
            return (op, *children)
        flat = []
        for c in children:
            flat.extend(c[1:] if c[0] == op else [c])
        flat = sorted(set(flat))
        return (op, *flat) if len(flat) > 1 else flat[0]
    try:
        return canonical(node)
    except Unsupported:
        return ('opaque', school_id, '/'.join(map(str, path)))


def sat_proof(a, b):
    """Independent propositional solver verifies absence of A AND NOT B."""
    symbols = {}
    def convert(expr):
        if expr[0] == 'and': return And(*(convert(c) for c in expr[1:]))
        if expr[0] == 'or': return Or(*(convert(c) for c in expr[1:]))
        if expr[0] == 'minus': return And(convert(expr[1]), Not(convert(expr[2])))
        key = tuple(expr)
        if key not in symbols: symbols[key] = Symbol('atom_' + str(len(symbols)))
        return symbols[key]
    left, right = convert(a), convert(b)
    assert satisfiable(left) is not False, 'Vacuous trigger expression'
    assert satisfiable(And(left, Not(right))) is False, 'Invalid implication'
    return dict(valid=True, nonvacuous=True, counterexample_unsatisfiable=True,
                atoms=[list(k) for k in symbols], solver='sympy.logic.satisfiable',
                unknown_truth_assigned=False, geographic_atoms_evaluated=False)


def build(rows, old_graph):
    blocked = {r['school_id'] for r in old_graph['inventory']
               if r['skip_reason'] == 'university_eligibility_or_pending_member'}
    expressions = {r['school_id']: partial(r['ast'], r['school_id']) for r in rows
                   if r['school_id'] not in blocked and
                   r['scope_status'] == 'individual_scope_requires_site_applicability'}
    byid = {r['school_id']: r for r in rows}
    before = {(e['trigger_school_id'], e['implied_school_id']) for e in old_graph['edges']}
    additions = []
    for a, x in expressions.items():
        for b, y in expressions.items():
            if a == b or (a, b) in before or not implies(x, y): continue
            additions.append(dict(trigger_school_id=a, implied_school_id=b,
                trigger_school_name=byid[a]['school_name'], implied_school_name=byid[b]['school_name'],
                trigger_expression=x, implied_expression=y, independent_proof=sat_proof(x, y),
                trigger_evidence=byid[a]['evidence'], implied_evidence=byid[b]['evidence'],
                proof='partial_scope_containment_for_all_opaque_assignments'))
    return dict(edges=old_graph['edges'] + additions, added=additions,
                inventory=old_graph['inventory'], source_schools=len(rows),
                expressions_checked=len(expressions), blocked_school_ids=sorted(blocked),
                transitive_closure=False, target_gold_read=False, target_polygon_read=False)
