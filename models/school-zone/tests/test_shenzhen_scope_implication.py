import importlib.util
import itertools
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('scope_implication', Path(__file__).resolve().parents[1] / 'scripts/shenzhen_scope_implication.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def value(x, env):
    if x[0] == 'or': return any(value(c, env) for c in x[1:])
    if x[0] == 'and': return all(value(c, env) for c in x[1:])
    if x[0] == 'minus': return value(x[1], env) and not value(x[2], env)
    return env[x]


def test_proofs_are_boolean_sound():
    atoms = [('membership', n) for n in 'ABC']
    expressions = atoms + [(op, a, b) for op in ('and', 'or', 'minus') for a in atoms for b in atoms]
    expressions += [('minus', ('or', atoms[0], atoms[1]), atoms[2]),
                    ('and', ('or', atoms[0], atoms[1]), atoms[2])]
    for a, b in itertools.product(expressions, repeat=2):
        if M.implies(a, b):
            for truth in itertools.product([False, True], repeat=3):
                env = dict(zip(atoms, truth))
                assert not value(a, env) or value(b, env), (a, b, env)


def test_road_part_of_whole_scope():
    a = dict(op='road', pairs=[['和平路', '东'], ['东环一路', '南']])
    b = dict(op='or', children=[a, dict(op='membership', text='其他社区')])
    assert M.implies(M.canonical(a), M.canonical(b))
    assert not M.implies(M.canonical(b), M.canonical(a))


def test_exclusion_not_discarded_in_destination():
    a = ('membership', '平湖社区'); x = ('named', '熙璟城')
    assert M.implies(('minus', a, x), a)
    assert not M.implies(a, ('minus', a, x))


def test_unknown_and_conditions_rejected_even_inside_union():
    for op in ['unknown', 'conditional', 'ref']:
        with pytest.raises(M.Unsupported):
            M.canonical(dict(op='or', children=[dict(op='named', text='甲苑'), dict(op=op, text='未确认')]))


def test_no_transitive_expansion():
    g = dict(edges=[dict(trigger_school_id='A', implied_school_id='B'),
                    dict(trigger_school_id='B', implied_school_id='C')])
    assert M.expand_original(['A'], g) == ['A', 'B']


def test_university_and_pending_schools_excluded():
    rows = [dict(school_id=s, school_name=s, ast=dict(op='membership', text='共同社区'),
                 source_text='共同社区', evidence=[], scope_status='individual_scope_requires_site_applicability') for s in 'ABCD']
    p = dict(relations=[dict(kind='university', school_ids=['A', 'B']), dict(kind='pending', school_ids=['C'])])
    g = M.build(rows, p)
    assert g['edges'] == []
    assert [r['school_id'] for r in g['inventory'] if r['admitted']] == ['D']
