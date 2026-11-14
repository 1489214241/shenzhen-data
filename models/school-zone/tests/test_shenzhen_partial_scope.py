import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_partial_scope import partial, sat_proof, build, implies


def test_known_union_branch_survives_unknown():
    a = dict(op='named', text='甲苑')
    b = dict(op='or', children=[a, dict(op='unknown', text='未定位边界')])
    x, y = partial(a, 'A'), partial(b, 'B')
    assert implies(x, y)
    assert sat_proof(x, y)['counterexample_unsatisfiable']
    assert not implies(y, x)


def test_unknown_exclusion_not_dropped():
    a = dict(op='named', text='甲苑')
    b = dict(op='minus', children=[a, dict(op='unknown', text='排除区')])
    assert not implies(partial(a, 'A'), partial(b, 'B'))
    assert implies(partial(b, 'B'), partial(a, 'A'))


def test_unknown_intersection_not_dropped():
    a = dict(op='named', text='甲苑')
    b = dict(op='and', children=[a, dict(op='conditional', text='居民小组')])
    assert not implies(partial(a, 'A'), partial(b, 'B'))
    assert implies(partial(b, 'B'), partial(a, 'A'))


def test_opaque_identity_never_matches_across_schools_or_paths():
    u = dict(op='unknown', text='同一未核文字')
    assert partial(u, 'A') != partial(u, 'B')
    assert partial(u, 'A', (0,)) != partial(u, 'A', (1,))


def test_unparsed_parent_preserves_whole_subtree():
    u = dict(op='conditional', text='未确认', children=[dict(op='named', text='甲苑')])
    assert partial(u, 'A')[0] == 'opaque'


def test_restricted_scope_cannot_gain_edges():
    rows = [dict(school_id=s, school_name=s, ast=dict(op='named', text='甲苑'), evidence=[],
                 scope_status='individual_scope_requires_site_applicability') for s in 'ABC']
    old = dict(edges=[], inventory=[dict(school_id=s, skip_reason='university_eligibility_or_pending_member' if s == 'A' else None) for s in 'ABC'])
    g = build(rows, old)
    assert {(e['trigger_school_id'], e['implied_school_id']) for e in g['added']} == {('B','C'), ('C','B')}
