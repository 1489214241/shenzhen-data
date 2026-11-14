"""Verify composed evidence does not erase a missing or excluded condition."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_joint_geographic_proof import JointGeographicProof, contains_witness


def make(road_values):
    def road(name, direction, alias, fact):
        v = road_values.get(name, -1)
        return v, [dict(kind='independent_road_side_estimate', policy_name=name, value=v)] if v >= 0 else []
    p = JointGeographicProof([], [], [dict(district='D', street='S', community='C社区')], 'D', [], road)
    f = dict(district='D', street='S', community='C社区', site_id='Q', source_dates=['2025-01-01'])
    return p, f


MEMBER = dict(op='membership', text='C社区')
ROAD = dict(op='road', pairs=[['R', '北']])


@pytest.mark.parametrize('side,expected', [(-1,-1),(0,0),(1,1)])
def test_community_and_road_requires_both(side, expected):
    p, f = make({'R': side})
    v, proof = p.evaluate(dict(op='and', children=[MEMBER, ROAD]), f)
    assert v == expected
    if v == 1:
        assert contains_witness(proof, 'official_community_membership')
        assert contains_witness(proof, 'independent_road_side_estimate')


def test_road_cannot_replace_unknown_community():
    p, f = make({'R': 1})
    unknown = dict(op='membership', text='Unknown社区')
    assert p.evaluate(dict(op='and', children=[unknown, ROAD]), f) == (-1, [])


def test_exclusion_requires_explicit_false_and_keeps_negative_road_witness():
    p, f = make({'R': 0})
    v, proof = p.evaluate(dict(op='minus', children=[MEMBER, ROAD]), f)
    assert v == 1 and contains_witness(proof, 'independent_road_side_estimate')
    assert proof[-1]['kind'] == 'proved_outside_excluded_region'
    assert proof[-1]['evidence'][0]['value'] == 0
    p, f = make({})
    assert p.evaluate(dict(op='minus', children=[MEMBER, ROAD]), f) == (-1, [])


def test_unknown_resident_group_and_household_remain_unknown():
    p, f = make({'R': 1})
    base = dict(op='and', children=[MEMBER, ROAD])
    exclusion = dict(op='membership', text='M居民小组')
    assert p.evaluate(dict(op='minus', children=[base, exclusion]), f) == (-1, [])
    assert p.evaluate(dict(op='conditional', children=[base], condition='hukou'), f) == (-1, [])


def test_road_conjunction_and_alternative_have_distinct_semantics():
    p, f = make({'R': 1, 'R2': 0})
    assert p.evaluate(dict(op='road', pairs=[['R','北'],['R2','东']]), f)[0] == 0
    assert p.evaluate(dict(op='or', children=[ROAD,dict(op='road',pairs=[['R2','东']])]), f)[0] == 1


def test_false_exclusion_of_an_or_needs_every_branch_false():
    p, f = make({'R': 0})
    ex = dict(op='or', children=[ROAD, dict(op='membership', text='M居民小组')])
    assert p.evaluate(dict(op='minus', children=[MEMBER, ex]), f) == (-1, [])


def test_reference_cycles_and_pending_schools_not_unrolled():
    p, f = make({'R': 1})
    p.inventory = [dict(status='individual_scope_requires_site_applicability')]
    p.asts = [dict(op='ref', school=0)]
    assert p.evaluate(p.asts[0], f) == (-1, [])
    p.inventory[0]['status'] = 'scope_pending'
    p.asts = [dict(op='and', children=[MEMBER,ROAD])]
    assert p.evaluate(dict(op='ref', school=0), f) == (-1, [])
