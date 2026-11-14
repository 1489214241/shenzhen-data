import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_aligned_policy import walk
from shenzhen_scoped_ring import ScopedRingParser


def parser():
    return ScopedRingParser([dict(school_name='甲小学')], parse_scope)


def test_community_ring_requires_both_membership_and_whole_boundary():
    a = parser().parse('平湖社区中由铁路、富安大道、平湖大街、良白路围合的片区')
    assert a['op'] == 'and'
    assert a['children'][0] == dict(op='membership', text='平湖社区')
    assert a['children'][1]['names'] == ['铁路', '富安大道', '平湖大街', '良白路']
    assert not any(x['op'] == 'named' for x in walk(a))


def test_list_union_does_not_spread_local_community_condition():
    a = parser().parse('良安田社区以及平湖社区中由铁路、富安大道、平湖大街、良白路围合的片区')
    assert a['op'] == 'or'
    assert a['children'][0]['text'] == '良安田社区'
    assert a['children'][1]['op'] == 'and'


def test_mixed_community_list_and_ring_keep_road_operands_together():
    a = parser().parse('新联社区、嶂背社区及龙岗大道、怡翠路、龙岗河围成区域')
    assert a['op'] == 'or'
    assert [x['text'] for x in a['children'][0]['children']] == ['新联社区', '嶂背社区']
    assert a['children'][1]['names'] == ['龙岗大道', '怡翠路', '龙岗河']


def test_exclusion_stays_local_to_ring_in_mixed_list():
    a = parser().parse('甲社区及甲路、乙路、丙河围合区域（甲苑除外）')
    assert a['op'] == 'or'
    assert a['children'][0]['op'] == 'membership'
    assert a['children'][1]['op'] == 'minus'
    assert a['children'][1]['children'][0]['op'] == 'ring'


def test_exclusion_of_ring_is_not_positive_boundary_or_house_list():
    a = parser().parse('新南社区（由平龙东路、铁路、守珍街、平湖大街围合的区域除外）')
    assert a['op'] == 'minus'
    assert a['children'][1]['op'] == 'ring'


def test_conditional_and_ambiguous_road_scopes_are_not_promoted():
    for text in ['甲社区中由甲路、乙路、丙路围合区域的非深户生',
                 '甲社区（甲路、乙路、丙路围合区域）的深户生']:
        assert parser().parse(text)['op'] == 'conditional'
    for text in ['甲路以西、乙路、丙路、丁路围合区域',
                 '甲公园东南方向、乙路、丙路、丁路围合区域',
                 '甲街道与乙街道交界线、乙路、丙路、丁路围合区域',
                 '甲社区中由甲路延长线、乙路、丙路围合区域',
                 '甲社区中由甲路、乙路围合区域']:
        assert parser().ring_clause(text) is None


def test_explicit_include_keeps_community_constraint_on_ring():
    a = parser().parse('良安田社区以及平湖社区中由铁路、富安大道、平湖大街、良白路围合的片区（含甲苑）')
    assert a['op'] == 'or'
    local = a['children'][1]
    assert local['op'] == 'or'
    assert local['children'][0]['op'] == 'and'
    assert local['children'][1] == dict(op='named', text='甲苑')


def test_phase_names_and_explicit_aliases_do_not_fragment_into_fake_sites():
    a = parser().parse('佳兆业一、二、三期（即君汇公馆、金御雅园、金御佳园）以及新南社区中由平龙东路、铁路、守珍街、平湖大街围合的区域')
    names = {v['text'] for v in walk(a) if v['op'] == 'named'}
    assert names == {'佳兆业一、二、三期', '君汇公馆', '金御雅园', '金御佳园'}
    assert a['children'][1]['op'] == 'and'


def test_enumerated_exclusion_keeps_old_source_scope_identity_handling():
    a = parser().parse('甲路、乙路、丙路围成的区域（不含甲学校招生范围中的甲苑、乙苑）')
    assert a['op'] == 'minus'
    assert {v['text'] for v in walk(a['children'][1]) if v['op'] == 'named'} == {'甲苑', '乙苑'}
    assert not any(v['op'] == 'unknown' for v in walk(a))
