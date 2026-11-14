import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_resident_group_scope import ResidentGroupParser
from shenzhen_aligned_policy import walk
from build_shenzhen_aligned_inputs import parse_scope


def parsers():
    inv=[dict(school_name='甲小学')]
    return ResidentGroupParser(inv,parse_scope),BoundaryConjunctionParserV2(inv,parse_scope)


def test_preserves_existing_admin_and_cardinal_intersection():
    old,new=parsers()
    for text in ['南至龙平西路、东至万科翰邻城、西和北至龙城街道行政区域界线。','北至龙岗大道、东至向银路、西和南至龙岗街道行政区域界线']:
        assert old.parse(text)==new.parse(text)


def test_unknown_sides_are_required_without_absorbing_community_alternatives():
    old,new=parsers()
    ast=new.parse('年丰社区、四方埔社区、坪东社区、龙岗大道以南与同心路以东、湖田路（黄沙河）以西')
    assert [a['op'] for a in ast['children']]==['membership','membership','membership','and']
    assert ast['children'][-1]['children'][-1]['op']=='unknown'
    ast=new.parse('龙岗大道以南，同心路以西，长美岭以北，惠盐高速以西')
    assert ast['op']=='and'


def test_fallback_diagonal_run_cannot_be_independent_admin_or_residential_name():
    ast=parsers()[1].parse('北至红棉路，东北至广达路、广宁路和广安路，东南至龙岗大道，西南至园山街道区域界线')
    assert ast['op']=='and'
    assert not any(a['op'] in ['admin','named'] for a in walk(ast))


def test_numbered_and_explicit_alternative_ranges_do_not_merge():
    old,new=parsers()
    ast=new.parse('1.龙岗大道以南，同心路以西，长美岭以北，惠盐高速以西；2.龙岗大道以北，发展路以西，兴华路以南，教育中路以东')
    assert ast['op']=='or' and [a['op'] for a in ast['children']]==['and','road']
    text='龙岗大道以南或同心路以西'
    assert old.parse(text)==new.parse(text)
