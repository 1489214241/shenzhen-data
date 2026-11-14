import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_boundary_conjunction import BoundaryConjunctionParser
from shenzhen_aligned_policy import walk
from build_shenzhen_aligned_inputs import parse_scope
from zoning.rules import AND
import numpy as np


def parse(text):return BoundaryConjunctionParser([dict(school_name='甲小学')],parse_scope).parse(text)


def test_missing_mountain_side_stays_a_required_operand():
    ast=parse('龙岗大道以南，同心路以西，长美岭以北，惠盐高速以西')
    assert ast['op']=='and'
    assert [a['op'] for a in ast['children']]==['road','road','unknown','road']
    assert AND([np.array([1]),np.array([1]),np.array([-1]),np.array([1])],1)[0]==-1


def test_numbered_scopes_remain_alternatives():
    ast=parse('1.龙岗大道以南，同心路以西，长美岭以北，惠盐高速以西；2.龙岗大道以北，发展路以西，兴华路以南，教育中路以东；3.龙岗大道以北，香元排路以西，兴华路以南，新园路往龙岗大道方向的延长线以东。')
    assert ast['op']=='or'
    assert [a['op'] for a in ast['children']]==['and','road','and']


def test_community_list_does_not_become_conjunction_with_road_group():
    ast=parse('年丰社区、四方埔社区、坪东社区、龙岗大道以南与同心路以东、湖田路（黄沙河）以西')
    assert ast['op']=='or'
    assert [a['op'] for a in ast['children']]==['membership','membership','membership','and']
    assert ast['children'][-1]['children'][-1]['op']=='unknown'


def test_diagonal_boundary_does_not_create_whole_admin_membership_or_house_name():
    ast=parse('北至红棉路，东北至广达路、广宁路和广安路，东南至龙岗大道，西南至园山街道区域界线')
    assert ast['op']=='and'
    assert not any(a['op'] in ['admin','named'] for a in walk(ast))
    assert sum(a['op']=='unknown' for a in walk(ast))==4


def test_complete_boundary_and_resident_qualification_are_retained():
    assert parse('龙岗大道以北，发展路以西，兴华路以南，教育中路以东')['op']=='road'
    ast=parse('龙新社区（爱南路以西居民小组）、风临四季、金汐府')
    assert any(a['op']=='conditional' for a in walk(ast))
    assert {a['text'] for a in walk(ast) if a['op']=='named'}=={'风临四季','金汐府'}
