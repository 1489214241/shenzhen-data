import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shenzhen_full_inputs
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_named_region_scope import NamedRegionParser,positive_witnesses
from shenzhen_aligned_policy import walk


def parser():return NamedRegionParser([],parse_scope)


def test_mixed_community_side_and_ring_are_two_branches():
    ast=parser().parse('宝龙社区比宝二路以西、宝荷路—积谷田路—南约河（水二支流）—沙荷路围合区域')
    assert ast['op']=='or'
    a,b=ast['children'];assert a['op']=='and' and b['op']=='ring'
    assert b['names']==['宝荷路','积谷田路','南约河(水二支流)','沙荷路']
    assert a['children'][0]['text']=='宝龙社区'


def test_qualified_road_is_not_silently_aliased():
    ast=parser().parse('宝龙社区宝坪路（比宝二路）以西片区，及宝荷路—积谷田路—南约河（水二支流）—沙荷路围合区域')
    assert ast['op']=='or'
    assert any(n['op']=='unknown' for n in walk(ast['children'][0]))


def test_included_names_prove_whole_conjunctive_region():
    ast=parser().parse('宝龙社区范围内比宝二路以西住宅（含峦山谷一期、峦山谷二期、金地名峰、宝荷欣苑等楼盘），卓弘高尔夫、仁恒峦山美地')
    witnesses=positive_witnesses(ast);assert len(witnesses)==1
    assert witnesses[0]['included_terms']==['峦山谷一期','峦山谷二期','金地名峰','宝荷欣苑']
    assert witnesses[0]['region_ast']['op']=='and'


def test_excluded_and_conditional_witnesses_never_escape():
    p=parser();a=p.parse('甲社区（含甲苑）')
    assert positive_witnesses(a)
    for op in ['minus','conditional','and']:
        # A conjunctive outer context requires additional evidence.
        assert positive_witnesses(dict(op=op,children=[a,dict(op='unknown',text='条件')]))==[]


def test_mixed_inclusion_exclusion_is_not_a_witness():
    assert not positive_witnesses(parser().parse('甲社区（含甲苑，不含乙苑）'))


def test_unknown_boundary_does_not_prove_partial_region():
    assert not positive_witnesses(parser().parse('甲社区未名道路东北片区（含甲苑）'))


def test_distinct_prefix_communities_not_promoted():
    a=parser().parse('甲社区、乙社区（含乙苑）')
    w=positive_witnesses(a);assert len(w)==1 and w[0]['region_ast']['text']=='乙社区'


def test_external_named_tail_is_not_part_of_region():
    a=parser().parse('甲社区（含甲苑），乙苑')
    assert positive_witnesses(a)[0]['included_terms']==['甲苑']
