import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_scoped_parenthesis import ScopedParenthesisParser
from shenzhen_named_region_scope import NamedRegionParser
from shenzhen_aligned_policy import walk
from build_shenzhen_aligned_inputs import parse_scope


def parser():return ScopedParenthesisParser([],parse_scope)


def test_local_exclusion_does_not_swallow_later_projects():
    a=parser().parse('甲苑、慢城（包括一、二、三期，不含来座山）、华浩源（景园）、万隆苑')
    assert a['op']=='or' and len(a['children'])==4
    assert a['children'][1]['op']=='minus'
    assert a['children'][1]['children']==[{'op':'named','text':'慢城一、二、三期'},{'op':'named','text':'来座山'}]
    assert a['children'][2]['op']=='named' and a['children'][3]['text']=='万隆苑'


def test_unsupported_local_condition_stays_unknown_but_siblings_survive():
    a=parser().parse('甲苑、乙苑（包括部分住宅，不含丙楼）、丁苑')
    assert a['op']=='or' and a['children'][1]['op']=='unknown'
    assert a['children'][0]['text']=='甲苑' and a['children'][2]['text']=='丁苑'


def test_geographic_conjunction_not_split_into_union():
    t='甲路以东、乙路以西、丙路以北（不含丁苑）'
    assert parser().parse(t)==NamedRegionParser([],parse_scope).parse(t)


def test_unknown_mixed_road_and_project_not_promoted():
    t='甲社区、乙路以东（不含丁苑）、丙苑'
    assert parser().parse(t)==NamedRegionParser([],parse_scope).parse(t)


def test_household_condition_kept():
    t='甲苑、慢城（包括一、二、三期，不含来座山），仅限深户生'
    assert parser().parse(t)['op']=='conditional'


def test_malformed_parenthesis_not_a_residential_name():
    for t in ['慢城（包括一期','来座山）']:
        assert parser().parse(t)['op']=='unknown'


def test_shared_phase_suffix_and_alias_preserved():
    for t in ['甲苑一、二、三期','华浩源（景园）']:
        assert parser().parse(t)==NamedRegionParser([],parse_scope).parse(t)


def test_phase_exclusion_stays_under_outer_exclusion():
    a=parser().parse('甲社区（不含慢城（包括一、二、三期，不含来座山））')
    assert a['op']=='minus' and a['children'][1]['op']=='minus'
    assert all(')' not in n.get('text','') or n['op']!='named' for n in walk(a))
