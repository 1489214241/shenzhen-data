import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_resident_group_scope import ResidentGroupParser
from shenzhen_aligned_policy import walk
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_community_facts import MembershipScopeProof


def parser():return ResidentGroupParser([dict(school_name='甲小学')],parse_scope)


def test_road_side_cannot_discard_resident_group_restriction():
    for text in ['爱南路以西居民小组','南通道以北的居民小组']:
        a=parser().parse(text)
        assert a['op']=='conditional' and a['condition']=='resident_group_membership'
        assert a['children'][0]['op']=='road'


def test_official_list_preserves_community_and_independent_named_branches():
    a=parser().parse('龙东社区、龙新社区（爱南路以西居民小组）、风临四季、金汐府')
    nodes=list(walk(a))
    assert any(v['op']=='conditional' for v in nodes)
    assert {v['text'] for v in nodes if v['op']=='named'}=={'风临四季','金汐府'}
    assert {'龙东社区','龙新社区'}<={v['text'] for v in nodes if v['op']=='membership'}


def test_known_community_does_not_prove_restricted_resident_group():
    a=parser().parse('龙新社区（爱南路以西居民小组）')
    p=MembershipScopeProof([],[],[dict(district='龙岗区',street='宝龙街道',community='龙新社区')],'龙岗区')
    val,_=p.evaluate(a,dict(street='宝龙街道',community='龙新社区',site_id='q'))
    assert val==-1


def test_unrestricted_road_and_explicit_group_identity_are_preserved():
    assert parser().parse('爱南路以西')['op']=='road'
    assert parser().parse('深坑居民小组')=={'op':'membership','text':'深坑居民小组'}
