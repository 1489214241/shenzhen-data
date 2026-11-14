"""Failure-oriented tests for independent community-to-school source proofs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_community_facts import MembershipScopeProof


def make_proof(extra=()):
    profiles=[dict(district='D',street='S',community='A社区')]+list(extra)
    fact=dict(district='D',street='S',community='A社区',site_id='Q')
    p=MembershipScopeProof([],[],profiles,'D')
    return p,fact


def test_direct_membership_and_unobserved_is_unknown():
    p,f=make_proof()
    assert p.evaluate(dict(op='membership',text='A社区'),f)[0]==1
    assert p.evaluate(dict(op='membership',text='B社区'),f)[0]==-1


def test_same_named_communities_in_other_streets_are_ambiguous():
    p,f=make_proof([dict(district='D',street='Other',community='A社区')])
    assert p.evaluate(dict(op='membership',text='A社区'),f)[0]==-1


def test_no_parent_to_subcommittee_or_suffix_collapse():
    p,f=make_proof()
    for name in ['A社区居委会','A社区某居委会','A']:
        assert p.evaluate(dict(op='membership',text=name),f)[0]==-1


def test_road_condition_cannot_be_erased_by_positive_membership():
    p,f=make_proof()
    a=dict(op='and',children=[dict(op='membership',text='A社区'),dict(op='road',pairs=[['R','北']])])
    assert p.evaluate(a,f)==(-1,[])
    a['op']='or'
    assert p.evaluate(a,f)[0]==1


def test_unknown_exclusion_blocks_positive_base():
    p,f=make_proof()
    a=dict(op='minus',children=[dict(op='membership',text='A社区'),dict(op='named',text='某小区')])
    assert p.evaluate(a,f)==(-1,[])


def test_known_exclusion_rejects_and_household_condition_stays_unknown():
    p,f=make_proof()
    member=dict(op='membership',text='A社区')
    assert p.evaluate(dict(op='minus',children=[member,member]),f)==(0,[])
    assert p.evaluate(dict(op='conditional',children=[member],condition='hukou'),f)==(-1,[])


def test_cyclic_explicit_school_scope_reference_is_unknown():
    p,f=make_proof()
    p.inventory=[dict(status='individual_scope_requires_site_applicability')]
    p.asts=[dict(op='ref',school=0)]
    assert p.evaluate(p.asts[0],f)==(-1,[])


def test_duplicate_profile_versions_are_not_different_community_identities():
    p,f=make_proof([dict(district='D',street='S',community='A社区')])
    assert p.evaluate(dict(op='membership',text='A社区'),f)[0]==1


def test_profile_street_identity_does_not_expand_to_other_streets():
    p,f=make_proof()
    assert p.evaluate(dict(op='admin',text='S'),f)[0]==1
    assert p.evaluate(dict(op='admin',text='Other'),f)[0]==-1
