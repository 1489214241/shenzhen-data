"""Check the failure boundaries of administrative identity substitution."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_administrative_identity import ReviewedAdministrativeScopeProof


def inputs():
    profiles = [dict(district='D', street='S', community='A社区')]
    mapping = dict(district='D', street='S', community='A社区', scope_term='A社区居委会',
        relation='community_and_sole_committee_coextensive', decision='accepted_dated_identity',
        version_not_before='2023-05-24', evidence=[dict(publication_date='2023-05-24',
            quote='A社区下辖A居委会，属一站一居。', url='https://example.gov.cn/notice', sha256='test')])
    fact = dict(district='D', street='S', community='A社区', site_id='site', source_dates=['2025-01-01'])
    return profiles, mapping, fact


def proof():
    profiles, mapping, fact = inputs()
    return ReviewedAdministrativeScopeProof([], [], profiles, 'D', [mapping]), fact


def test_identity_proof_is_dated_and_traceable():
    p, f = proof()
    value, evidence = p.evaluate(dict(op='membership', text='A社区居委会'), f)
    assert value == 1
    assert {w['kind'] for w in evidence} == {'official_community_membership', 'reviewed_administrative_identity'}
    assert evidence[1]['evidence'][0]['publication_date'] == '2023-05-24'


@pytest.mark.parametrize('dates', [[], ['2022-01-01'], ['2026-07-01']])
def test_old_or_late_or_undated_property_does_not_cross_version(dates):
    p, f = proof()
    f['source_dates'] = dates
    assert p.evaluate(dict(op='membership', text='A社区居委会'), f) == (-1, [])


@pytest.mark.parametrize('field,value', [('district', 'Other'), ('street', 'Other'), ('community', 'B社区')])
def test_same_name_mapping_does_not_cross_district_street_or_community(field, value):
    p, f = proof()
    f[field] = value
    assert p.evaluate(dict(op='membership', text='A社区居委会'), f) == (-1, [])


def test_no_generic_suffix_stripping_or_inference_to_subcommittee():
    p, f = proof()
    for name in ['A', 'A居委会', 'A社区第一居委会', 'A居民小组']:
        assert p.evaluate(dict(op='membership', text=name), f) == (-1, [])


def test_substitution_preserves_road_exclusion_and_household_conditions():
    p, f = proof()
    member = dict(op='membership', text='A社区居委会')
    road = dict(op='road', pairs=[['R', '北']])
    assert p.evaluate(dict(op='and', children=[member, road]), f) == (-1, [])
    assert p.evaluate(dict(op='minus', children=[member, road]), f) == (-1, [])
    assert p.evaluate(dict(op='conditional', children=[member], condition='hukou'), f) == (-1, [])
    assert p.evaluate(dict(op='minus', children=[member, member]), f) == (0, [])


@pytest.mark.parametrize('change', [
    {'relation': 'parent_contains_child'}, {'decision': 'pending'}, {'evidence': []},
    {'version_not_before': '2026-07-01'},
])
def test_unproved_or_late_identity_is_rejected(change):
    profiles, mapping, _ = inputs()
    mapping.update(change)
    with pytest.raises(ValueError):
        ReviewedAdministrativeScopeProof([], [], profiles, 'D', [mapping])


def test_ambiguity_and_contradictory_literal_identity_are_rejected():
    profiles, mapping, _ = inputs()
    with pytest.raises(ValueError):
        ReviewedAdministrativeScopeProof([], [], profiles, 'D', [mapping, copy.deepcopy(mapping)])
    profiles.append(dict(district='D', street='Other', community='A社区'))
    with pytest.raises(ValueError):
        ReviewedAdministrativeScopeProof([], [], profiles, 'D', [mapping])


def test_no_school_transitive_expansion_is_introduced():
    p, f = proof()
    p.inventory = [dict(status='individual_scope_requires_site_applicability')]
    p.asts = [dict(op='ref', school=0)]
    assert p.evaluate(p.asts[0], f) == (-1, [])
