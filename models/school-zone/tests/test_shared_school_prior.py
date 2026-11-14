import numpy as np
from zoning.shared_school_prior import build_prior, decode_prior, rank_with_count


def rule(rid, schools, triggers=(), names=(), kind='shared'):
    return dict(rule_id=rid, school_ids=list(schools), trigger_school_ids=list(triggers),
                exact_site_names=list(names), kind=kind)


def test_priors_do_not_chain_and_unknown_is_not_fact():
    p = dict(relations=[rule('ab', ['a', 'b'], ['a']), rule('bc', ['b', 'c'], ['b'])])
    site = dict(building_name='X')
    assert build_prior(p, site, ['a'])['verified_members'] == ['a', 'b']
    result = build_prior(p, site, predicted_root='a')
    assert result['verified_members'] == []
    assert result['estimated_members'] == ['a', 'b']
    assert not result['exact_count_verified']


def test_named_partial_scope_cannot_expand_whole_school_or_ambiguous_site():
    p = dict(relations=[rule('local', ['a', 'b', 'c'], names=['X'])])
    assert len(build_prior(p, dict(building_name='X'))['verified_members']) == 3
    assert not build_prior(p, dict(building_name='Y'), predicted_root='a')['estimated_members']
    site = dict(building_name='X', coordinate_group_building_names=['X', 'Y'])
    assert not build_prior(p, site)['verified_members']


def test_negative_blocks_relation_and_university_never_expands():
    p = dict(relations=[rule('ab', ['a', 'b'], ['a']),
                        rule('uni', ['a', 'c'], ['a'], kind='university')])
    result = build_prior(p, dict(building_name='X'), ['a'], ['b'])
    assert result['verified_members'] == ['a']
    assert result['blocked_rules'] == ['ab']


def test_arbitrary_cardinality_ranking_and_university_guard():
    scores, ids = np.array([.9, .8, .7, .6]), ['a', 'b', 'c', 'd']
    out, ok = rank_with_count(scores, ids, 3, university_groups=[['a', 'b']])
    assert ok and out == ['a', 'c', 'd']
    out, ok = rank_with_count(scores, ids, 3, university_groups=[['a', 'b']],
                              allowed_shared_members=['a', 'b'])
    assert ok and out == ['a', 'b', 'c']
    assert rank_with_count(scores, ids, 4, excluded=['b'])[1] is False


def test_count_only_does_not_smuggle_members_and_minimum_is_not_exact():
    prior = dict(verified_members=['a', 'c'], estimated_members=['a', 'c'])
    ids, scores = ['a', 'b', 'c'], np.array([.9, .8, .1])
    assert decode_prior(['b'], prior, scores, ids, 'count_exact')[0] == ['a', 'b']
    assert decode_prior(['b'], prior, scores, ids, 'group_exact')[0] == ['a', 'c']
    assert decode_prior(['b'], prior, scores, ids, 'minimum')[0] == ['a', 'b', 'c']
