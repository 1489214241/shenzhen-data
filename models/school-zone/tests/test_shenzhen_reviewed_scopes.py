import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_shenzhen_reviewed_scope_inputs import exact_unconditional_names,literal_group_hit
from experiment_shenzhen_reviewed_scopes import add_reviewed_positive_facts


def test_partial_conditions_do_not_become_unconditional_names():
    ast=dict(op='or',children=[dict(op='named',text='甲苑'),
        dict(op='conditional',children=[dict(op='named',text='乙苑')]),
        dict(op='minus',children=[dict(op='named',text='丙苑'),dict(op='named',text='丁苑')])])
    assert exact_unconditional_names(ast)=={'甲苑'}
    assert literal_group_hit(dict(coordinate_group_building_names=['甲苑']),{'甲苑'})
    assert not literal_group_hit(dict(coordinate_group_building_names=['甲苑','乙苑']),{'甲苑'})
    assert not literal_group_hit(dict(coordinate_group_building_names=['甲苑一期']),{'甲苑'})


def prior():return [dict(district='区',site_id='q',building_name='苑',after=['A'])]


def fact(kind='literal_unconditional_name',sid='B',q='q'):
    return dict(site_id=q,school_id=sid,evidence_kind=kind)


def test_positive_lower_bound_keeps_prior_school_and_components_separate():
    facts=[fact(),fact('independent_admin_geography','C')]
    out=add_reviewed_positive_facts(prior(),facts,{'literal_unconditional_name'},['A','B','C'])
    assert out[0]['after']==['A','B'] and out[0]['before']==['A']
    assert not out[0]['complete_school_set'] and not out[0]['enrollment_eligibility']
    assert prior()[0]['after']==['A']


def test_empty_or_duplicate_facts_preserve_set_and_do_not_expand_relations():
    assert add_reviewed_positive_facts(prior(),[],set(),['A'])[0]['after']==['A']
    out=add_reviewed_positive_facts(prior(),[fact(),fact()],{'literal_unconditional_name'},['A','B','C'])
    assert out[0]['after']==['A','B'] and out[0]['added_school_ids']==['B']


@pytest.mark.parametrize('facts,members',[
    ([fact(q='unknown')],()),([fact(sid='unknown')],()),([fact()],('B',))])
def test_identity_mismatch_and_university_member_need_review(facts,members):
    with pytest.raises(ValueError):add_reviewed_positive_facts(prior(),facts,{'literal_unconditional_name'},['A','B'],members)
