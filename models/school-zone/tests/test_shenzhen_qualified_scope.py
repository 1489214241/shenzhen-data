import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_aligned_policy import walk
import shenzhen_aligned_policy as P
import numpy as np
from build_shenzhen_aligned_inputs import parse_scope


def parser():return QualifiedParser([dict(school_name='甲小学')],parse_scope)


def test_bracket_condition_keeps_unconditional_prefix_and_suffix():
    a=parser().parse('甲社区，乙社区，[丙苑、丁苑]的非深户生、戊花园、己花园')
    assert a['op']=='or'
    assert [x['op'] for x in a['children']]==['or','conditional','or']
    assert a['children'][1]['condition']=='non_shenzhen_hukou'
    assert all(x['op']=='membership' for x in a['children'][0]['children'])
    assert all(x['op']=='named' for x in a['children'][2]['children'])


def test_conditioned_names_not_promoted_to_unconditional_branch():
    a=parser().parse('甲苑、乙苑，[丙苑、丁苑]的非深户生')
    assert [x['op'] for x in a['children']]==['or','conditional']
    assert [x['text'] for x in a['children'][0]['children']]==['甲苑','乙苑']


def test_unbracketed_and_nested_qualifications_stay_unknown():
    for text in ['甲苑、乙苑、丙苑的深户生','甲社区（[甲苑、乙苑]的非深户生）']:
        assert parser().parse(text)['op']=='conditional'


def test_single_qualified_range_does_not_become_whole_school_positive():
    a=parser().parse('[甲苑、乙苑]的深户生')
    assert a['op']=='conditional'


def test_executor_retains_unconditional_positives_without_household_guess(monkeypatch):
    monkeypatch.setattr(P,'Parser',QualifiedParser)
    names=['春风花园','明月花园','星河花园','云海花园']
    sites=[dict(coordinate_group_id=str(i),building_name=n,coordinate_group_building_names=[n],representative_point_wgs84=[114+i*.001,22.6]) for i,n in enumerate(names)]
    inventory=[dict(school_name='甲小学',base_scope_text='春风花园，[明月花园]的非深户生、星河花园',
        status='mixed_qualification_partial_scope',evidence=[])]
    unknown=lambda *a,**kw:np.full(4,-1,np.int8)
    ev=P.Evaluator(sites,[dict(school_id='a',school_name='甲小学')],inventory,
        dict(district='龙岗区',relations=[]),{},parse_scope,unknown,unknown,lambda xy:np.asarray(xy),None)
    result=ev.build()
    assert result['known']=={0:[0],2:[0]}
    assert not result['negative'][1,0] and not result['negative'][3,0]
