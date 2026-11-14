import sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from shenzhen_composed_features import composed_scope,encode
from zoning import rules


def fixture(values):
    values=np.asarray(values,np.int8);n,k=values.shape
    O=dict(n=n,school_n=1,name_known={},generic=[False]*n,
        clauses=[dict(school=0,kind='area',sides=[dict(column=j,direction='',status='ready') for j in range(k)],
            admitted_by_parse=True,parse_complete=True,exclusions=[],admin=None)])
    A=dict(side_values=values);S=deepcopy(rules.PRESETS['unified_20261106']['芙蓉区'])
    return O,A,S


def test_noncardinal_composition_can_carry_information_without_fabricated_directions():
    O,A,S=fixture([[1,1],[1,-1],[0,1]])
    states,_=composed_scope(O,A,S)
    np.testing.assert_array_equal(states[:,0],[1,-1,0])
    _,directions,_=rules.school(0,O['clauses'],O,A,S)
    assert (directions==-1).all()


def test_nonarea_alternative_does_not_turn_missing_name_into_negative():
    O,A,S=fixture([[0],[1]])
    O['clauses'].append(dict(school=0,kind='named'))
    states,_=composed_scope(O,A,S)
    np.testing.assert_array_equal(states[:,0],[-1,1])


def test_native_conflict_block_is_not_silently_disabled():
    O,A,S=fixture([[1],[1],[1]])
    O['name_known']={0:[1],1:[1]}
    states,_=composed_scope(O,A,S)
    assert (states==-1).all()  # Native K8 holds this whole clause.
    O,A,S=fixture([[0],[1]])
    O['name_known']={0:[0]}
    states,audit=composed_scope(O,A,S)
    assert (states==-1).all() and audit[0]['blocked']


def test_unknown_is_distinct_from_negative_and_positive():
    np.testing.assert_array_equal(encode(np.array([-1,0,1])),np.eye(3))
