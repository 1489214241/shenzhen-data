import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_explicit_scope_facts import evaluate,prove


def name(t):return dict(op='named',text=t)
def geo():return dict(op='unknown',text='unresolved boundary')
def expr(op,*args):return dict(op=op,children=list(args))


def test_whole_scope_explicit_exclusion_is_negative_without_base_geometry():
    a=expr('minus',geo(),name('甲苑'));m={'甲苑':np.array([1,-1],np.int8)}
    assert evaluate(a,m,2).tolist()==[0,-1]
    assert prove(a,['甲苑'],0)['valid']


def test_excluding_one_branch_is_not_excluding_whole_school():
    a=expr('or',expr('minus',geo(),name('甲苑')),geo())
    assert evaluate(a,{'甲苑':np.array([1],np.int8)},1).tolist()==[-1]
    with pytest.raises(AssertionError):prove(a,['甲苑'],0)


def test_all_alternatives_must_be_excluded():
    a=expr('or',expr('minus',geo(),name('甲苑')),expr('minus',geo(),name('乙名')))
    m={t:np.array([1],np.int8) for t in ['甲苑','乙名']}
    assert evaluate(a,m,1).tolist()==[0] and prove(a,list(m),0)['valid']


def test_unknown_is_not_a_negative_or_a_name_mismatch():
    a=expr('or',name('甲苑'),name('乙苑'));m={t:np.array([-1],np.int8) for t in ['甲苑','乙苑']}
    assert evaluate(a,m,1).tolist()==[-1]
    with pytest.raises(AssertionError):prove(a,[],0)


def test_positive_sibling_survives_local_exclusion():
    a=expr('or',expr('minus',name('甲苑'),name('乙苑')),name('丙苑'))
    m={t:np.array([1 if t=='丙苑' else -1],np.int8) for t in ['甲苑','乙苑','丙苑']}
    assert evaluate(a,m,1).tolist()==[1] and prove(a,['丙苑'],1)['valid']


def test_unknown_conditions_block_named_promotions():
    a=expr('and',name('甲苑'),dict(op='conditional',text='户籍待核'))
    assert evaluate(a,{'甲苑':np.array([1],np.int8)},1).tolist()==[-1]
    with pytest.raises(AssertionError):prove(a,['甲苑'],1)
