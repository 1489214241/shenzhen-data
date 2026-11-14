import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_shenzhen_source_multireferences import select_pairs


def test_only_complete_training_multis_can_add_refs():
    x=SimpleNamespace(ref_sites=np.array([0,2]),ref_schools=np.array([0,0]),groups=np.array(list('abcdefg')))
    y=dict(train_ids=np.array([0,1,2,3,4,5]),valid_ids=np.array([6]),
        labels=np.array([[1,1,0],[1,0,1],[0,1,1],[1,-1,-1],[0,1,0],[1,1,0],[-1,-1,-1]]))
    added,held=select_pairs(x,y,{5})
    assert added==[(0,1),(1,0),(1,2)]
    assert held==[dict(site=2,reason='conflicting_reference'),dict(site=5,reason='excluded_reference')]


def test_same_property_validation_is_rejected():
    import pytest
    x=SimpleNamespace(ref_sites=np.array([],int),ref_schools=np.array([],int),groups=np.array(['a','a']))
    y=dict(train_ids=np.array([0]),valid_ids=np.array([1]),labels=np.array([[1,1],[-1,-1]]))
    with pytest.raises(AssertionError):select_pairs(x,y,set())
