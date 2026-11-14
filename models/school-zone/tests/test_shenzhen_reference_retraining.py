import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_reference_retraining import fixed_source_weights,direct_positive_union


def test_added_partial_groups_do_not_dilute_source_weights_or_become_negatives():
    groups={'S':np.array([0,1]),'T':np.array([0,0,1])}
    base={'S':dict(train_ids=np.array([0,1]),labels=np.array([[1,0],[1,1]])),
          'T':dict(train_ids=np.array([0]),labels=np.array([[1,-1],[-1,-1],[-1,-1]]))}
    changed={'S':base['S'],'T':dict(train_ids=np.array([0,1,2]),labels=np.array([[1,-1],[1,-1],[-1,1]]))}
    a,_=fixed_source_weights(groups,base,base,['S'])
    b,_=fixed_source_weights(groups,base,changed,['S'])
    np.testing.assert_array_equal(a['S'],b['S'])
    assert np.isclose(b['T'][:2].sum(),a['T'][0])
    assert np.isclose(b['T'][2],a['T'][0])
    assert not (changed['T']['labels']==0).any()


def test_late_positive_supplement_preserves_original_set_without_closure():
    d={'predicted_school_ids':['a','b']}
    assert direct_positive_union(d,['b','c'])==['a','b','c']
    assert d=={'predicted_school_ids':['a','b']}
