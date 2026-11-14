import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from shenzhen_schoolwise_reference_mass import schoolwise_bias


def test_new_school_relation_preserves_original_school_evidence_mass():
    old=schoolwise_bias([10,20],[0,1],[0,0])
    new=schoolwise_bias([10,20],[0,0,1],[0,1,0])
    np.testing.assert_array_equal(old,new[[0,2]])
    assert np.exp(new[1])==1


def test_duplicate_coordinates_of_same_property_are_not_extra_evidence():
    bias=schoolwise_bias([10,10,20],[0,1,0,2],[0,0,1,0])
    np.testing.assert_allclose(np.exp(bias[:2]).sum(),1)
    np.testing.assert_allclose(np.exp(bias[2:]),[1,1])


def test_preserves_old_weights_for_exclusively_single_school_properties():
    groups=np.array([10,10,20,30]);ri=np.array([0,1,2,3]);rs=np.array([0,0,1,1])
    _,inverse,count=np.unique(groups[ri],return_inverse=True,return_counts=True)
    np.testing.assert_array_equal(schoolwise_bias(groups,ri,rs),-np.log(count[inverse]).astype(np.float32))


def test_reference_order_invariance_and_empty_bank():
    ri=np.array([0,1,0,2]);rs=np.array([0,0,1,0]);perm=np.array([3,0,2,1])
    expected=schoolwise_bias([10,10,20],ri,rs)
    np.testing.assert_array_equal(schoolwise_bias([10,10,20],ri[perm],rs[perm]),expected[perm])
    assert schoolwise_bias([1],[],[]).shape==(0,)
