import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_relation_probe import pool_references


def test_masked_property_cannot_supply_evidence_and_missing_is_not_zero():
    pair=np.array([[[0.,99.],[1.,2.],[2.,3.]]],np.float32)
    out=pool_references(pair,np.array([[False,True,False]]),np.zeros(3),np.array([0,0,1]),3,0)
    for start in [0,3,6]:
        np.testing.assert_allclose(out[0,0,start:start+2],[1,2])
        assert np.isnan(out[0,1,start:start+2]).all()
        assert np.isnan(out[0,2,start:start+2]).all()
        assert out[0,1,start+2]==out[0,2,start+2]==0


def test_reference_and_school_permutations_do_not_change_meaning():
    pair=np.array([[[1.,2.],[2.,3.],[3.,4.]]],np.float32)
    valid=np.ones((1,3),bool);bias=np.zeros(3);school=np.array([0,0,1])
    old=pool_references(pair,valid,bias,school,2,0)
    order=np.array([2,0,1])
    new=pool_references(pair[:,order],valid[:,order],bias[order],1-school[order],2,0)
    np.testing.assert_allclose(old,new[:,::-1],rtol=1e-6,atol=1e-7)
