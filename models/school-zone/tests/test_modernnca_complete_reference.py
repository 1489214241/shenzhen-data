import numpy as np
import pytest
import torch
from zoning.modernnca_complete_reference import coverage_select,catalogue,complete_targets,CompleteReferenceModernNCA
from zoning.modernnca_multilabel import MultiLabelModernNCA

def test_partial_labels_cannot_be_promoted():
    with pytest.raises(ValueError):complete_targets([[1,-1]],[(0,),(0,1)])
    assert complete_targets([[1,1],[0,1]],[(0,),(0,1)]).tolist()==[1,-1]

def test_coordinate_selection_is_unique_reproducible_and_spatial():
    xy=np.array([[0,0],[1,0],[2,0],[10000,0],[10001,0]])
    a=coverage_select(xy,2,7)
    assert len(set(a))==2 and (a<3).sum()==1
    np.testing.assert_array_equal(a,coverage_select(xy,2,7))

def test_native_logits_and_masked_reference_invariance():
    torch.manual_seed(3);b,r,s=2,3,2
    t=dict(px=torch.randn(b,r,74),ax=torch.randn(b,s,33),gb=torch.zeros(r),
        valid=torch.tensor([[False,True,True],[True,False,True]]),
        mem=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[1.,0.],[0.,1.]]),
        set_mem=torch.tensor([[0.,0.,1.],[0.,0.,1.],[1.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
        geo_prior=torch.tensor([[-3.,-4.,-1.],[-2.,-3.,-1.]]),set_available=torch.ones(b,3,dtype=torch.bool))
    native=MultiLabelModernNCA(41,27,9).eval();m=CompleteReferenceModernNCA(41,27,9).eval()
    m.load_state_dict(native.state_dict(),strict=False)
    with torch.no_grad():
        school,sets=m(t);torch.testing.assert_close(school,native(t))
        changed={k:v.clone() for k,v in t.items()}
        changed['px'][~t['valid']]=1e6
        ss,zz=m(changed)
        torch.testing.assert_close(school,ss);torch.testing.assert_close(sets,zz)

def test_catalogue_keeps_shared_combination_and_distinct_schools():
    assert catalogue([(2,1),(1,2)],3)==[(0,),(1,),(2,),(1,2)]
