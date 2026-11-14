import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zoning.low_reference_training import choose_relations,rebuild_view,SchoolLocalModernNCA
from zoning.modernnca_multilabel import MultiLabelModernNCA,observed_bce_loss
from zoning import model as MD


class LowReferenceTests(unittest.TestCase):
    def example(self):
        torch.manual_seed(5)
        return dict(px=torch.randn(3,4,18),ax=torch.randn(3,3,13),gb=torch.tensor([0.,-.7,-.7,0.]),
            valid=torch.tensor([[1,1,0,1],[0,1,1,1],[0,0,0,0]],dtype=torch.bool),
            mem=torch.nn.functional.one_hot(torch.tensor([0,1,1,2,0,1,2]),3).float())

    def test_local_same_parameter_budget_and_backward(self):
        a=SchoolLocalModernNCA(5,7,42);b=MultiLabelModernNCA(5,7,42)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
        z=a(self.example());self.assertTrue(torch.isfinite(z).all())
        observed_bce_loss(z,torch.tensor([[1.,1.,0.],[0.,-1.,1.],[1.,0.,0.]]),torch.ones(3)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))

    def test_excluded_reference_has_no_effect(self):
        t=self.example();m=SchoolLocalModernNCA(5,7,42).eval()
        changed=dict(t,px=t['px'].clone());changed['px'][~t['valid']]=9999.
        torch.testing.assert_close(m(t),m(changed),atol=0,rtol=0)

    def test_school_permutation_equivariance(self):
        t=self.example();m=SchoolLocalModernNCA(5,7,42).eval();perm=torch.tensor([2,0,1])
        # Keep reference order, permute candidate anchor tokens and school columns.
        changed=dict(t,ax=t['ax'][:,perm],mem=torch.cat([t['mem'][:4,perm],torch.eye(3)]))
        torch.testing.assert_close(m(t)[:,perm],m(changed),atol=2e-6,rtol=2e-6)

    def fixture(self):
        from zoning.evidence import Inputs
        x=Inputs('toy',np.full((4,4,11),99.,np.float32),np.ones((4,3,5),np.float32),
            dict(distance=slice(0,5),learned_boundary=slice(5,11),crossing=slice(0,0)),
            dict(distance=slice(0,1),name=slice(1,5)),np.array([0,0,1,2]),np.array([0,1,0,2]),
            np.zeros(4,np.float32),np.zeros((4,4),bool),np.array(['a','a','b','c']),np.array([0,0,2,-1]),np.arange(3),3)
        return x

    def test_sampling_and_rebuild_hidden_information(self):
        x=self.fixture();ix=choose_relations(x,np.arange(4),7,True)
        self.assertTrue((np.bincount(x.ref_schools[ix],minlength=3)==1).all())
        np.testing.assert_array_equal(ix,choose_relations(x,np.arange(4),7,True))
        # Keep one member of original multischool site0; it must NOT become a
        # singleton label for boundary learning. Hidden site1 cannot contribute.
        seen={}
        def boundary(roads,coords,groups,known,refs):
            seen.update(known);return np.zeros((4,len(refs),6),np.float32)
        raw=np.full((4,3,4),7.,np.float32)
        with patch('zoning.low_reference_training.EV.learned_boundary',side_effect=boundary):
            v=rebuild_view(x,[0,3],np.zeros((4,2)),None,raw)
        self.assertEqual(seen,{2:[2]})
        self.assertTrue((v.pair[:,:,5:]==0).all())
        self.assertTrue((v.anchor[1,:,1:]==7).all())
        self.assertTrue((v.anchor[0,:,1:]==0).all())
        np.testing.assert_array_equal(v.y,[-1,-1,2,-1])
        self.assertTrue(v.self_mask[1,0]);self.assertFalse(v.self_mask[3].any())
        np.testing.assert_array_equal(v.group_bias,[0,0])
        self.assertEqual(tuple(v.membership.shape),(5,3))

    def test_empty_support_keeps_all_school_candidates(self):
        x=self.fixture()
        with patch('zoning.low_reference_training.EV.learned_boundary',return_value=np.zeros((4,0,6),np.float32)):
            v=rebuild_view(x,[],np.zeros((4,2)),None,np.ones((4,3,4),np.float32))
        self.assertEqual(v.pair.shape,(4,0,11));torch.testing.assert_close(v.membership,torch.eye(3))

    def test_unknown_is_not_negative(self):
        z=torch.zeros((1,3),requires_grad=True)
        observed_bce_loss(z,torch.tensor([[1.,-1.,0.]]),torch.ones(1)).backward()
        self.assertEqual(float(z.grad[0,1]),0.)


if __name__=='__main__':unittest.main()
