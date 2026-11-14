import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.modernnca_multilabel import MultiLabelModernNCA,observed_bce_loss,categorical_positive_loss


class MultiLabelTests(unittest.TestCase):
    def test_native_probabilities_and_mask(self):
        torch.manual_seed(12);b,r,s,np_,na=3,4,3,5,7
        t=dict(px=torch.randn(b,r,np_+na+6),ax=torch.randn(b,s,na+6),gb=torch.tensor([0.,-.7,-.7,0.]),
               valid=torch.tensor([[1,1,0,1],[0,1,1,1],[1,0,1,1]],dtype=torch.bool),
               mem=torch.nn.functional.one_hot(torch.tensor([0,1,1,2,0,1,2]),s).float())
        native=LayerNormModernNCA(np_,na,42).eval();multi=MultiLabelModernNCA(np_,na,42).eval()
        for k,v in native.state_dict().items():self.assertTrue(torch.equal(v,multi.state_dict()[k]))
        with torch.no_grad():
            p=native(t);q=multi(t).softmax(1)
            torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-6)
            altered=dict(t,px=t['px'].clone());altered['px'][~t['valid']]=9999
            torch.testing.assert_close(multi(t),multi(altered),atol=0,rtol=0)

    def test_unknown_has_no_binary_gradient(self):
        z=torch.tensor([[.2,.3,.4]],requires_grad=True);y=torch.tensor([[1.,-1.,0.]])
        observed_bce_loss(z,y,torch.ones(1)).backward()
        self.assertLess(z.grad[0,0],0);self.assertEqual(z.grad[0,1],0);self.assertGreater(z.grad[0,2],0)

    def test_both_positives_are_trained(self):
        z=torch.zeros(1,3,requires_grad=True)
        observed_bce_loss(z,torch.tensor([[1.,1.,0.]]),torch.ones(1)).backward()
        self.assertTrue((z.grad[0,:2]<0).all());self.assertGreater(z.grad[0,2],0)

    def test_singleton_control_matches_native_nll(self):
        z=torch.tensor([[.1,.4,.2]])
        a=categorical_positive_loss(z,torch.tensor([[0.,1.,0.]]),torch.ones(1))
        torch.testing.assert_close(a,-z.log_softmax(1)[0,1])


if __name__=='__main__':unittest.main()
