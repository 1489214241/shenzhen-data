import unittest
import torch
from zoning.partial_set_learning import compatible_sets,marginal_set_loss


class PartialSetTests(unittest.TestCase):
    def test_unknown_fourth_school_does_not_become_negative(self):
        labels=torch.tensor([[1,1,1,-1]])
        mem=torch.tensor([[1,1,0,0],[1,1,1,0],[1,1,1,1]])
        allowed=compatible_sets(labels,mem)
        self.assertEqual(allowed.tolist(),[[False,True,True]])
        z=torch.zeros(1,3,requires_grad=True);marginal_set_loss(z,allowed,torch.ones(1)).backward()
        self.assertGreater(z.grad[0,0],0);self.assertLess(z.grad[0,1],0)
        torch.testing.assert_close(z.grad[0,1],z.grad[0,2])

    def test_all_unknown_has_zero_loss_and_gradient(self):
        allowed=compatible_sets(torch.full((1,2),-1),torch.tensor([[1,0],[0,1],[1,1]]))
        z=torch.randn(1,3,requires_grad=True);loss=marginal_set_loss(z,allowed,torch.ones(1));loss.backward()
        self.assertEqual(float(loss),0.);torch.testing.assert_close(z.grad,torch.zeros_like(z))

    def test_complete_label_reduces_to_exact_set_cross_entropy(self):
        mem=torch.tensor([[1,0],[0,1],[1,1]]);z=torch.tensor([[.2,.1,.5]])
        mask=compatible_sets(torch.tensor([[1,1]]),mem)
        torch.testing.assert_close(marginal_set_loss(z,mask,torch.ones(1)),torch.nn.functional.cross_entropy(z,torch.tensor([2])))

    def test_explicit_negative_is_distinct_from_unknown(self):
        mem=torch.tensor([[1,0],[1,1]])
        self.assertEqual(compatible_sets(torch.tensor([[1,0]]),mem).tolist(),[[True,False]])
        self.assertEqual(compatible_sets(torch.tensor([[1,-1]]),mem).tolist(),[[True,True]])


if __name__=='__main__':unittest.main()
