import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_adaptive_threshold import query_features, at_loss, ThresholdHead, choose


class ThresholdTests(unittest.TestCase):
    def test_author_loss_and_gradient(self):
        path = Path(__file__).resolve().parents[1] / 'outputs/shenzhen_resolution_20261112/adaptive_threshold_method_review/author_losses.py'
        spec = importlib.util.spec_from_file_location('author_atloss', path)
        author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
        z = torch.tensor([[1., -2., 3., 0.], [-2., 3., 1., -1.]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([.4, -.3], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])
        loss = at_loss(z, t, y, torch.ones(2, dtype=torch.float64)/2)
        v = torch.cat([t[:, None], z], 1)
        expected = author.ATLoss()(v, torch.cat([torch.zeros(2, 1), y], 1).to(v))
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
        a = torch.autograd.grad(loss, (z,t), retain_graph=True)
        b = torch.autograd.grad(expected, (z,t))
        for x,w in zip(a,b): torch.testing.assert_close(x,w,atol=1e-12,rtol=1e-12)

    def test_unknown_and_all_empty(self):
        z=torch.tensor([[1.,2.,3.],[1.,2.,3.]],requires_grad=True)
        t=torch.tensor([.3,.5],requires_grad=True)
        y=torch.tensor([[1,-1,0],[-1,-1,-1]])
        loss=at_loss(z,t,y,torch.ones(2))
        gz,gt=torch.autograd.grad(loss,(z,t))
        self.assertEqual(float(gz[0,1]),0.)
        self.assertTrue(bool((gz[1]==0).all()))
        self.assertEqual(float(gt[1]),0.)

    def test_invariance(self):
        torch.manual_seed(42)
        z=torch.randn(6,9,dtype=torch.float64);ax=torch.randn(6,9,33,dtype=torch.float64)
        perm=torch.randperm(9)
        f=query_features(z,ax)
        torch.testing.assert_close(f,query_features(z[:,perm],ax[:,perm]),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(f,query_features(z+7,ax),atol=1e-12,rtol=1e-12)

    def test_equal_start_and_set_contract(self):
        z=torch.tensor([[1.,2.,3.],[5.,0.,-1.]])
        f=torch.randn(2,35)
        a=ThresholdHead(np.zeros(35),np.ones(35),True,42)
        b=ThresholdHead(np.zeros(35),np.ones(35),False,42)
        torch.testing.assert_close(a(z,f),b(z,f),rtol=0,atol=0)
        np.testing.assert_array_equal(choose(np.array([[-1.,-2.,-3.],[0.,.1,.2]])),[[1,0,0],[0,1,1]])


if __name__=='__main__': unittest.main()
