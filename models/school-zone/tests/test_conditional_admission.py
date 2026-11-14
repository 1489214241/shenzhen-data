import numpy as np
import torch
from zoning.conditional_admission import Admission,partial_loss,combine

def test_unknown_gradient_and_negative():
    for kind in ['bce','zlpr']:
        z=torch.tensor([[2.,1.,-1.]],requires_grad=True)
        partial_loss(z,torch.tensor([[1.,-1.,0.]]),torch.ones(1),kind).backward()
        assert z.grad[0,0]<0 and z.grad[0,1]==0 and z.grad[0,2]>0

def test_all_unknown_zero():
    for kind in ['bce','zlpr']:
        z=torch.randn(2,3,requires_grad=True)
        loss=partial_loss(z,torch.full_like(z,-1),torch.ones(2),kind)
        loss.backward();assert loss==0 and not z.grad.any()

def test_top_preservation_and_arbitrary_subset():
    p=np.array([[.6,.3,.1]])
    q=combine(p,np.array([[.01,.1,.8]]))
    assert np.array_equal(q>=.5,[[True,False,True]])

def test_shared_admission_permutation():
    model=Admission(5,42);x=torch.randn(4,3,5)
    assert torch.equal(model(x[:,[2,0,1]]),model(x)[:,[2,0,1]])
