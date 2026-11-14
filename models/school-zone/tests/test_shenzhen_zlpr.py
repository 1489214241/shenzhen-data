import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from shenzhen_zlpr import zlpr_rows,source_loss
from shenzhen_partial_model import masked_bce
from shenzhen_balanced_ranking import masked_lsep


def test_equation_and_finite_extreme_gradient():
    s=torch.tensor([[.3,-.7,1.2],[-1000.,1000.,1000.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1,0,1],[1,0,-1]])
    v=zlpr_rows(s,y)
    expected=torch.log1p(torch.exp(-s[0,0])+torch.exp(-s[0,2]))+torch.log1p(torch.exp(s[0,1]))
    torch.testing.assert_close(v[0],expected)
    v.sum().backward();assert torch.isfinite(v).all() and torch.isfinite(s.grad).all()
    assert s.grad[1,2]==0


def test_unknowns_and_empty_observations():
    s=torch.tensor([[.2,.8,-.5],[1.,-1.,2.]],requires_grad=True)
    y=torch.tensor([[1,-1,-1],[-1,-1,-1]])
    v=zlpr_rows(s,y)
    torch.testing.assert_close(v[0],torch.nn.functional.softplus(-s[0,0]))
    assert v[1]==0
    v.sum().backward();assert (s.grad[y==-1]==0).all()


def test_partial_source_and_target_losses_preserved():
    s=torch.tensor([[.2,.8,-.5],[.4,-.3,.7],[1.,-2.,-1.]],requires_grad=True)
    y=torch.tensor([[1,0,1],[1,-1,0],[1,-1,-1]],dtype=s.dtype)
    w=torch.tensor([.2,.3,.5])
    expected=zlpr_rows(s[:1],y[:1])[0]*w[0]+masked_bce(s[1:],y[1:],w[1:])+masked_lsep(s[1:],y[1:],w[1:])
    torch.testing.assert_close(source_loss(s,y,w),expected)
    all_partial=y[1:];torch.testing.assert_close(source_loss(s[1:],all_partial,w[1:]),masked_bce(s[1:],all_partial,w[1:])+masked_lsep(s[1:],all_partial,w[1:]))
    torch.testing.assert_close(source_loss(s[:1],y[:1],w[:1]),(zlpr_rows(s[:1],y[:1])*w[:1]).sum())


def test_permutation_and_bounded_group_gradient():
    s=torch.tensor([[.3,-.7,-.7,-.7]],dtype=torch.float64,requires_grad=True);y=torch.tensor([[1,0,0,0]])
    loss=zlpr_rows(s,y).sum();loss.backward()
    assert s.grad[y==0].sum()<1 and -s.grad[y==1].sum()<1
    order=[3,0,2,1];torch.testing.assert_close(zlpr_rows(s[:,order],y[:,order]),zlpr_rows(s,y))
