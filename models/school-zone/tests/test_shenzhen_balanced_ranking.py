import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_balanced_ranking import balanced_weights,masked_lsep


def test_balance_preserves_total_and_partial_target_weights():
    y={'source':np.array([[1,0],[1,0],[1,1],[1,-1]]),'target':np.array([[1,-1],[1,1]])}
    w={'source':np.array([.25,.25,.5,.2]),'target':np.array([.3,.4])}
    b,a=balanced_weights(y,w,['source'])
    np.testing.assert_allclose(b['target'],w['target']);assert b['source'][-1]==.2
    assert np.isclose(sum(x.sum() for x in b.values()),sum(x.sum() for x in w.values()))
    assert np.isclose(b['source'][:2].sum(),b['source'][2])


def test_weighted_property_counts_not_raw_row_counts():
    y={'s':np.array([[1,0],[1,0],[1,0],[1,1]])};w={'s':np.array([.5,.5,1.,1.])}
    b,a=balanced_weights(y,w,['s'])
    np.testing.assert_allclose(b['s'],[.375,.375,.75,1.5])


def test_factorized_pair_loss_matches_explicit_values_and_gradients():
    s=torch.tensor([[1.,-2.,.3,4.],[-2.,3.,1.,0.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1.,0.,1.,-1.],[0.,1.,1.,0.]],dtype=torch.float64);w=torch.tensor([.3,.7])
    actual=masked_lsep(s,y,w);actual.backward();grad=s.grad.clone();s.grad=None
    expected=sum(w[i]*torch.log1p(torch.exp(s[i,y[i]==0][:,None]-s[i,y[i]==1][None]).sum()) for i in range(2))
    expected.backward();torch.testing.assert_close(actual,expected);torch.testing.assert_close(grad,s.grad)
    assert grad[0,3]==0


def test_no_guessed_negative_for_partial_labels_and_extreme_logits_finite():
    s=torch.tensor([[1000.,-1000.,0.],[.1,.2,.3]],requires_grad=True)
    y=torch.tensor([[0.,1.,-1.],[1.,-1.,-1.]])
    loss=masked_lsep(s,y,torch.ones(2));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    assert (s.grad[1]==0).all() and s.grad[0,2]==0


def test_loss_falls_when_worst_positive_outranks_negative():
    y=torch.tensor([[1.,1.,0.]])
    assert masked_lsep(torch.tensor([[2.,2.,0.]]),y,torch.ones(1))<masked_lsep(torch.tensor([[2.,-2.,0.]]),y,torch.ones(1))
