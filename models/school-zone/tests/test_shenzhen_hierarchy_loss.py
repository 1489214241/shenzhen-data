import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_hierarchy_loss import masked_max_constraint_loss as loss, relation_matrix
from shenzhen_partial_model import masked_bce


def test_identity_matches_masked_bce_values_and_gradients():
    torch.manual_seed(23)
    z = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(-1, 2, z.shape).double(); w = torch.rand(7).double()
    a = masked_bce(z, y, w); b = loss(z, y, w, torch.eye(5, dtype=torch.bool))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(a, z)[0], torch.autograd.grad(b, z)[0], rtol=0, atol=0)


def test_paper_example_wrong_child_decreases_positive_parent_increases():
    z = torch.logit(torch.tensor([[.3, .1]], dtype=torch.float64)).requires_grad_()
    r = torch.tensor([[True, False], [True, True]])
    grad = torch.autograd.grad(loss(z, torch.tensor([[0., 1.]]), torch.ones(1), r), z)[0]
    assert grad[0, 0] > 0 and grad[0, 1] < 0


def test_unknown_child_cannot_explain_observed_parent():
    z = torch.tensor([[9., -2.]], requires_grad=True)
    r = torch.tensor([[True, False], [True, True]])
    grad = torch.autograd.grad(loss(z, torch.tensor([[-1., 1.]]), torch.ones(1), r), z)[0]
    assert grad[0, 0] == 0 and grad[0, 1] < 0


def test_positive_only_cannot_supply_downward_logit_gradient():
    z = torch.tensor([[2., -2.]], requires_grad=True)
    r = torch.tensor([[True, False], [True, True]])
    grad = torch.autograd.grad(loss(z, torch.ones_like(z), torch.ones(1), r), z)[0]
    assert (grad <= 0).all() and grad[0, 0] < 0 and grad[0, 1] == 0


def test_observed_negative_parent_can_penalize_unknown_child():
    z = torch.tensor([[2., -2.]], requires_grad=True)
    r = torch.tensor([[True, False], [True, True]])
    grad = torch.autograd.grad(loss(z, torch.tensor([[-1., 0.]]), torch.ones(1), r), z)[0]
    assert grad[0, 0] > 0 and grad[0, 1] == 0


def test_all_unknown_is_finite_zero_with_zero_gradient():
    z = torch.tensor([[1000., -1000.]], requires_grad=True)
    a = loss(z, -torch.ones_like(z), torch.ones(1), torch.ones(2, 2, dtype=torch.bool))
    assert a == 0 and torch.isfinite(a)
    assert (torch.autograd.grad(a, z)[0] == 0).all()


def test_permutation_and_equivalence_cycle():
    z = torch.tensor([[3., 1., -2.]], requires_grad=True)
    y = torch.tensor([[1., 1., -1.]]); w = torch.ones(1)
    r = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    p = [2, 0, 1]
    torch.testing.assert_close(loss(z, y, w, r), loss(z[:, p], y[:, p], w, r[p][:, p]))


def test_relation_direction_and_missing_self_rejected():
    r = relation_matrix(['a', 'b'], [dict(trigger_school_id='a', implied_school_id='b')])
    assert r.tolist() == [[True, False], [True, True]]
    with pytest.raises(ValueError, match='itself'):
        loss(torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1), ~r)
