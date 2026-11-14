import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_set_evidence import school_pool, SetEvidenceRetrieval
from shenzhen_partial_model import BernoulliRetrieval


def fixture():
    torch.manual_seed(13)
    pair = torch.rand(2, 4, 3)
    anchor = torch.rand(2, 3, 2)
    rs = torch.tensor([0, 1, 0, 2])
    mem = torch.cat([torch.nn.functional.one_hot(rs, 3), torch.eye(3)], 0).float()
    return dict(pair=pair, anchor=anchor, px=torch.rand(2, 4, 11),
                ax=torch.rand(2, 3, 8), gb=torch.zeros(4),
                valid=torch.tensor([[True, False, True, True], [False] * 4]),
                mem=mem, meta=(0, [], 3)), SimpleNamespace(
                    p=torch.zeros(3), a=torch.zeros(2), offset=torch.tensor(-2.))


def test_initial_output_equals_parent_and_no_support_is_finite():
    t, start = fixture()
    parent = BernoulliRetrieval(3, 2, start, 23)
    candidate = SetEvidenceRetrieval(3, 2, start, 23)
    candidate.load_parent(parent.state_dict())
    torch.testing.assert_close(parent(t), candidate(t), rtol=0, atol=0)
    assert torch.isfinite(candidate(t)).all()


def test_pool_reference_and_school_permutations():
    e = torch.randn(2, 4, 7, dtype=torch.float64)
    w = torch.rand(2, 4, dtype=torch.float64)
    m = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    ref = [3, 1, 0, 2]; school = [2, 0, 1]
    torch.testing.assert_close(school_pool(e, w, m)[:, school],
                              school_pool(e[:, ref], w[:, ref], m[ref][:, school]))
    assert (school_pool(e, w, m)[:, 2] == 0).all()


def test_masked_reference_has_zero_gradient_and_no_effect():
    e = torch.randn(2, 4, 3, requires_grad=True)
    w = torch.tensor([[1., 0., 2., 0.], [0., 0., 0., 0.]])
    m = torch.eye(4)
    out = school_pool(e, w, m)
    out.sum().backward()
    assert (e.grad[w == 0] == 0).all()
    changed = e.detach().clone(); changed[w == 0] = 1e8
    torch.testing.assert_close(out, school_pool(changed, w, m), rtol=0, atol=0)


def test_duplicate_property_with_divided_mass_is_invariant():
    e = torch.randn(2, 2, 4, dtype=torch.float64)
    w = torch.rand(2, 2, dtype=torch.float64)
    m = torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(school_pool(e, w, m), school_pool(
        e.repeat_interleave(2, 1), w.repeat_interleave(2, 1) / 2, m.repeat_interleave(2, 0)))


def test_learned_residual_is_reference_invariant_and_school_equivariant():
    t, start = fixture(); model = SetEvidenceRetrieval(3, 2, start, 23)
    with torch.no_grad():
        model.set_rho[-1].weight.fill_(.2)
    ref = torch.tensor([2, 0, 3, 1]); school = torch.tensor([2, 0, 1])
    other = dict(t)
    for k in ['pair', 'px', 'valid']: other[k] = t[k][:, ref]
    other['gb'] = t['gb'][ref]
    for k in ['anchor', 'ax']: other[k] = t[k][:, school]
    other['mem'] = torch.cat([t['mem'][:4][ref][:, school], torch.eye(3)], 0)
    torch.testing.assert_close(model(t)[:, school], model(other), rtol=1e-6, atol=1e-6)
    model(t).sum().backward()
    assert model.set_phi[0].weight.grad.abs().sum() > 0


def test_empty_reference_bank():
    t, start = fixture(); model = SetEvidenceRetrieval(3, 2, start, 23)
    for k in ['pair', 'px', 'valid']: t[k] = t[k][:, :0]
    t['gb'] = t['gb'][:0]; t['mem'] = torch.eye(3)
    assert torch.isfinite(model(t)).all()
