import copy
import math
import unittest
import torch
from test_full_evidence import toy
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.modernnca_support import SupportModernNCA, support_covariance, ridge_scores


class SupportTests(unittest.TestCase):
    def test_identity_and_initialization(self):
        t, p, a = toy(); base = LayerNormModernNCA(p, a, 7).eval()
        rng = torch.get_rng_state()
        for kind in ['identity', 'scalar', 'diagonal', 'covariance', 'ridge']:
            m = SupportModernNCA(p, a, 7, kind, strength=0).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for n, v in base.state_dict().items(): self.assertTrue(torch.equal(v, m.state_dict()[n]))
            if kind != 'ridge': self.assertTrue(torch.allclose(base(t), m(t), atol=2e-7))

    def test_contract(self):
        for kind in ['scalar', 'diagonal', 'covariance', 'ridge']:
            t, p, a = toy(); m = SupportModernNCA(p, a, 7, kind).eval()
            out = m(t)
            self.assertTrue(torch.allclose(out.sum(-1), torch.ones(3), atol=1e-6))
            self.assertTrue((out[:, 3] > 0).all())
            self.assertTrue(torch.equal(out, m(dict(t, y=t['y'].flip(0)))))
            c = copy.deepcopy(t); c['px'][:, 0] = 1e6
            self.assertTrue(torch.equal(out, m(c)))
            perm = torch.tensor([4, 2, 0, 3, 1])
            c = dict(t, px=t['px'][:, perm], valid=t['valid'][:, perm], gb=t['gb'][perm],
                     mem=torch.cat([t['mem'][:5][perm], t['mem'][5:]]))
            self.assertTrue(torch.allclose(out, m(c), atol=3e-6))
            perm = torch.tensor([3, 1, 0, 2])
            c = dict(t, ax=t['ax'][:, perm], mem=torch.cat([t['mem'][:5], t['mem'][5:][perm]])[:, perm])
            self.assertTrue(torch.allclose(out[:, perm], m(c), atol=3e-6))
            # All provided feature channels have a differentiable path.
            t['px'].requires_grad_(); t['ax'].requires_grad_()
            m(t)[:, 1].log().sum().backward()
            self.assertEqual(float(t['px'].grad[:, 0].abs().sum()), 0.)
            self.assertTrue((t['px'].grad[:, 1:].abs().sum((0, 1)) > 0).all())
            self.assertTrue((t['ax'].grad.abs().sum((0, 1)) > 0).all())
            for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            for n in [0, 1]:
                c = dict(t, px=t['px'][:, :n], valid=torch.ones_like(t['valid'][:, :n]),
                         gb=t['gb'][:n]*0, mem=torch.cat([t['mem'][:n], t['mem'][5:]]))
                m.zero_grad(); pred = m(c); pred[:, 0].log().sum().backward()
                self.assertTrue(torch.isfinite(pred).all())
                for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            c = dict(t, valid=torch.zeros_like(t['valid']))
            self.assertTrue(torch.isfinite(m(c)).all())

    def test_covariance_group_weights(self):
        e = torch.tensor([[[1., 2.], [3., 4.], [6., 2.]]], dtype=torch.double)
        w = torch.ones(1, 3, dtype=torch.double)
        cov, rel, n = support_covariance(e, w)
        self.assertTrue(torch.allclose(cov[0], torch.cov(e[0].T, correction=0)))
        ix = torch.tensor([0, 0, 1, 2])
        cc, rr, nn = support_covariance(e[:, ix], torch.tensor([[.5, .5, 1., 1.]], dtype=torch.double))
        self.assertTrue(torch.allclose(cc, cov)); self.assertTrue(torch.equal(rr, rel)); self.assertTrue(torch.equal(nn, n))
        for weight in [w*0, w/3]:
            _, reliability, _ = support_covariance(e, weight)
            self.assertTrue(torch.equal(reliability, torch.zeros_like(reliability)))

    def test_ridge_against_augmented_least_squares(self):
        torch.manual_seed(3)
        e = torch.randn(2, 7, 4, dtype=torch.double, requires_grad=True)
        q = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        w = torch.rand(2, 7, dtype=torch.double)
        mem = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 2, 0, 1, 2]), 3).double()
        actual = ridge_scores(e, q, w, mem, .1)
        for b in range(2):
            x = torch.cat([e[b]/2, torch.ones(7, 1, dtype=torch.double)], 1)
            sw = (w[b]/w[b].sum()).sqrt()
            penalty = torch.cat([math.sqrt(.1)*torch.eye(4, dtype=torch.double), torch.zeros(4, 1, dtype=torch.double)], 1)
            xx = torch.cat([x * sw[:, None], penalty])
            yy = torch.cat([mem * sw[:, None], torch.zeros(4, 3, dtype=torch.double)])
            coef = torch.linalg.lstsq(xx, yy).solution
            expected = torch.cat([q[b]/2, torch.ones(1, dtype=torch.double)]) @ coef
            self.assertTrue(torch.allclose(actual[b], expected, atol=1e-10))
        self.assertTrue(torch.autograd.gradcheck(lambda ee, qq: ridge_scores(ee, qq, w, mem), (e, q)))


if __name__ == '__main__': unittest.main()
