import copy
import unittest
import torch
from test_full_evidence import toy
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.modernnca_targeted import TargetedModernNCA


class TargetedTests(unittest.TestCase):
    def test_recovery_and_rng(self):
        t, p, a = toy()
        baseline = LayerNormModernNCA(p, a, 7).eval()
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        for kind in ['schoolquery', 'adaptivegate', 'constantgate']:
            m = TargetedModernNCA(p, a, 7, kind).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(cuda_rng, torch.cuda.get_rng_state_all())))
            self.assertTrue(torch.allclose(baseline(t), m(t), atol=2e-7))
            for name, value in baseline.named_parameters():
                self.assertTrue(torch.equal(value, dict(m.named_parameters())[name]))

    def test_contract_after_nonzero_adaptation(self):
        for kind in ['schoolquery', 'adaptivegate', 'constantgate']:
            t, p, a = toy(); m = TargetedModernNCA(p, a, 7, kind).eval()
            with torch.no_grad():
                if kind == 'constantgate': m.offset.fill_(.4)
                else: m.residual[-1].weight.normal_(0., .1)
            out = m(t)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.allclose(out.sum(-1), torch.ones(3), atol=1e-6))
            self.assertTrue((out[:, 3] > 0).all())
            self.assertTrue(torch.equal(out, m(dict(t, y=t['y'].flip(0)))))
            changed = copy.deepcopy(t); changed['px'][:, 0] = 1e6
            self.assertTrue(torch.equal(out, m(changed)))
            perm = torch.tensor([4, 2, 0, 3, 1])
            changed = dict(t, px=t['px'][:, perm], valid=t['valid'][:, perm], gb=t['gb'][perm],
                mem=torch.cat([t['mem'][:5][perm], t['mem'][5:]]))
            self.assertTrue(torch.allclose(out, m(changed), atol=2e-6))
            perm = torch.tensor([3, 1, 0, 2])
            changed = dict(t, ax=t['ax'][:, perm], mem=torch.cat([t['mem'][:5], t['mem'][5:][perm]])[:, perm])
            self.assertTrue(torch.allclose(out[:, perm], m(changed), atol=2e-6))
            t['px'].requires_grad_(); t['ax'].requires_grad_()
            m(t)[:, 1].log().sum().backward()
            self.assertTrue((t['px'].grad[:, 1:].abs().sum((0, 1)) > 0).all())
            self.assertTrue((t['ax'].grad.abs().sum((0, 1)) > 0).all())
            self.assertEqual(float(t['px'].grad[:, 0].abs().sum()), 0.)
            for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            for empty in [False, True]:
                no = dict(t, valid=torch.zeros_like(t['valid']))
                if empty: no.update(px=t['px'][:, :0], valid=t['valid'][:, :0], gb=t['gb'][:0], mem=t['mem'][5:])
                m.zero_grad(); pred = m(no); pred[:, 0].log().sum().backward()
                self.assertTrue(torch.isfinite(pred).all())
                for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())


if __name__ == '__main__': unittest.main()
