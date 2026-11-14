"""Mechanism checks: legal support masking, label independence and metric gradients."""
import unittest
from types import SimpleNamespace
import torch
from zoning.model import RetrievalScorer
from zoning.modernnca_relation import ModernNCARelation


class ModernNCARelationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np_, na, q, r, s = 5, 3, 4, 6, 3
        self.start = SimpleNamespace(p=torch.tensor([-1., 0, 0, 0, 0]),
                                     a=torch.tensor([-1., 0, 0]), offset=torch.tensor(-2.))
        pair = torch.rand(q, r, np_); anchor = torch.rand(q, s, na)
        ctx = torch.rand(q, s, 6); rs = torch.tensor([0, 0, 1, 1, 2, 2])
        self.t = dict(pair=pair, anchor=anchor, gb=torch.zeros(r),
            px=torch.cat([pair, anchor[:, rs], ctx[:, rs]], -1),
            ax=torch.cat([anchor, ctx], -1), valid=torch.ones(q, r, dtype=torch.bool),
            mem=torch.cat([torch.nn.functional.one_hot(rs, s).float(), torch.eye(s)]),
            coords=torch.randn(10, 2), qids=torch.arange(q), refids=torch.arange(q, q+r),
            y=torch.tensor([0, 1, 2, 0]))
        self.m = ModernNCARelation(np_, na, self.start, 77).eval()

    def test_zero_metric_recovers_reference(self):
        base = RetrievalScorer(5, 3, self.start, 77).eval()
        with torch.no_grad(): self.m.log_metric_scale.fill_(-100)
        torch.testing.assert_close(self.m(self.t), base(self.t), atol=1e-6, rtol=1e-6)

    def test_reference_permutation(self):
        t = dict(self.t); p = torch.tensor([3, 0, 5, 1, 4, 2])
        for k in ['pair', 'px', 'valid']: t[k] = t[k][:, p]
        for k in ['gb', 'refids']: t[k] = t[k][p]
        t['mem'] = torch.cat([t['mem'][:6][p], t['mem'][6:]])
        torch.testing.assert_close(self.m(t), self.m(self.t), atol=1e-6, rtol=1e-6)

    def test_query_labels_do_not_affect_prediction(self):
        t = dict(self.t, y=torch.tensor([2, 2, 0, 1]))
        torch.testing.assert_close(self.m(t), self.m(self.t), atol=0, rtol=0)

    def test_masked_reference_does_not_contribute(self):
        t = dict(self.t, valid=torch.zeros_like(self.t['valid']))
        p = self.m(t)
        torch.testing.assert_close(p.sum(1), torch.ones(4))
        # Changing masked reference coordinates cannot affect school anchors.
        t2 = dict(t, coords=t['coords'].clone()); t2['coords'][4:] += 100
        torch.testing.assert_close(p, self.m(t2), atol=0, rtol=0)

    def test_metric_receives_nonzero_gradient(self):
        p = self.m(self.t); loss = -p[torch.arange(4), self.t['y']].log().mean(); loss.backward()
        self.assertTrue(torch.isfinite(self.m.metric.encoder.weight.grad).all())
        self.assertGreater(self.m.metric.encoder.weight.grad.abs().sum().item(), 1e-8)


if __name__ == '__main__': unittest.main()
