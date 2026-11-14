import copy
import unittest
import torch
from test_full_evidence import toy
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.full_evidence import FullEvidenceModel


class NormalizationTests(unittest.TestCase):
    def test_only_normalization_changes_and_batch_independence(self):
        t, p, a = toy()
        original = FullEvidenceModel('modernnca', p, a, 7)
        candidate = LayerNormModernNCA(p, a, 7).eval()
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm1d) for m in candidate.modules()))
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in candidate.modules()), 2)
        for name, value in original.named_parameters():
            self.assertTrue(torch.equal(value, dict(candidate.named_parameters())[name]))
        before = candidate(t)
        changed = dict(t, px=t['px'].clone(), ax=t['ax'].clone())
        changed['px'][1:] *= 100
        changed['ax'][1:] *= 100
        self.assertTrue(torch.equal(before[0], candidate(changed)[0]))
        one = dict(t, px=t['px'][:1], ax=t['ax'][:1], valid=t['valid'][:1], y=t['y'][:1])
        self.assertTrue(torch.allclose(before[:1], candidate(one), atol=2e-6))

    def test_masks_labels_candidates_permutations_and_gradients(self):
        t, p, a = toy()
        m = LayerNormModernNCA(p, a, 7).eval()
        out = m(t)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.sum(-1), torch.ones(3)))
        self.assertTrue((out[:, 3] > 0).all())
        self.assertTrue(torch.equal(out, m(dict(t, y=t['y'].flip(0)))))
        changed = copy.deepcopy(t); changed['px'][:, 0] = 1e6
        self.assertTrue(torch.equal(out, m(changed)))
        perm = torch.tensor([4, 2, 0, 3, 1])
        changed = dict(t, px=t['px'][:, perm], valid=t['valid'][:, perm],
            gb=t['gb'][perm], mem=torch.cat([t['mem'][:5][perm], t['mem'][5:]]))
        self.assertTrue(torch.allclose(out, m(changed), atol=2e-6))
        perm = torch.tensor([3, 1, 0, 2])
        changed = dict(t, ax=t['ax'][:, perm],
            mem=torch.cat([t['mem'][:5], t['mem'][5:][perm]])[:, perm])
        self.assertTrue(torch.allclose(out[:, perm], m(changed), atol=2e-6))
        t['px'].requires_grad_(); t['ax'].requires_grad_()
        m(t)[:, 1].log().sum().backward()
        self.assertTrue((t['px'].grad[:, 1:].abs().sum((0,1)) > 0).all())
        self.assertTrue((t['ax'].grad.abs().sum((0,1)) > 0).all())
        self.assertEqual(float(t['px'].grad[:,0].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
