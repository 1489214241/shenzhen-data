import unittest
import torch
from zoning.full_evidence import FullEvidenceModel, tokens, ModernNCA


def toy():
    torch.manual_seed(31)
    b, r, s, np_, na = 3, 5, 4, 7, 6
    rs = torch.tensor([0, 0, 1, 2, 2])
    mem = torch.cat([torch.nn.functional.one_hot(rs, s), torch.eye(s)], 0).float()
    return dict(px=torch.randn(b, r, np_ + na + 6), ax=torch.randn(b, s, na + 6),
                gb=torch.tensor([-.69, -.69, 0., 0., 0.]),
                valid=torch.tensor([[False, True, True, True, True]] * b), mem=mem,
                y=torch.tensor([0, 1, 2])), np_, na


class FullEvidenceTests(unittest.TestCase):
    def test_lossless_packing(self):
        t, p, a = toy()
        x, v = tokens(t, p)
        self.assertTrue(torch.equal(x[:, :5, :-2], t['px']))
        self.assertTrue(torch.equal(x[:, 5:, p:-2], t['ax']))
        self.assertTrue(torch.equal(x[:, :5, -2], t['gb'][None].expand(3, -1)))
        self.assertTrue(v[:, 5:].all())  # school 3 has zero reference, remains a candidate

    def test_permutation_masks_labels_and_gradients(self):
        for kind in ['matching', 'modernnca', 'proto', 'deepsets']:
            with self.subTest(method=kind):
                t, p, a = toy()
                m = FullEvidenceModel(kind, p, a, 7).eval()
                out = m(t)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.allclose(out.sum(-1), torch.ones(3)))
                self.assertTrue((out[:, 3] > 0).all())
                changed = dict(t, y=t['y'].flip(0))
                self.assertTrue(torch.equal(out, m(changed)))
                perm = torch.tensor([4, 2, 0, 3, 1])
                changed = dict(t, px=t['px'][:, perm], valid=t['valid'][:, perm],
                               gb=t['gb'][perm], mem=torch.cat([t['mem'][:5][perm], t['mem'][5:]]))
                self.assertTrue(torch.allclose(out, m(changed), atol=2e-6))
                school_perm = torch.tensor([3, 1, 0, 2])
                changed = dict(t, ax=t['ax'][:, school_perm],
                               mem=torch.cat([t['mem'][:5], t['mem'][5:][school_perm]])[:, school_perm])
                self.assertTrue(torch.allclose(out[:, school_perm], m(changed), atol=2e-6))
                changed = dict(t, px=t['px'].clone())
                changed['px'][:, 0] = 1e6  # masked references cannot alter normalization
                self.assertTrue(torch.equal(out, m(changed)))
                t['px'].requires_grad_(); t['ax'].requires_grad_()
                m(t)[:, 1].log().sum().backward()
                self.assertTrue((t['px'].grad[:, 1:].abs().sum((0, 1)) > 0).all())
                self.assertTrue((t['ax'].grad.abs().sum((0, 1)) > 0).all())
                self.assertEqual(float(t['px'].grad[:, 0].abs().sum()), 0.)

    def test_modernnca_readout_against_official_core(self):
        t, p, a = toy()
        t['gb'].zero_()  # ordinary equally weighted official support set
        m = FullEvidenceModel('modernnca', p, a, 7).eval()
        captured = {}
        h1 = m.encoder.post_encoder.register_forward_hook(lambda mod, args, out: captured.update(e=out))
        h2 = m.query.register_forward_hook(lambda mod, args, out: captured.update(q=out))
        actual = m(t)
        h1.remove(); h2.remove()
        _, valid = tokens(t, p)
        e = torch.zeros(3, 9, m.dim); e[valid] = captured['e']
        official = ModernNCA(d_in=m.dim, d_num=m.dim, d_out=4, dim=m.dim,
                             dropout=0., d_block=64, n_blocks=0, num_embeddings=None).eval()
        with torch.no_grad():
            official.encoder.weight.copy_(torch.eye(m.dim)); official.encoder.bias.zero_()
        for i in range(3):
            expected = official(captured['q'][i:i+1], None, e[i][valid[i]],
                                t['mem'][valid[i]].argmax(-1), False).exp() - 1e-7
            self.assertTrue(torch.allclose(actual[i], expected, atol=2e-6))


if __name__ == '__main__':
    unittest.main()
