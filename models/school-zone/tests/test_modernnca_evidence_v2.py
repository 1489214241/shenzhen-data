import unittest
import torch
from zoning.modernnca_multilabel import MultiLabelModernNCA
from zoning.modernnca_evidence_v2 import expanded_model, observed_mean_loss


class ModernEvidenceTests(unittest.TestCase):
    def test_padding_preserves_all_old_token_projections(self):
        old = MultiLabelModernNCA(41, 27, 17).eval()
        new = expanded_model(old.state_dict(), 17).eval()
        x = torch.randn(8, 76)
        y = torch.zeros(8, 82)
        y[:, :41] = x[:, :41]; y[:, 44:71] = x[:, 41:68]; y[:, 74:] = x[:, 68:]
        torch.testing.assert_close(old.encoder.encoder(x), new.encoder.encoder(y), atol=1e-6, rtol=1e-6)

    def test_unknown_cells_have_zero_gradient_and_explicit_extras_are_penalized(self):
        z = torch.tensor([[0., 0., 0.], [0., 0., 0.]], requires_grad=True)
        y = torch.tensor([[1., -1., 0.], [1., 0., 0.]])
        observed_mean_loss(z, y, torch.ones(2)).backward()
        self.assertEqual(float(z.grad[0, 1]), 0.)
        self.assertGreater(float(z.grad[0, 2]), 0.)
        self.assertGreater(float(z.grad[1, 1]), 0.)
        self.assertLess(float(z.grad[0, 0]), 0.)


if __name__ == '__main__':
    unittest.main()
