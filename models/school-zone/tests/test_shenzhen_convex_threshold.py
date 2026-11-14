import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shenzhen_adaptive_threshold import at_loss
from shenzhen_convex_threshold import statistics, design, objective, solve, thresholds


class ConvexThresholdTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        self.z = rng.normal(size=(17, 8))
        self.y = np.zeros((17, 8), dtype=int)
        for i in range(17): self.y[i, rng.choice(8, 1 + i % 3, replace=False)] = 1
        self.w = rng.uniform(.1, 1, 17); self.w /= self.w.sum()
        self.f = rng.normal(size=(17, 4))
        self.x = design(self.f, np.zeros(4), np.ones(4), True)
        self.s = statistics(self.z, self.y, self.w)

    def test_full_school_loss_gradient_and_hessian(self):
        beta = np.array([-.9, .1, -.2, .3, -.4])
        value, grad, hess = objective(beta, self.x, self.s, .01)
        z, x, w = [torch.tensor(v, dtype=torch.float64) for v in [self.z, self.x, self.w]]
        y = torch.tensor(self.y)
        b = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
        def full(v):
            return at_loss(z, z.max(1).values + x @ v, y, w) + .01 * ((v[0] + 1)**2 + v[1:].square().mean())
        loss = full(b)
        self.assertAlmostEqual(value, float(loss), places=12)
        np.testing.assert_allclose(grad, torch.autograd.grad(loss, b)[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(hess, torch.autograd.functional.hessian(full, b), atol=1e-12, rtol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(hess).min(), .005)

    def test_scalar_optimum_and_convergence_certificate(self):
        x = design(self.f, None, None, False)
        beta, report = solve(x, self.s)
        exact = brentq(lambda b: objective(np.array([b]), x, self.s, .01)[1][0], -100, 100)
        self.assertEqual(report['reason'], 'gradient_converged')
        self.assertAlmostEqual(beta[0], exact, places=5)
        fitted = objective(beta, x, self.s, .01)[0]
        optimal = objective(np.array([exact]), x, self.s, .01)[0]
        self.assertLessEqual(fitted - optimal, report['final']['global_objective_gap_upper_bound'] + 1e-13)

    def test_multifeature_solve_and_score_shift(self):
        beta, report = solve(self.x, self.s)
        self.assertEqual(report['reason'], 'gradient_converged')
        vals = [h['objective'] for h in report['history']]
        self.assertTrue(np.all(np.diff(vals) <= 1e-12))
        t = thresholds(self.z, self.f, np.zeros(4), np.ones(4), beta, True)
        tt = thresholds(self.z + 8, self.f, np.zeros(4), np.ones(4), beta, True)
        np.testing.assert_allclose(t + 8, tt, atol=1e-12, rtol=1e-12)
        perm = np.arange(7, -1, -1)
        ss = statistics(self.z[:, perm], self.y[:, perm], self.w)
        bb, _ = solve(self.x, ss)
        np.testing.assert_allclose(beta, bb, atol=1e-12, rtol=1e-12)

    def test_unknown_not_negative(self):
        y = self.y.copy(); y[0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'complete labels'):
            statistics(self.z, y, self.w)


if __name__ == '__main__': unittest.main()
