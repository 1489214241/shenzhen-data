"""Convex, frozen-ranker AT threshold with analytic Newton derivatives.

This is a local linear-head probe, not a reproduction of ATLOP. All school
scores are frozen. Complete training labels are required; unknowns are never
converted to negatives. Positive ridge on every coefficient makes the training
objective strongly convex, allowing a global objective-gap upper bound.
"""
import numpy as np
from scipy.special import expit, logsumexp


def statistics(logits, labels, weights):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim != 2 or y.shape != z.shape or w.shape != z.shape[:1]:
        raise ValueError('Mismatched training shapes')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Stopping boundary requires complete labels')
    if not ((y == 1).any(1) & (y == 0).any(1)).all():
        raise ValueError('Each training row needs positive and negative schools')
    if not np.isfinite(z).all() or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ValueError('Invalid scores or weights')
    z = z - z.max(1, keepdims=True)
    return dict(a=logsumexp(np.where(y == 1, z, -np.inf), axis=1),
                b=logsumexp(np.where(y == 0, z, -np.inf), axis=1),
                n=(y == 1).sum(1), p=np.where(y == 1, z, 0).sum(1), w=w)


def design(features, mean, scale, adaptive):
    f = np.asarray(features, dtype=np.float64)
    if not adaptive:
        return np.ones((len(f), 1), dtype=np.float64)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if f.ndim != 2 or f.shape[1:] != mean.shape or mean.shape != scale.shape or (scale <= 0).any():
        raise ValueError('Invalid feature normalization')
    return np.column_stack([np.ones(len(f)), (f - mean) / scale])


def penalty_diagonal(size, reg):
    if size < 1 or reg <= 0:
        raise ValueError('Positive ridge required for global certificate')
    return np.array([reg] + [reg / max(1, size - 1)] * (size - 1))


def objective(beta, x, stats, reg):
    delta = x @ beta
    a, b, n, p, w = [stats[k] for k in ['a', 'b', 'n', 'p', 'w']]
    shifted = beta.copy(); shifted[0] += 1
    ridge = penalty_diagonal(len(beta), reg)
    terms = n * np.logaddexp(delta, a) - p + np.logaddexp(delta, b) - delta
    value = float(w @ terms + ridge @ (shifted ** 2))
    sa, sb = expit(delta - a), expit(delta - b)
    gradient = x.T @ (w * (n * sa + sb - 1)) + 2 * ridge * shifted
    # expit(-v) avoids losing curvature by subtracting an almost-one sigmoid.
    curvature = w * (n * sa * expit(a - delta) + sb * expit(b - delta))
    hessian = x.T @ (curvature[:, None] * x) + np.diag(2 * ridge)
    return value, gradient, hessian


def solve(x, stats, reg=.01, tolerance=1e-6, max_steps=100):
    x = np.asarray(x, dtype=np.float64)
    beta = np.zeros(x.shape[1]); beta[0] = -1
    history = []
    reason = 'max_iterations'
    strong_convexity = float(2 * penalty_diagonal(len(beta), reg).min())
    for step in range(max_steps + 1):
        value, gradient, hessian = objective(beta, x, stats, reg)
        gi = float(np.abs(gradient).max())
        gap = float(gradient @ gradient / (2 * strong_convexity))
        history.append(dict(iteration=step, objective=value, gradient_inf=gi,
                            global_objective_gap_upper_bound=gap))
        if gi <= tolerance:
            reason = 'gradient_converged'; break
        if step == max_steps:
            break
        direction = np.linalg.solve(hessian, -gradient)
        descent = float(gradient @ direction)
        if not descent < 0:
            raise ArithmeticError('Newton direction is not a descent direction')
        rate = 1.
        for _ in range(60):
            candidate = beta + rate * direction
            next_value = objective(candidate, x, stats, reg)[0]
            if np.isfinite(next_value) and next_value <= value + 1e-4 * rate * descent:
                beta = candidate
                history[-1]['accepted_step'] = rate
                break
            rate *= .5
        else:
            reason = 'line_search_failed'; break
    return beta, dict(reason=reason, final=history[-1], history=history,
                      strong_convexity_lower_bound=strong_convexity)


def thresholds(logits, features, mean, scale, beta, adaptive):
    return np.asarray(logits, dtype=np.float64).max(1) + design(features, mean, scale, adaptive) @ beta
