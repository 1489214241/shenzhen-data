"""Round37: source-only globally solvable linear stopping boundary.

Changes head hypothesis class, not the frozen school scorer. Not a rescue of
round36, not an ATLOP reproduction, not a claim of equal parameter capacity.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit
import experiment_shenzhen_adaptive_threshold as T
from shenzhen_adaptive_threshold import at_loss
from shenzhen_convex_threshold import statistics, design, objective, solve, thresholds
F, R, E = T.F, T.R, T.E
PARENT = T.OUT
OUT = R.ROOT / 'round37_convex_threshold'


@contextmanager
def output_root():
    old = T.OUT; T.OUT = OUT
    try: yield
    finally: T.OUT = old


def verify():
    with output_root(): return T.verify()


def prepare():
    F.guard(); c = T.verify(); R.check(PARENT, 'cache_freeze.json')
    if OUT.exists(): raise FileExistsError('Preserve round37')
    c.update(steps=100, head='adaptive_head is a LINEAR35d query function plus intercept; constant_head is intercept only.',
        hypothesis_change='Changes head class from nonlinear35->32->1 to linear35->1; not capacity matched. Does not alter or rescue round36.',
        optimizer='float64 full Newton analytic Hessian, Armijo1e-4 backtracking; at most100steps, at most60line-search halvings.',
        stopping='Training gradient infinity norm<=1e-6; no validation checkpoint selection.',
        gradient_tolerance=1e-6, max_steps=100,
        regularization='0.01*((intercept+1)^2+mean(feature_weights^2)); no feature term for constant head.',
        convergence_gate='All10linear adaptive heads must reach gradient_inf<=1e-6; source predictive gates unchanged.',
        certificate='Strong convexity m>=2*reg/max(1,number_of_feature_weights); training objective gap <= ||gradient||_2^2/(2*m).',
        initializer='Intercept=-1; feature coefficients=0. Train-only frozen normalization from round34.',
        controls='Same source data, loss, ridge coefficient and solver for constant and linear heads; original relative-probability strong control retained.',
        checkpoint='Final training iterate only; no fit or early stopping on source validation or Shenzhen.',
        width=None, parameter_matching=False,
        research_basis='Reviewed ATLOP two categorical terms; affine threshold and strictly positive quadratic regularization make the objective strongly convex. Local probe, not ATLOP reproduction.')
    F.write(OUT/'protocol.json', c)
    links = dict(F.read(PARENT/'input_links.json')['files'])
    for name in ['input_freeze.json', 'cache_freeze.json', 'cache_audit.json']:
        links[str(PARENT/name)] = F.sha(PARENT/name)
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_convex_threshold.py',
              F.REPO/'tests/test_shenzhen_convex_threshold.py', F.REPO/'scripts/audit_shenzhen_round37_metrics.py',
              F.REPO/'scripts/audit_shenzhen_round37_seed_pairs.py']:
        links[str(p)] = F.sha(p)
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    for p in (PARENT/'cache').glob('*.npz'):
        dest = OUT/'cache'/p.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        assert F.sha(dest) == F.sha(p); links[str(p)] = F.sha(p)
    shutil.copyfile(PARENT/'cache_freeze.json', OUT/'cache_freeze.json')
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'input_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('PREPARED_ROUND37', flush=True)


def training_data(bank, c, adaptive):
    parts = [statistics(bank[d+'_train_z'], bank[d+'_train_y'], bank[d+'_train_w']) for d in c['sources']]
    stats = {k: np.concatenate([s[k] for s in parts]) for k in parts[0]}
    f = np.concatenate([bank[d+'_train_f'] for d in c['sources']])
    return design(f, bank['mean'], bank['scale'], adaptive), stats


def train():
    F.guard(); c = verify(); R.check(OUT, 'cache_freeze.json')
    if (OUT/'prediction_freeze.json').exists(): R.check(OUT, 'prediction_freeze.json'); return
    history = F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        bank = dict(np.load(OUT/'cache'/f'{seed}.npz'))
        for arm in T.ARMS:
            key = f'{arm}/{seed}'; adaptive = arm == 'adaptive_head'
            ck = OUT/'checkpoints'/f'{arm}_{seed}.npz'; dest = OUT/'margins'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck) == history[key]['checkpoint_sha256'] and F.sha(dest) == history[key]['margins_sha256']; continue
            x, stats = training_data(bank, c, adaptive)
            beta, result = solve(x, stats, c['reg'], c['gradient_tolerance'], c['max_steps'])
            payload = {}
            for d in c['sources']:
                for view in ['train', 'valid', 'stress_0', 'stress_1']:
                    k = d+'_'+view; z = bank[k+'_z']; f = bank[k+'_f']
                    th = thresholds(z, f, bank['mean'], bank['scale'], beta, adaptive)
                    payload[k+'_threshold'] = th; payload[k] = z.astype(float)-th[:, None]
            ck.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ck, beta=beta, mean=bank['mean'], scale=bank['scale'], adaptive=np.array(adaptive), protocol_hash=np.array(F.sha(OUT/'protocol.json')))
            dest.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(dest, **payload)
            result.update(checkpoint_sha256=F.sha(ck), margins_sha256=F.sha(dest), parameters=len(beta))
            history[key] = result; F.write(OUT/'training_history.json', history)
            print('CONVEX_TRAINED', key, result['reason'], result['final'], flush=True)
    F.write(OUT/'prediction_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for folder in ['checkpoints', 'margins'] for p in (OUT/folder).rglob('*') if p.is_file()}, target_predictions_generated=False))


def report():
    with output_root(): T.report()
    c = verify(); result = F.read(OUT/'source_report.json'); history = F.read(OUT/'training_history.json')
    converged = all(history[f'adaptive_head/{s}']['reason'] == 'gradient_converged' and history[f'adaptive_head/{s}']['final']['gradient_inf'] <= c['gradient_tolerance'] for s in c['seeds'])
    F.write(OUT/'admission_gate.json', dict(predictive_candidate=result['selected_target_candidate'], all_adaptive_converged=converged,
        selected_target_candidate='adaptive_head' if converged and result['selected_target_candidate']=='adaptive_head' else None,
        target_predictions_generated=False, adopted=False))


def replay():
    """Independent uncompressed PyTorch school loss, gradient and Hessian."""
    F.guard(); c = verify(); R.check(OUT, 'prediction_freeze.json'); R.check(OUT, 'cache_freeze.json')
    dest = OUT/'independent_replay_audit.json'
    if dest.exists(): raise FileExistsError('Preserve independent replay')
    history = F.read(OUT/'training_history.json'); checks = {}
    torch.set_num_threads(2)
    for seed in c['seeds']:
        bank = dict(np.load(OUT/'cache'/f'{seed}.npz'))
        assert F.sha(OUT/'cache'/f'{seed}.npz') == F.sha(PARENT/'cache'/f'{seed}.npz')
        for arm in T.ARMS:
            adaptive = arm == 'adaptive_head'; h = history[f'{arm}/{seed}']
            ck = OUT/'checkpoints'/f'{arm}_{seed}.npz'; path = OUT/'margins'/arm/f'{seed}.npz'
            assert F.sha(ck) == h['checkpoint_sha256'] and F.sha(path) == h['margins_sha256']
            saved = dict(np.load(ck)); margins = dict(np.load(path))
            assert saved['protocol_hash'].item() == F.sha(OUT/'protocol.json') and saved['adaptive'].item() == adaptive
            np.testing.assert_array_equal(saved['mean'], bank['mean']); np.testing.assert_array_equal(saved['scale'], bank['scale'])
            beta = torch.tensor(saved['beta'], dtype=torch.float64, requires_grad=True)
            frames = []; rows = 0; max_output_difference = 0.
            for d in c['sources']:
                for view in ['train', 'valid', 'stress_0', 'stress_1']:
                    k = d+'_'+view
                    z = torch.tensor(bank[k+'_z'], dtype=torch.float64)
                    f = torch.tensor(bank[k+'_f'], dtype=torch.float64)
                    f = (f - torch.tensor(saved['mean'], dtype=torch.float64)) / torch.tensor(saved['scale'], dtype=torch.float64)
                    def threshold(v, z=z, f=f): return z.max(1).values + v[0] + (f @ v[1:] if adaptive else 0)
                    th = threshold(beta).detach().numpy()
                    # Independent operation ordering may differ at last-bit precision.
                    np.testing.assert_allclose(th, margins[k+'_threshold'], atol=2e-12, rtol=2e-12)
                    np.testing.assert_allclose(z.numpy()-th[:, None], margins[k], atol=2e-12, rtol=2e-12)
                    max_output_difference = max(max_output_difference, float(np.abs(th-margins[k+'_threshold']).max()))
                    np.testing.assert_array_equal(z.numpy()-margins[k+'_threshold'][:, None], margins[k])
                    rows += len(z)
                    if view == 'train':
                        frames.append((z, f, torch.tensor(bank[k+'_y']), torch.tensor(bank[k+'_w'], dtype=torch.float64)))
            def full_loss(v):
                loss = c['reg'] * ((v[0]+1).square() + (v[1:].square().mean() if adaptive else 0))
                for z, f, y, w in frames:
                    th = z.max(1).values + v[0] + (f @ v[1:] if adaptive else 0)
                    loss = loss + at_loss(z, th, y, w)
                return loss
            full = full_loss(beta); grad = torch.autograd.grad(full, beta)[0].numpy()
            hessian = torch.autograd.functional.hessian(full_loss, beta).numpy()
            x, stats = training_data(bank, c, adaptive)
            value, gg, hh = objective(saved['beta'], x, stats, c['reg'])
            np.testing.assert_allclose(value, float(full.detach()), atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(gg, grad, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(hh, hessian, atol=2e-12, rtol=2e-12)
            assert value == h['final']['objective'] and float(np.abs(gg).max()) == h['final']['gradient_inf']
            converged = float(np.abs(grad).max()) <= c['gradient_tolerance']
            assert converged == (h['reason'] == 'gradient_converged')
            bound = 2*c['reg']/(len(beta)-1 if adaptive else 1)
            assert h['strong_convexity_lower_bound'] == bound
            assert np.linalg.eigvalsh(hessian).min() >= bound-1e-12
            np.testing.assert_allclose(grad@grad/(2*bound), h['final']['global_objective_gap_upper_bound'], atol=1e-14, rtol=1e-5)
            assert all(b['objective'] <= a['objective']+1e-12 for a,b in zip(h['history'],h['history'][1:]))
            check = dict(rows=rows, converged=converged, objective=value, gradient_inf=float(np.abs(grad).max()),
                global_objective_gap_upper_bound=h['final']['global_objective_gap_upper_bound'],
                full_school_loss_gradient_hessian_match=True, max_threshold_difference=max_output_difference)
            if not adaptive:
                a,b,n,w = [stats[k] for k in ['a','b','n','w']]
                exact = brentq(lambda v: np.sum(w*(n*expit(v-a)+expit(v-b)-1))+2*c['reg']*(v+1), -100,100,xtol=1e-13)
                assert abs(exact-saved['beta'][0])<1e-5
                check['scalar_analytic_optimum'] = exact
            checks[f'{arm}/{seed}'] = check
        print('CONVEX_REPLAYED', seed, flush=True)
    F.write(dest, dict(heads=checks, cached_inputs_identical_to_verified_round34=True,
        all_outputs_independently_replayed=True, full_loss_gradient_hessian_independently_verified=True,
        target_predictions_generated=False, audit_code_sha256=F.sha(__file__)))


def score():
    import audit_shenzhen_round37_metrics as A
    with output_root(): A.score()
    c = verify(); gate = F.read(OUT/'admission_gate.json'); h = F.read(OUT/'training_history.json'); report0 = F.read(OUT/'source_report.json')
    conv = all(h[f'adaptive_head/{s}']['reason']=='gradient_converged' and h[f'adaptive_head/{s}']['final']['gradient_inf']<=c['gradient_tolerance'] for s in c['seeds'])
    assert gate['all_adaptive_converged'] == conv
    assert gate['selected_target_candidate'] == ('adaptive_head' if conv and report0['selected_target_candidate']=='adaptive_head' else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare','train','report','replay','score'])
    globals()[p.parse_args().phase]()
