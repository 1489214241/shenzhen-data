"""Round33: separate late textual supervision from its reference-bank extension."""
import argparse
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import torch
import experiment_shenzhen_explicit_negative_training as PREV
import build_shenzhen_late_text_inputs as B

F, R, S = PREV.F, PREV.R, PREV.S
OUT = R.ROOT/'round33_late_text_training'
ARMS = ['scalar_control', 'textual_labels', 'textual_references']


def load(c, arm):
    xs, ys = PREV.load(c, 'features_labels')
    if arm != 'scalar_control':
        for d in c['targets']:
            with (B.OUT/d/('reference_inputs.pkl' if arm == 'textual_references' else 'control_inputs.pkl')).open('rb') as h:
                xs[d] = pickle.load(h)
            ys[d] = dict(np.load(B.OUT/d/'labels.npz'))
    return xs, ys


def prepare():
    F.guard(); c = PREV.verify(); R.check(PREV.OUT, 'probability_freeze.json'); R.check(B.OUT, 'manifest.json')
    audit = R.ROOT/'late_text_inputs_independent_audit.json'
    assert F.read(audit)['input_manifest_sha256'] == F.sha(B.OUT/'manifest.json')
    if OUT.exists(): raise FileExistsError('Preserve source experiment')
    config = {k: c[k] for k in ['sources', 'targets', 'seeds', 'steps', 'lr', 'reg', 'width', 'chunk', 'ranking_coefficient', 'source_gate']}
    config.update(arms=ARMS, primary_decoder='relative', primary_objective='balanced_exact',
        control='Reuse round32 features_labels checkpoints and source probabilities, renamed scalar_control for reporting. It retains the7 official negatives and repaired negative features.',
        textual_labels='19 previously unused textual positive labels,7new training locations. Target features and references identical to control.',
        textual_references='Same19positive labels plus19reference pairs;6more schools covered,7new reference locations. Withdraw3old singleton border-support locations now known to have extra positives; no new partial labels become negative border evidence.',
        geometry='12late geographic-estimate relations remain held; they are not mixed into textual supervision or reference extension.',
        source_bank='Identical full source multireference bank, labels and weights to round32. This differs from the959development incumbent; target comparison with that incumbent remains required.',
        weights='All original source/target weights bitwise preserved;7new independent property groups receive original10/5211mass each.',
        source_selection='Only textual_references_relative may advance: original gate AND balanced_exact >= BOTH scalar_control and textual_labels AND both stress exact counts >= BOTH controls. No fallback to secondary arm/decoder.',
        target_rule='Only after source audit passes, separately freeze target predictions with identical cumulative textual facts, prior geographic supplements and exclusions for ALL comparators; compare matched controls and959development incumbent.',
        calibration='Original1246source development rows only;1186single60multi. Reused, not independent confirmation.',
        no_target_gold=True, no_target_polygon=True, no_target_cardinality=True,
        target_predictions_generated=False, independent_validation=False, adopted=False)
    F.write(OUT/'protocol.json', config)
    links = dict(F.read(PREV.OUT/'input_links.json')['files'])
    for folder, name in [(B.OUT, 'manifest.json'), (PREV.OUT, 'probability_freeze.json')]:
        links[str(folder/name)] = F.sha(folder/name)
        links.update({str(folder/p): h for p, h in F.read(folder/name)['files'].items()})
    links.update(F.read(B.OUT/'input_links.json')['files'])
    for p in [audit, PREV.OUT/'source_report.json', R.ROOT/'round32_independent_completion_audit.json']:
        links[str(p)] = F.sha(p)
    previous_report = F.read(PREV.OUT/'source_report.json')
    F.write(OUT/'reused_control/source_report.json', dict(reports={
        'scalar_control_'+mode: previous_report['reports']['features_labels_'+mode] for mode in ['relative', 'absolute']},
        origin=str(PREV.OUT/'source_report.json'), origin_arm='features_labels'))
    bx, by = load(config, 'scalar_control'); checks = {}
    for arm in ARMS:
        xs, ys = load(config, arm); checks[arm] = {}
        for d, x in xs.items():
            oldweight_path = PREV.OUT/'weights/features_labels'/f'{d}.npz'
            old = np.load(oldweight_path)['weights']; w = old.copy(); added = sorted(set(ys[d]['train_ids'])-set(by[d]['train_ids']))
            if added:
                assert d in c['targets'] and not set(x.groups[added]) & set(x.groups[by[d]['train_ids']])
                for q in added: w[q] = (10/5211)/sum(x.groups[added] == x.groups[q])
            np.testing.assert_array_equal(w[by[d]['train_ids']], old[by[d]['train_ids']])
            assert not ((w != old) & (old != 0)).any()
            if d in c['sources'] or arm == 'scalar_control':
                np.testing.assert_array_equal(w, old)
                for k in by[d]: np.testing.assert_array_equal(ys[d][k], by[d][k])
            if arm != 'textual_references' or d in c['sources']:
                np.testing.assert_array_equal(x.pair, bx[d].pair)
                np.testing.assert_array_equal(x.ref_sites, bx[d].ref_sites)
            np.testing.assert_array_equal(x.anchor, bx[d].anchor)
            np.testing.assert_array_equal(x.self_mask, x.groups[:, None] == x.groups[x.ref_sites][None])
            path = OUT/'weights'/arm/f'{d}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, weights=w); links[str(oldweight_path)] = F.sha(oldweight_path)
            checks[arm][d] = dict(training_rows=len(ys[d]['train_ids']), added_rows=list(map(int, added)),
                references=len(x.ref_sites), reference_school_coverage=len(set(x.ref_schools)),
                all_old_weights_identical=True, new_weighted_mass=float(w.sum()-old.sum()))
    F.write(OUT/'input_weight_audit.json', checks)
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_source_trial_report.py',
              F.REPO/'scripts/shenzhen_partial_model.py', F.REPO/'scripts/shenzhen_balanced_ranking.py']:
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); links[str(p)] = F.sha(p)
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'input_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}, target_gold_read=False))
    print('TEXT_TRIAL_PREPARED', flush=True)


def verify():
    R.check(OUT, 'input_freeze.json')
    for p, h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p) == h, p
    return F.read(OUT/'protocol.json')


def frames(c, xs, ys, arm, districts):
    result = []
    for d in districts:
        ids = ys[d]['train_ids']; w = np.load(OUT/'weights'/arm/f'{d}.npz')['weights']
        for lo in range(0, len(ids), c['chunk']):
            q = ids[lo:lo+c['chunk']]; t = R.M.batch(xs[d], q)
            t['labels'] = torch.from_numpy(ys[d]['labels'][q]).to(R.DEVICE)
            t['sample_weight'] = torch.tensor(w[q], dtype=torch.float32, device=R.DEVICE); result.append(t)
    return result


def train():
    F.guard(); c = verify()
    if (OUT/'probability_freeze.json').exists(): R.check(OUT, 'probability_freeze.json'); return
    assert F.read(OUT/'positive_gradient_audit.json')['all_new_labels_push_up']
    banks = {a: load(c, a) for a in ARMS}; xs, ys = banks['scalar_control']
    start = R.M.LinearRetrieval(next(iter(xs.values())))
    source = frames(c, xs, ys, 'scalar_control', c['sources'])
    batches = {a: source+frames(c, *banks[a], a, c['targets']) for a in ARMS[1:]}
    ph = F.sha(OUT/'protocol.json'); history = F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        parent = torch.load(R.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt', map_location=R.DEVICE, weights_only=False)
        for arm in ARMS[1:]:
            key = f'{arm}/{seed}'; ck = OUT/'checkpoints'/f'{arm}_{seed}.pt'; dest = OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck) == history[key]['checkpoint_sha256'] and F.sha(dest) == history[key]['probabilities_sha256']; continue
            began = time.perf_counter(); model = R.BernoulliRetrieval(41, 27, start, seed, c['width']).to(R.DEVICE)
            model.load_state_dict(parent['model']); opt = torch.optim.Adam(model.parameters(), lr=c['lr']); begin = 0; losses = []
            if ck.exists():
                saved = torch.load(ck, map_location=R.DEVICE, weights_only=False); assert saved['protocol_hash'] == ph
                model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); begin = saved['step']; losses = saved['history']
            for step in range(begin, c['steps']):
                model.train(); opt.zero_grad(); reg = model.regularization(c['reg']); reg.backward(); bt = rt = 0.
                for t in batches[arm]:
                    z = model(t); bce = R.masked_bce(z, t['labels'], t['sample_weight']); rank = R.masked_lsep(z, t['labels'], t['sample_weight'])
                    loss = bce+c['ranking_coefficient']*rank; assert torch.isfinite(loss); loss.backward()
                    bt += float(bce.detach()); rt += float(rank.detach())
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5)); opt.step()
                losses.append(dict(bce=bt, ranking=rt, regularizer=float(reg.detach()), grad_norm=norm))
                if (step+1)%10 == 0:
                    R.checkpoint(ck, model, opt, 'full', step+1, losses, ph); print('TRAIN', arm, seed, step+1, losses[-1], flush=True)
            model.eval(); payload = S.source_probabilities(model, *banks[arm], c)
            dest.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(dest, **payload)
            history[key] = dict(checkpoint_sha256=F.sha(ck), probabilities_sha256=F.sha(dest), steps=len(losses), elapsed_seconds=time.perf_counter()-began, losses=losses)
            F.write(OUT/'training_history.json', history); print('MODEL_COMPLETE', key, flush=True)
        cp = OUT/'probabilities/scalar_control'/f'{seed}.npz'; cp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PREV.OUT/'probabilities/features_labels'/cp.name, cp)
        ck = OUT/'checkpoints'/f'scalar_control_{seed}.pt'; shutil.copyfile(PREV.OUT/'checkpoints'/f'features_labels_{seed}.pt', ck)
        history[f'scalar_control/{seed}'] = dict(reused=True, original_arm='round32/features_labels', checkpoint_sha256=F.sha(ck), probabilities_sha256=F.sha(cp))
        F.write(OUT/'training_history.json', history)
    F.write(OUT/'probability_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p)
        for folder in ['probabilities', 'checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()}, target_predictions_generated=False))


def report():
    from shenzhen_source_trial_report import report_trial
    F.guard(); c = verify(); R.check(OUT, 'probability_freeze.json'); _, ys = load(c, 'scalar_control')
    report_trial(F, R, OUT, c, ys, 'textual_references', ['scalar_control', 'textual_labels'], OUT/'reused_control')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'train', 'report']); globals()[p.parse_args().phase]()
