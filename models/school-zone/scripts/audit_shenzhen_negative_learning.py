"""Training-observation diagnostics for round32, without target evaluation.

Probe only the seven official exclusions used as observed training labels.
This does not generate complete Shenzhen predictions or select a candidate.
"""
from pathlib import Path
import numpy as np
import torch
import experiment_shenzhen_explicit_negative_training as E

F, R = E.F, E.R


def main():
    F.guard(); c = E.verify(); R.check(E.OUT, 'probability_freeze.json')
    dest = E.OUT/'observed_negative_learning_audit.json'
    if dest.exists(): raise FileExistsError('Preserve observed-label diagnostic')
    data = {a: E.load(c, a) for a in E.ARMS}
    start = R.M.LinearRetrieval(next(iter(data['scalar_control'][0].values())))
    rows = []; all_values = {}
    for d in c['targets']:
        meta = F.read(R.OLD/'inputs'/f'{d}.json')
        site_index = {s['coordinate_group_id']: q for q, s in enumerate(meta['sites'])}
        school_index = {s['school_id']: j for j, s in enumerate(meta['schools'])}
        facts = F.read(E.B.FACTS/d/'facts.json')['negative_facts']
        pairs = [(site_index[f['site_id']], school_index[f['school_id']], f) for f in facts]
        if not pairs: continue
        ids = sorted({q for q, j, f in pairs}); local = {q: i for i, q in enumerate(ids)}
        for arm in E.ARMS:
            xs, ys = data[arm]
            t = R.M.batch(xs[d], ids)
            for seed in c['seeds']:
                ck = E.OUT/'checkpoints'/f'{arm}_{seed}.pt'
                model = R.BernoulliRetrieval(41, 27, start, seed, c['width']).to(R.DEVICE)
                model.load_state_dict(torch.load(ck, map_location=R.DEVICE, weights_only=False)['model'])
                model.eval()
                with torch.no_grad(): z = model(t).cpu().numpy()
                assert np.isfinite(z).all()
                all_values[arm, seed, d] = z
        for q, j, fact in pairs:
            y = data['features_labels'][1][d]['labels'][q]
            assert y[j] == 0 and data['scalar_control'][1][d]['labels'][q, j] == -1
            positive = np.flatnonzero(y == 1)
            per_seed = {}
            for seed in c['seeds']:
                measured = {}
                for arm in E.ARMS:
                    z = all_values[arm, seed, d][local[q]]
                    measured[arm] = dict(negative_logit=float(z[j]),
                        negative_bce=float(np.logaddexp(0., float(z[j]))),
                        least_positive_minus_negative=float(z[positive].min()-z[j]) if len(positive) else None)
                delta = measured['features_labels']['negative_logit']-measured['features_only']['negative_logit']
                per_seed[str(seed)] = dict(arms=measured, label_intervention_logit_delta=delta)
            rows.append(dict(district=d, site_id=fact['site_id'], school_id=fact['school_id'],
                building_name=meta['sites'][q]['building_name'], school_name=meta['schools'][j]['school_name'],
                observed_positive_count=len(positive), per_seed=per_seed,
                label_intervention_mean_logit_delta=float(np.mean([v['label_intervention_logit_delta'] for v in per_seed.values()]))))
    assert len(rows) == 7
    delta = [v['label_intervention_logit_delta'] for r in rows for v in r['per_seed'].values()]
    summary = dict(observed_pairs=len(rows), seeds=len(c['seeds']), pair_seed_probes=len(delta),
        logits_decreased=sum(v < 0 for v in delta), logits_increased=sum(v > 0 for v in delta),
        mean_logit_delta=float(np.mean(delta)), min_logit_delta=min(delta), max_logit_delta=max(delta),
        interpretation='Training-label response only; no held-out accuracy, no new model selection or target scoring.')
    F.write(dest, dict(summary=summary, rows=rows,
        probability_freeze_sha256=F.sha(E.OUT/'probability_freeze.json'), audit_code_sha256=F.sha(__file__),
        target_gold_read=False, target_polygon_read=False, full_target_predictions_generated=False))
    print('OBSERVED_NEGATIVE_LEARNING', summary, flush=True)


if __name__ == '__main__': main()
