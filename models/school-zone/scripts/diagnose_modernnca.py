"""Read-only baseline diagnosis on previously used development snapshots.

This does not select a new model using target labels. Counterfactual readouts
are diagnostics, not deployable accuracy claims or independent confirmation.
"""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import run_full_evidence as base
from zoning import DEVICE, evaluate
from zoning.full_evidence import tokens
from run_modernnca_relation import cpu_task, device_batch, save, digest

DOC = REPO.parent / '科研推进/ModernNCA机制诊断与改进_20261112'


@torch.no_grad()
def components(m, t):
    x, valid = tokens(t, m.np)
    b, n, _ = x.shape
    r, s = t['px'].shape[1], t['ax'].shape[1]
    ev = m.encoder.post_encoder(m.encoder.encoder(x[valid]))
    e = ev.new_zeros(b, n, m.dim); e[valid] = ev
    mass = torch.cat([t['gb'].exp(), t['gb'].new_ones(s)])[None] * valid
    count = mass @ t['mem']
    proto = ((e * mass[..., None]).transpose(1, 2) @ t['mem']).transpose(1, 2) / count[..., None].clamp_min(1e-12)
    query = m.query(torch.cat([proto.mean(1), e[:, r:].mean(1)], -1))
    logits = -torch.linalg.vector_norm(query[:, None] - e, dim=-1) * m.log_temperature.exp().clamp(max=100.)
    logits += mass.clamp_min(1e-30).log()
    logits.masked_fill_(~valid, -torch.inf)
    w = logits.softmax(-1)
    return dict(P=w @ t['mem'], reference=w[:, :r] @ t['mem'][:r],
                anchor=w[:, r:], top=t['mem'][logits.argmax(1)].argmax(1),
                count=count, logits=logits)


def diagnose():
    result, records, replay = {}, [], 0.
    for k, s in [(0, 1), (1, 1)]:
        c, D, R, X = base.loaded(k, s)
        batch = f'k{k}_s{s}'; result[batch] = {}
        for d in D:
            rows = evaluate.eval_rows(D[d], R[d].known)
            ids = [row['q'] for row in rows]
            task = cpu_task(d, ids, D, X)
            district_records = []
            for seed in base.SEEDS:
                weight = base.OUT / batch / 'modernnca/seeds' / f'{d}_{seed}.pt'
                m = base.make('modernnca', X[d], None, seed)
                m.load_state_dict(torch.load(weight, map_location=DEVICE, weights_only=True)); m.eval()
                saved = np.load(weight.with_suffix('.npz'))['P'][ids]
                for st in range(0, len(ids), 64):
                    t = device_batch(task, slice(st, st+64)); z = components(m, t)
                    p = z['P'].cpu().numpy()
                    replay = max(replay, float(np.max(np.abs(p-saved[st:st+64]))))
                    arrays = {key: value.cpu().numpy() for key, value in z.items()}
                    valid = t['valid'].cpu().numpy()
                    for j, row in enumerate(rows[st:st+64]):
                        if len(row['truth']) != 1:
                            continue  # mechanism categories below require a unique true school
                        truth = int(row['truth'][0]); pred = int(p[j].argmax())
                        has_true = bool(np.any(valid[j] & (X[d].ref_schools == truth)))
                        ref_pred = int(arrays['reference'][j].argmax())
                        top = int(arrays['top'][j])
                        normalized = p[j] / arrays['count'][j]
                        if pred == truth:
                            category = 'correct'
                        elif not has_true:
                            category = 'wrong_no_true_reference'
                        elif ref_pred == truth:
                            category = 'anchor_overturns_correct_reference_sum'
                        elif top == truth:
                            category = 'sum_overturns_correct_top_token'
                        else:
                            category = 'wrong_top_token_despite_true_reference'
                        rr = dict(batch=batch, district=d, city=D[d].city, q=int(row['q']),
                                  seed=seed, truth=truth, pred=pred, category=category,
                                  true_reference=has_true, top_token=top, reference_pred=ref_pred,
                                  anchor_mass=float(arrays['anchor'][j].sum()),
                                  predicted_mass=float(arrays['count'][j,pred]),
                                  true_mass=float(arrays['count'][j,truth]),
                                  count_normalized_pred=int(normalized.argmax()))
                        district_records.append(rr)
                del m
            counts = Counter(r['category'] for r in district_records)
            counts['count_normalization_fixed'] = sum(r['pred'] != r['truth'] and r['count_normalized_pred'] == r['truth'] for r in district_records)
            counts['count_normalization_broken'] = sum(r['pred'] == r['truth'] and r['count_normalized_pred'] != r['truth'] for r in district_records)
            result[batch][d] = dict(counts)
            records.extend(district_records)
            print(batch, d, dict(counts), flush=True)
        del c, D, R, X, task
    selection = {}
    for f in sorted((base.OUT / 'selection').glob('*_modernnca.json')):
        z = json.loads(f.read_text(encoding='utf8'))
        trial = next(t for t in z['trials'] if t['hp'] == z['hp'])
        selection[f.stem] = dict(steps=z['steps'], best=z['best'], last=trial['history'][-1],
                                 hard_cap=z['steps'] == 1280)
    save(DOC / '诊断逐题.json', records)
    save(DOC / '诊断.json', dict(counts=result, replay_max_error=replay, selection=selection,
        baseline_source_sha256=digest(REPO / 'zoning/full_evidence.py'),
        caveat='Seed-case counts, not independent samples. Existing development labels; diagnostic counterfactuals not confirmation. Multi-truth cases excluded from categories.'))
    assert replay < 2e-5


if __name__ == '__main__':
    diagnose()
