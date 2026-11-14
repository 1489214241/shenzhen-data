"""Validation-only fixed-weight diagnostic for district-ordered BatchNorm."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import run_full_evidence as base
from zoning import DEVICE, train
from run_modernnca_relation import tasks, cpu_task, device_batch, predict, save

DOC = REPO.parent / '科研推进/ModernNCA机制诊断与改进_20261112'
OUT = REPO / 'outputs/modernnca_mechanism_20261112'


@torch.no_grad()
def pooled_bn(m, tr):
    """Exact pooled population moments, layer by layer, training inputs only.

Frozen weights/dropout; unequal district sizes weighted by their token counts.
This is a diagnostic, not a claim that token-weighting is an optimal population.
"""
    m.eval()
    audit = []
    for name, bn in m.named_modules():
        if not isinstance(bn, torch.nn.BatchNorm1d):
            continue
        sums, squares, n = None, None, 0
        def collect(module, args):
            nonlocal sums, squares, n
            z = args[0].double()
            su, sq = z.sum(0), z.square().sum(0)
            sums = su if sums is None else sums + su
            squares = sq if squares is None else squares + sq
            n += len(z)
        handle = bn.register_forward_pre_hook(collect)
        for _, t in tr:
            for st in range(0, len(t['ids']), 128):
                m(device_batch(t, slice(st, st+128)))
        handle.remove()
        mean = sums / n; var = (squares / n - mean.square()).clamp_min(0)
        oldmean = bn.running_mean.clone(); oldvar = bn.running_var.clone()
        audit.append(dict(layer=name, tokens=n,
            mean_shift_sd=float(((mean-oldmean).abs() / oldvar.clamp_min(1e-8).sqrt()).mean()),
            variance_ratio_mean=float((var / oldvar.clamp_min(1e-8)).mean())))
        bn.running_mean.copy_(mean); bn.running_var.copy_(var)
    return audit


@torch.no_grad()
def ordered_bn(m, tr, reverse):
    """Change only BN running buffers with fixed identical inputs and weights."""
    m.eval()
    for bn in m.modules():
        if isinstance(bn, torch.nn.BatchNorm1d):
            bn.reset_running_stats(); bn.train()
    for _ in range(4):
        for _, t in (list(reversed(tr)) if reverse else tr):
            # Identical deterministic sample of queries per district, both orders.
            m(device_batch(t, slice(0, min(64, len(t['ids'])))))
    m.eval()


def audit():
    c, D, R, X = base.loaded(0, 1)
    results = {}
    for d in D:
        v = base.VALIDATION[d]
        choice = json.loads((base.OUT / 'selection' / f'{d}_modernnca.json').read_text(encoding='utf8'))
        tr = tasks(d, D, R, X, exclude={v})
        labels = train.uncovered_answers(D[v], X[v])
        val = cpu_task(v, sorted(labels), D, X, labels)
        m, info = base.fit('modernnca', choice['hp'], base.SEEDS[0], d, X, tr, None, steps=choice['steps'])
        P = predict(m, val); y = val['y'].numpy()
        initial = copy.deepcopy(m.state_dict()); scores = {}
        predictions = {'original': P}
        for name in ['forward', 'reverse', 'pooled']:
            m.load_state_dict(initial)
            if name == 'pooled':
                moments = pooled_bn(m, tr)
            else:
                ordered_bn(m, tr, reverse=name == 'reverse')
            predictions[name] = predict(m, val)
        for name, p in predictions.items():
            a, b = p.argmax(1) == y, P.argmax(1) == y
            scores[name] = dict(accuracy=float(a.mean()*100), nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-30)).mean()),
                changed=int((p.argmax(1) != P.argmax(1)).sum()), fixed=int((a & ~b).sum()), broken=int((b & ~a).sum()))
        assert abs(scores['original']['accuracy']/100 - choice['best']['accuracy']) < 1e-7
        order_changes = int((predictions['forward'].argmax(1) != predictions['reverse'].argmax(1)).sum())
        results[d] = dict(validation=v, n=len(y), scores=scores, pooled_moments=moments,
                          forward_reverse_prediction_changes=order_changes)
        dest = OUT / 'normalization_audit' / f'{d}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **predictions, ids=val['ids'], y=y)
        print(d, json.dumps(results[d], ensure_ascii=False), flush=True)
        save(DOC / '归一化诊断.json', results)
        del tr, m, val
        torch.cuda.empty_cache()


if __name__ == '__main__':
    audit()
