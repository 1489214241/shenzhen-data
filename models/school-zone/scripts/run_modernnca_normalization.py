"""Frozen normalization-only comparison; serial GPU, selection then sealed scoring."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import run_full_evidence as base
from zoning import DEVICE, evaluate
from zoning.modernnca_normalization import LayerNormModernNCA
from audit_modernnca_normalization import pooled_bn
from run_modernnca_relation import tasks, cpu_task, predict, save, digest

OLD = base.OUT
OUT = REPO / 'outputs/modernnca_mechanism_20261112'
DOC = REPO.parent / '科研推进/ModernNCA机制诊断与改进_20261112'
original_make = base.make


def make(kind, x, start, seed):
    if kind == 'layernorm':
        return LayerNormModernNCA(x.np_, x.na, seed).to(DEVICE)
    return original_make(kind, x, start, seed)


def freeze():
    paths = [Path(__file__), REPO / 'zoning/modernnca_normalization.py',
        REPO / 'scripts/audit_modernnca_normalization.py', REPO / 'scripts/diagnose_modernnca.py',
        REPO / 'tests/test_modernnca_normalization.py']
    old = json.loads((OLD / 'frozen.json').read_text(encoding='utf8'))
    for relative, h in old['files'].items():
        assert digest(REPO / relative) == h, relative
    paths += [REPO / relative for relative in old['files']]
    paths = list(dict.fromkeys(paths))
    plan = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
        baseline='Independent full-evidence relation-input ModernNCA, not hybrid/official unchanged.',
        hypotheses=['BN running statistics depend on district order; LayerNorm removes this path.',
                    'Exact pooled training-input BN calibration isolates statistics without changing learned parameters.'],
        methods=['modernnca_cached', 'reference_cached', 'layernorm', 'pooled_bn'],
        primary='layernorm minus independently tuned ModernNCA; RetrievalScorer comparator retained.',
        calibration_control='pooled_bn: unchanged original selected hyperparameters and weights, no additional tuning; exact layerwise moments from ALL AND ONLY admitted training queries, dropout off. This control has no new optimizer fit.',
        layernorm='Replace two BatchNorm1d modules with LayerNorm; identical trainable parameter initialization, features, query network, Euclidean distance, anchors and aggregation. No RetrievalScorer weights/scores/warm start.',
        training=old['training'], seeds=base.SEEDS, grid=base.GRID,
        checkpoints=base.CHECKPOINTS, extensions=base.EXTENSIONS, validation=base.VALIDATION,
        batches=base.BATCHES, batch_size=base.BS,
        selection='Same K0 validation district, four trials, accuracy/NLL/earlier step, extensions as frozen baseline. K1 inherits. Run both variants once; no target-result-dependent new trials.',
        decision='Report both candidates vs baseline and RetrievalScorer. Stable K1 benefit requires all three batch 98.75% spatial-block CI lower bounds >0 (four candidate/comparator pairs), with no city 95% confirmed harm; K0 separate. 10000 resamples. No default replacement on development data.',
        status='Development only; target labels previously used, Hangzhou not finalized, not geographic independent confirmation. BN/order diagnosis is not novelty proof.',
        baseline_frozen_sha256=digest(OLD / 'frozen.json'),
        input_hashes=old['input_hashes'], files={str(p.relative_to(REPO)):digest(p) for p in paths},
        baseline_selection_hashes={p.name:digest(p) for p in sorted((OLD/'selection').glob('*_modernnca.json'))},
        diagnosis_hashes={p.name:digest(p) for p in [DOC/'诊断.json', DOC/'归一化诊断.json']})
    f = OUT / 'normalization_frozen.json'
    if f.exists():
        existing = json.loads(f.read_text(encoding='utf8')); plan['created'] = existing['created']
        assert json.loads(json.dumps(plan)) == existing, 'Frozen protocol/source changed'
    else:
        save(f, plan); save(DOC / '改进方案.json', plan)
        for p in paths:
            dest = OUT / 'code' / p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())


def calibrate(k, s):
    c, D, R, X = base.loaded(k, s)
    folder = OUT / f'k{k}_s{s}'
    for d in D:
        tr = tasks(d, D, R, X)
        task = cpu_task(d, np.arange(D[d].n), D, X)
        probs = []
        for seed in base.SEEDS:
            dest = folder / 'pooled_bn/seeds' / f'{d}_{seed}.npz'
            if dest.exists():
                probs.append(np.load(dest)['P']); continue
            src = OLD / f'k{k}_s{s}/modernnca/seeds' / f'{d}_{seed}.pt'
            m = original_make('modernnca', X[d], None, seed)
            m.load_state_dict(torch.load(src, map_location=DEVICE, weights_only=True)); m.eval()
            original_params = {name:p.detach().clone() for name,p in m.named_parameters()}
            moments = pooled_bn(m, tr)
            assert all(torch.equal(p, original_params[name]) for name,p in m.named_parameters())
            P = predict(m, task)
            dest.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(dest, P=P)
            torch.save(m.cpu().state_dict(), dest.with_suffix('.pt'))
            save(dest.with_suffix('.json'), dict(source=str(src), source_sha256=digest(src), moments=moments,
                unchanged_trainable_parameters=True, calibration_ids={dd:t['ids'].tolist() for dd,t in tr}))
            probs.append(P); del m
            print('CALIBRATED', k, s, d, seed, flush=True)
        dest = folder / 'pooled_bn/probs' / f'{d}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(dest, probs=np.stack(probs))
        del tr, task
        torch.cuda.empty_cache()


def seal():
    hashes = {}
    for k,s in base.BATCHES:
        for kind in ['layernorm','pooled_bn']:
            files = sorted((OUT/f'k{k}_s{s}'/kind/'probs').glob('*.npz'))
            assert len(files) == 8
            for f in files:
                hashes[str(f.relative_to(OUT))] = digest(f)
    save(OUT/'sealed_predictions.json', hashes)


def adjusted(rows, other, seed):
    rng = np.random.default_rng(seed); num = np.zeros(10000); den = np.zeros(10000)
    delta = np.array([float(a['ok'])-float(b['ok']) for a,b in zip(rows,other)])
    for d in sorted({r['d'] for r in rows}):
        ix = [i for i,r in enumerate(rows) if r['d']==d]
        keys = sorted({rows[i]['block'] for i in ix}); lookup={v:j for j,v in enumerate(keys)}
        bi = np.array([lookup[rows[i]['block']] for i in ix])
        n = np.bincount(bi); ds=np.bincount(bi,weights=delta[ix])
        draw=rng.integers(0,len(keys),(10000,len(keys)))
        num+=ds[draw].sum(1); den+=n[draw].sum(1)
    return (np.quantile(num/den,[.00625,.99375])*100).tolist()


def score():
    hashes=json.loads((OUT/'sealed_predictions.json').read_text(encoding='utf8'))
    assert all(digest(OUT/f)==h for f,h in hashes.items())
    results={}
    for k,s in base.BATCHES:
        c,D,R,X=base.loaded(k,s); scored={}; whole={}; seeds={}; cases=[]
        methods=['modernnca','reference','layernorm','pooled_bn']; predictions={}
        for kind in methods:
            root=OLD if kind in ['modernnca','reference'] else OUT
            scored[kind]=[]; whole[kind]=[]; seed_rows=[[],[],[]]; predictions[kind]={}
            for d in D:
                P=np.load(root/f'k{k}_s{s}'/kind/'probs'/f'{d}.npz')['probs']
                rows=evaluate.eval_rows(D[d],R[d].known)
                scored[kind]+=evaluate.correct(P.mean(0),D[d],rows)
                predictions[kind][d]=P.mean(0).argmax(1)
                for j,p in enumerate(P): seed_rows[j]+=evaluate.correct(p,D[d],rows)
                rr=evaluate.overall(P.mean(0),D[d],R[d])
                whole[kind]+=[row for q,row in zip(sorted(D[d].answers),rr) if R[d].kind.get(q)!='verified']
            seeds[kind]=[evaluate.summary(rr) for rr in seed_rows]
        comparisons={}
        for kind in ['layernorm','pooled_bn']:
            for comparator in ['modernnca','reference']:
                key=f'{kind}-{comparator}'
                comparisons[key]=evaluate.compare(scored[kind],scored[comparator],dict(c['evaluate'],draws=10000))
                comparisons[key]['adjusted_ci']=adjusted(scored[kind],scored[comparator],c['evaluate']['bootstrap_seed'])
        for d in D:
            cases += [dict(district=d,**row,predictions={m:int(predictions[m][d][row['q']]) for m in methods})
                      for row in evaluate.eval_rows(D[d],R[d].known)]
        save(OUT/f'k{k}_s{s}'/'scored_cases.json',cases)
        results[f'k{k}_s{s}']=dict(model={m:evaluate.summary(r) for m,r in scored.items()},
            overall_excluding_verified={m:evaluate.summary(r) for m,r in whole.items()},
            per_seed=seeds,comparisons=comparisons)
        print('SCORED',k,s,{m:results[f'k{k}_s{s}']['model'][m]['两城']['accuracy'] for m in methods},flush=True)
    verdict={key:dict(confirmed=all(results[f'k1_s{s}']['comparisons'][key]['adjusted_ci'][0]>0 for s in [1,2,3])
        and not any(results[f'k1_s{s}']['comparisons'][key][city]['ci'][1]<0 for s in [1,2,3] for city in ['长沙','杭州']))
        for key in ['layernorm-modernnca','layernorm-reference','pooled_bn-modernnca','pooled_bn-reference']}
    save(DOC/'结果.json',results); save(DOC/'判定.json',verdict)
    print('VERDICT',verdict,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','select','fit','score','all'])
    args=parser.parse_args(); freeze()
    base.OUT=OUT; base.make=make
    if args.phase in ['select','all']:
        base.select(methods=['layernorm'])
    if args.phase in ['fit','all']:
        for k,s in base.BATCHES:
            base.run(k,s,methods=['layernorm']); calibrate(k,s)
        seal()
    if args.phase in ['score','all']:
        score()


if __name__=='__main__':
    main()
