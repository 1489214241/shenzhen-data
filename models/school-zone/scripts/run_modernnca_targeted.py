"""Frozen targeted ModernNCA-LN experiments; one GPU process, no target selection."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/'scripts'))
import run_full_evidence as base
from run_modernnca_relation import digest, save
from zoning import DEVICE, evaluate
from zoning.modernnca_targeted import TargetedModernNCA

OLD = REPO/'outputs/full_evidence_20261112'
LN = REPO/'outputs/modernnca_mechanism_20261112'
OUT = REPO/'outputs/modernnca_targeted_20261113'
DOC = REPO.parent/'科研推进/ModernNCA针对性改进_20261113'
METHODS = ['schoolquery', 'adaptivegate', 'constantgate']
PAIRS = [(m,c) for m in METHODS for c in ['layernorm','reference']] + [('adaptivegate','constantgate')]


def make(kind, x, start, seed):
    return TargetedModernNCA(x.np_, x.na, seed, kind).to(DEVICE)


def freeze():
    old = json.loads((LN/'normalization_frozen.json').read_text(encoding='utf8'))
    for p,h in old['files'].items(): assert digest(REPO/p)==h, p
    paths = [REPO/p for p in old['files']] + [Path(__file__), REPO/'zoning/modernnca_targeted.py',
        REPO/'tests/test_modernnca_targeted.py']
    plan = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), methods=METHODS, pairs=PAIRS,
        baseline='ModernNCA full-evidence relation adaptation, LayerNorm variant; not unmodified ModernNCA.',
        hypotheses={
            'schoolquery':'Residual query per school from global query, school prototype, anchor, log mass; tests whether school-specific context helps Euclidean matching. FEAT paper Eq5 inspired; not FEAT replication.',
            'adaptivegate':'Residual global reference-channel log-mass from query/context and 7 count/confidence/agreement summaries. End-to-end input-dependent probability fusion; inspired by GMU gating principle, not its hidden-unit architecture.',
            'constantgate':'Same joint training with one learned reference-channel log-mass offset, no input-dependent residual. Calibration control.'},
        architecture='LN encoder 32/64, same initial encoder/query/temperature and RNG state as baseline. Zero residual recovers baseline. Shared school function, no school IDs; retain every legal reference and every school anchor.',
        training=old['training'], seeds=base.SEEDS, grid=base.GRID, checkpoints=base.CHECKPOINTS,
        extensions=base.EXTENSIONS, batch_size=base.BS, batches=base.BATCHES, validation=base.VALIDATION,
        selection='Each of 3 variants: all 4 lr/reg trials on K0 validation district, accuracy then NLL then earlier step, same extensions. No early variant screening; all 3 run 4 batches x 8 districts x 3 seeds. K1 inherits selection. No target-driven extra combinations.',
        decision='Seven predeclared paired contrasts; 10000 stratified spatial-block resamples. Bonferroni two-sided level 1-.05/7 per contrast within each batch; all three K1 adjusted lower bounds >0 AND no city ordinary 95% confirmed harm required for stable development benefit. K0 separate. Adaptive gate vs constant gate is required to claim value of input dependence. No geographic independence or novelty inferred.',
        status='pre_v1 reused development cases. Diagnosis used K0_s1/K1_s1 target answers before design. LN baseline chosen after past development results. Hangzhou answers unfinished; not independent confirmation. Do not replace default model.',
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in dict.fromkeys(paths)},
        baseline_manifest_sha256=digest(LN/'normalization_frozen.json'),
        input_hashes=old['input_hashes'],
        diagnostic_hashes={p.name:digest(p) for p in DOC.glob('基线诊断*.json')},
        literature_hashes={p.name:digest(p) for p in (DOC/'papers').glob('*.pdf')},
        cached_hashes={str(p.relative_to(REPO)):digest(p) for root,kind in [(LN,'layernorm'),(OLD,'reference')]
            for k,s in base.BATCHES for p in (root/f'k{k}_s{s}'/kind/'probs').glob('*.npz')})
    f=OUT/'frozen.json'
    if f.exists():
        prev=json.loads(f.read_text(encoding='utf8')); plan['created']=prev['created']
        assert json.loads(json.dumps(plan))==prev, 'Frozen experiment changed'
    else:
        save(f,plan); save(DOC/'方案.json',plan)
        for p in dict.fromkeys(paths):
            dest=OUT/'code'/p.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(p.read_bytes())
    for p,h in plan['cached_hashes'].items(): assert digest(REPO/p)==h


def seal():
    files=[f for k,s in base.BATCHES for m in METHODS for f in (OUT/f'k{k}_s{s}'/m/'probs').glob('*.npz')]
    assert len(files)==96
    save(OUT/'sealed_predictions.json',{str(p.relative_to(OUT)):digest(p) for p in files})


def interval(a,b,seed):
    # evaluate.correct preserves order but does not carry query IDs; score builds
    # both rows from the very same eval_rows and records IDs in scored_cases.
    assert [(r['d'],r['city'],r['block']) for r in a]==[(r['d'],r['city'],r['block']) for r in b]
    delta=np.array([float(x['ok'])-float(y['ok']) for x,y in zip(a,b)])
    rng=np.random.default_rng(seed); num=np.zeros(10000); den=np.zeros(10000)
    for d in sorted({r['d'] for r in a}):
        ix=[i for i,r in enumerate(a) if r['d']==d]
        keys=sorted({a[i]['block'] for i in ix}); lookup={v:j for j,v in enumerate(keys)}
        bi=np.array([lookup[a[i]['block']] for i in ix]); n=np.bincount(bi); ds=np.bincount(bi,weights=delta[ix])
        draw=rng.integers(0,len(keys),(10000,len(keys)))
        num+=ds[draw].sum(1); den+=n[draw].sum(1)
    alpha=.05/len(PAIRS)
    return (np.quantile(num/den,[alpha/2,1-alpha/2])*100).tolist()


def root_for(m): return LN if m=='layernorm' else OLD if m=='reference' else OUT


def score():
    sealed=json.loads((OUT/'sealed_predictions.json').read_text(encoding='utf8'))
    assert len(sealed)==96 and all(digest(OUT/p)==h for p,h in sealed.items())
    result={}
    for k,s in base.BATCHES:
        c,D,R,X=base.loaded(k,s); scored={}; whole={}; seedrows={}; predictions={}
        for m in ['layernorm','reference']+METHODS:
            scored[m]=[]; whole[m]=[]; seedrows[m]=[[],[],[]]; predictions[m]={}
            for d in D:
                P=np.load(root_for(m)/f'k{k}_s{s}'/m/'probs'/f'{d}.npz')['probs']
                rows=evaluate.eval_rows(D[d],R[d].known)
                scored[m]+=evaluate.correct(P.mean(0),D[d],rows)
                for j,p in enumerate(P): seedrows[m][j]+=evaluate.correct(p,D[d],rows)
                rr=evaluate.overall(P.mean(0),D[d],R[d])
                whole[m]+=[row for q,row in zip(sorted(D[d].answers),rr) if R[d].kind.get(q)!='verified']
                predictions[m][d]=P.mean(0).argmax(1)
        comp={}
        for a,b in PAIRS:
            v=evaluate.compare(scored[a],scored[b],dict(c['evaluate'],draws=10000))
            v['adjusted_ci']=interval(scored[a],scored[b],c['evaluate']['bootstrap_seed']); comp[f'{a}-{b}']=v
        cases=[dict(district=d,**r,predictions={m:int(predictions[m][d][r['q']]) for m in predictions})
            for d in D for r in evaluate.eval_rows(D[d],R[d].known)]
        save(OUT/f'k{k}_s{s}'/'scored_cases.json',cases)
        result[f'k{k}_s{s}']=dict(model={m:evaluate.summary(r) for m,r in scored.items()},
            overall_excluding_verified={m:evaluate.summary(r) for m,r in whole.items()},
            per_seed={m:[evaluate.summary(rr) for rr in rs] for m,rs in seedrows.items()},comparisons=comp)
        print('SCORED',k,s,{m:v['两城']['accuracy'] for m,v in result[f'k{k}_s{s}']['model'].items()},flush=True)
        del c,D,R,X; gc.collect()
    verdict={f'{a}-{b}':dict(confirmed_development_benefit=
        all(result[f'k1_s{s}']['comparisons'][f'{a}-{b}']['adjusted_ci'][0]>0 for s in [1,2,3]) and
        not any(result[f'k1_s{s}']['comparisons'][f'{a}-{b}'][city]['ci'][1]<0 for s in [1,2,3] for city in ['长沙','杭州'])) for a,b in PAIRS}
    save(DOC/'结果.json',result); save(DOC/'判定.json',verdict); print('VERDICT',verdict,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['freeze','select','fit','score','all'])
    args=p.parse_args(); freeze(); base.OUT=OUT; base.make=make
    if args.phase in ['select','all']: base.select(methods=METHODS)
    if args.phase in ['fit','all']:
        for k,s in base.BATCHES: base.run(k,s,methods=METHODS)
        seal()
    if args.phase in ['score','all']: score()


if __name__=='__main__': main()
