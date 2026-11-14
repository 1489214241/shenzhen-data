"""Fixed source-only ablation of absolute versus query-relative score cuts.

No new scorer, label, cardinality prior, or school-specific parameter. The
relative cut includes s iff p(s) >= alpha * max_s p(s), with top1 fallback.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
import experiment_shenzhen_relation_probe as P
from shenzhen_exact_calibration import exact_thresholds, calibration_intervals
from zoning.evaluate import bootstrap

F=P.F; R=P.R
OUT=R.ROOT/'round15_relative_decode'


def relative(p):
    p=np.asarray(p,np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Finite probability matrix required')
    return np.divide(p,p.max(1,keepdims=True),out=np.zeros_like(p),where=p.max(1,keepdims=True)>0)


def select(p,t):
    result=p>=t; empty=~result.any(1)
    result[empty,p[empty].argmax(1)]=True
    return result


def prepare():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve frozen experiment')
    c=P.verify(); R.check(P.OUT,'probability_freeze.json')
    F.write(OUT/'protocol.json',dict(sources=c['sources'],arms=['absolute','relative'],
        scorer='Exactly reused round11 pooled_relations HGB; no retraining.',
        primary='balanced_exact; one source-calibrated scalar for each decoder',
        relative_formula='p(s)/max(p) >= alpha; all-zero scores use original argmax fallback',
        source_gate=c['source_gate'],target_gold_read=False,target_polygon_read=False,
        target_cardinality_used=False,independent_validation=False,adopted=False,
        motivation='28/60 source multi sets are rank-separable but only9/60 are recovered by a global absolute threshold. Test query-specific scale without learning additional parameters.',
        limitation='Scores normalized by their maximum are relative scores, not probabilities. Ordering errors cannot be repaired.',
        research=[dict(url='https://www.mdpi.com/2076-3417/13/13/7591',
            reviewed='Introduction and abstract distinguish ranking and thresholding. This simple fixed normalization is not a reproduction of ThresNets; no F1 guarantee transferred to exact sets.')]))
    links={}
    for name in ['input_freeze.json','probability_freeze.json']:
        links[str(P.OUT/name)]=F.sha(P.OUT/name)
        for rel,h in F.read(P.OUT/name)['files'].items(): links[str(P.OUT/rel)]=h
    for path in [Path(__file__),F.REPO/'scripts/shenzhen_exact_calibration.py']:
        dest=OUT/'code'/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);links[str(path)]=F.sha(path)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def report():
    F.guard();c=verify()
    if (OUT/'source_report.json').exists(): raise FileExistsError('Preserve source outcomes')
    payload=dict(np.load(P.OUT/'probabilities/pooled_relations.npz'))
    ys=[np.load(P.OUT/'valid'/f'{d}.npz')['labels'] for d in c['sources']]
    original=[payload[d+'_valid'] for d in c['sources']]
    variants={'absolute':original,'relative':[relative(p) for p in original]}
    reports={};predictions={}
    for arm,ps in variants.items():
        calibration=exact_thresholds(ps,ys)
        chosen=calibration['selections']['balanced_exact'];gate=c['source_gate']
        capacity=sum(k>1 for lo,hi,k in calibration_intervals(ps,ys)[0])
        passed=(chosen['multi_correct']>=gate['min_multi_correct'] and chosen['single_correct']>=gate['min_single_correct'] and capacity>=gate['min_attainable_multi'])
        reports[arm]=dict(selections=calibration['selections'],attainable_multi=capacity,source_gate_passed=bool(passed))
        predictions[arm]=[select(p,chosen['threshold']) for p in ps]
        F.write(OUT/'calibration'/f'{arm}.json',calibration)
        print(arm,reports[arm],flush=True)
    prior=F.read(P.OUT/'source_report.json')['reports']['pooled_relations']
    assert reports['absolute']=={k:prior[k] for k in reports['absolute']}
    assert reports['absolute']['attainable_multi']==reports['relative']['attainable_multi']
    rows=[]
    for i,d in enumerate(c['sources']):
        v=dict(np.load(P.OUT/'valid'/f'{d}.npz'));coords=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for j,q in enumerate(v['ids']):
            y=ys[i][j];a=predictions['absolute'][i][j];b=predictions['relative'][i][j]
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in coords[q])),
                multi=bool(y.sum()>1),gold=np.flatnonzero(y).tolist(),
                before=np.flatnonzero(a).tolist(),after=np.flatnonzero(b).tolist(),
                before_ok=bool((a==y).all()),after_ok=bool((b==y).all())))
    summary={}
    for label,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
        delta=[int(r['after_ok'])-int(r['before_ok']) for r in rr]
        summary[label]=dict(n=len(rr),before_correct=sum(r['before_ok'] for r in rr),after_correct=sum(r['after_ok'] for r in rr),
            corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    F.write(OUT/'paired_source_rows.json',rows)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=summary,target_scored=False,adopted=False,
        next_action='Separately freeze target evaluation only if the unchanged source gate passes.'))
    print('PAIRED',summary['all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','report']);globals()[p.parse_args().phase]()
