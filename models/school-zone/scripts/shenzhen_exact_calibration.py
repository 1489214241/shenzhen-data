"""Exact global threshold calibration on complete source sets, with top1 fallback.

For a multiple-school answer, exact prediction is possible iff
max(non-answer probabilities) < threshold <= min(answer probabilities).
For singleton answers whose top1 is correct, fallback removes the upper bound.
This enumerates every exact-accuracy transition, without a truncated grid.
"""
import numpy as np


def calibration_intervals(probabilities,labels):
    intervals=[];sizes=[]
    for p,y in zip(probabilities,labels):
        p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=bool)
        if p.shape!=y.shape or p.ndim!=2:raise ValueError('Shape mismatch')
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
        if not y.any(1).all():raise ValueError('Complete nonempty labels required')
        for row,gold in zip(p,y):
            size=int(gold.sum());sizes.append(size)
            low=float(row[~gold].max()) if (~gold).any() else -1.
            high=float(row[gold].min())
            if size==1 and gold[row.argmax()]:high=1.
            start=0. if low<0 else float(np.nextafter(low,np.inf))
            end=float(np.nextafter(high,np.inf))
            if start<=high and start<=1:intervals.append((start,end,size))
    if not sizes:raise ValueError('Empty validation population')
    return intervals,np.array(sizes)


def exact_thresholds(probabilities,labels):
    intervals,sizes=calibration_intervals(probabilities,labels)
    # Include the last valid threshold before each interval ends, and 1.0
    # for the fallback-only endpoint. Higher threshold cannot increase FP.
    candidates=np.array(sorted({0.,1.}|{v for lo,hi,k in intervals for v in [lo,float(np.nextafter(hi,-np.inf))] if 0<=v<=1}))
    counts=[];groups=[]
    for multi in [False,True]:
        group=[(lo,hi) for lo,hi,k in intervals if (k>1)==multi]
        starts=sorted(lo for lo,hi in group);ends=sorted(hi for lo,hi in group)
        counts.append(np.searchsorted(starts,candidates,side='right')-np.searchsorted(ends,candidates,side='right'))
        groups.append(int(((sizes>1)==multi).sum()))
    single,multi=counts;ns,nm=groups
    if not ns or not nm:raise ValueError('Both singleton and multi-label source groups required')
    objectives={'micro_exact':single+multi,'balanced_exact':single*nm+multi*ns}
    selections={}
    for objective,score in objectives.items():
        j=int(np.flatnonzero(score==score.max())[-1]);t=float(candidates[j])
        metrics=threshold_metrics(probabilities,labels,t)
        assert metrics['correct']==int(single[j]+multi[j])
        assert metrics['single_correct']==int(single[j]) and metrics['multi_correct']==int(multi[j])
        selections[objective]=dict(threshold=t,**metrics)
    return dict(selections=selections,source_single=ns,source_multi=nm,intervals=len(intervals),
        candidate_thresholds=len(candidates),attainable_rows=len(intervals),
        candidates=[dict(threshold=float(t),single_correct=int(s),multi_correct=int(m)) for t,s,m in zip(candidates,single,multi)])


def threshold_metrics(probabilities,labels,threshold):
    n=correct=fp=fn=single_correct=multi_correct=0
    for p,y in zip(probabilities,labels):
        p=np.asarray(p,np.float64);y=np.asarray(y,bool)
        pred=p>=threshold;empty=~pred.any(1);pred[empty,p[empty].argmax(1)]=True
        ok=(pred==y).all(1);multi=y.sum(1)>1
        n+=len(y);correct+=int(ok.sum());fp+=int((pred&~y).sum());fn+=int((~pred&y).sum())
        single_correct+=int((ok&~multi).sum());multi_correct+=int((ok&multi).sum())
    return dict(n=n,correct=correct,fp=fp,fn=fn,single_correct=single_correct,multi_correct=multi_correct)
