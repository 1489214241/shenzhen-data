import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_exact_calibration import exact_thresholds,threshold_metrics


def test_upper_grid_edge_is_not_silently_capped():
    p=[np.array([[.99,.95],[.98,.97]])];y=[np.array([[1,0],[1,1]],bool)]
    result=exact_thresholds(p,y)
    assert result['selections']['micro_exact']['threshold']==.97
    assert result['selections']['micro_exact']['correct']==2


@pytest.mark.parametrize('seed',range(10))
def test_interval_solver_matches_exhaustive_prediction_breakpoints(seed):
    rng=np.random.default_rng(seed)
    p=[rng.choice([0.,.1,.3,.7,.9,1.],size=(20,4)),rng.random((5,3))]
    y=[rng.random(a.shape)>.5 for a in p]
    for a in y:a[~a.any(1),0]=True
    y[0][0]=[1,0,0,0];y[0][1]=[1,1,0,0]
    thresholds=sorted({0.,1.}|{float(v) for a in p for v in a.ravel()}|
        {float(np.nextafter(v,np.inf)) for a in p for v in a.ravel() if v<1})
    result=exact_thresholds(p,y);ns=result['source_single'];nm=result['source_multi']
    candidates=[dict(threshold=t,**threshold_metrics(p,y,t)) for t in thresholds]
    for name in ['micro_exact','balanced_exact']:
        def score(r):return r['correct'] if name=='micro_exact' else r['single_correct']*nm+r['multi_correct']*ns
        best=max(candidates,key=lambda r:(score(r),-r['fp'],r['threshold']))
        assert result['selections'][name]==best


def test_tied_nongold_score_makes_multi_exact_impossible():
    p=[np.array([[.8,.8,.8],[.9,.2,.1]])];y=[np.array([[1,1,0],[1,0,0]],bool)]
    r=exact_thresholds(p,y)
    assert r['attainable_rows']==1 and r['selections']['micro_exact']['threshold']==1.


def test_invalid_source_labels_rejected():
    with pytest.raises(ValueError):exact_thresholds([np.ones((2,2))],[np.zeros((2,2),bool)])
