import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_logmass_decode import mixture_log_odds, odds_ratio, probability_ratio


def test_mixture_matches_arithmetic_mean_probabilities_without_clipping():
    z=np.random.default_rng(17).uniform(-8,8,size=(10,5,7))
    p=(1/(1+np.exp(-z))).mean(0)
    np.testing.assert_allclose(mixture_log_odds(z),np.log(p)-np.log1p(-p),rtol=1e-12,atol=1e-12)


def test_single_model_common_shift_invariance_and_probability_ratio_difference():
    z=np.array([[[2.,0.,-3.],[7.,3.,-1.]]])
    shifted=z+np.array([[[5.],[-2.]]])
    np.testing.assert_allclose(odds_ratio(z),odds_ratio(shifted),rtol=1e-13,atol=1e-13)
    assert not np.allclose(probability_ratio(z),probability_ratio(shifted))
    # Mixtures of heterogeneous logits need not share this shift property.


def test_saturated_positive_logits_keep_real_margins():
    z=np.array([[[1000.,999.,995.],[-1000.,-999.,-995.]]])
    np.testing.assert_allclose(odds_ratio(z),[[1,np.exp(-1),np.exp(-5)],[np.exp(-5),np.exp(-4),1]])
    np.testing.assert_allclose(mixture_log_odds(z),z[0])


def test_model_and_school_order_invariance():
    z=np.random.default_rng(4).normal(size=(3,4,5));schools=[3,1,4,0,2]
    np.testing.assert_allclose(odds_ratio(z)[:,schools],odds_ratio(z[::-1,:,schools]))


@pytest.mark.parametrize('z',[np.array([[1.]]),np.empty((0,2,3)),np.array([[[np.inf]]])])
def test_invalid_logits_rejected(z):
    with pytest.raises(ValueError):mixture_log_odds(z)
