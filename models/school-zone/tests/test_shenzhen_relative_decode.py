import numpy as np
from scripts.experiment_shenzhen_relative_decode import relative,select


def test_query_scale_and_ranking():
    p=np.array([[.8,.6,.1],[.4,.3,.05]])
    a=relative(p)
    np.testing.assert_array_equal(a[0],a[1])
    np.testing.assert_array_equal(select(a,.7),[[True,True,False],[True,True,False]])


def test_zero_fallback_and_ties():
    a=relative(np.array([[0.,0.,0.],[.4,.4,.1]]))
    np.testing.assert_array_equal(select(a,1),[[True,False,False],[True,True,False]])


def test_invalid_scores():
    import pytest
    for p in [np.array([[np.nan,.1]]),np.array([[-.1,.3]])]:
        with pytest.raises(ValueError): relative(p)
