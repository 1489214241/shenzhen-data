from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_shenzhen_school_membership_trees import observed_training,selected,local_features


def test_unknown_not_negative_and_row_weight_normalization():
    f=np.arange(12).reshape(2,3,2);y=np.array([[1,-1,0],[-1,1,-1.]])
    xx,yy,w=observed_training(f,y,np.array([4.,3.]),np.array([0,1]))
    np.testing.assert_array_equal(xx,[f[0,0],f[0,2],f[1,1]])
    np.testing.assert_array_equal(yy,[1,0,1]);np.testing.assert_array_equal(w,[2,2,3])


def test_new_ranking_can_replace_first_school_and_select_nonprefix_old_labels():
    q=np.array([[.1,.8,.2],[.7,.1,.8],[.1,.3,.2]])
    np.testing.assert_array_equal(selected(q,.6),[[False,True,False],[True,False,True],[False,True,False]])


def test_local_references_exclude_query_and_mark_missing():
    x=SimpleNamespace(n=2,school_n=3,ref_sites=np.array([0,1]),ref_schools=np.array([0,1]),
        self_mask=np.eye(2,dtype=bool),pair=np.array([[[0.,5.],[1.,6.]],[[1.,7.],[0.,8.]]]))
    f=local_features(x,chunk=1)
    assert f.shape==(2,3,12)
    np.testing.assert_array_equal(f[0,0,:4],0)
    np.testing.assert_array_equal(f[0,1,:2],[1,6])
    assert f[0,1,4]==np.float32(.3) and f[0,1,9]==1
    assert f[0,2,4]==15 and f[0,2,9]==0
    assert f[0,0,4]==15 and f[0,0,9]==0
