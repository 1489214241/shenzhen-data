from types import SimpleNamespace
import numpy as np
from scripts.shenzhen_source_reference_completion import eligible_pairs,complete


def fixture():
    x=SimpleNamespace(n=4,school_n=3,ref_sites=np.array([0,1,2]),ref_schools=np.array([0,0,0]),groups=np.array(['a','b','c','d']),
        pair=np.zeros((4,3,22),np.float32),anchor=np.zeros((4,3,13),np.float32),
        pair_blocks=dict(named_reference=slice(0,1),policy=slice(1,14),agreement=slice(14,18),disagreement=slice(18,22)),
        anchor_blocks=dict(policy=slice(0,13)))
    x.anchor[:,:,:12]=np.tile([1,0,0],4)
    return x


def test_validation_partial_and_conflicting_references_are_not_completed():
    x=fixture();y=np.array([[1,1,0],[1,1,0],[0,1,1],[-1,1,-1]])
    added,conflict=eligible_pairs(x,y,[0,2,3])
    assert added==[(0,1)] and len(conflict)==1 and conflict[0]['site']==2


def test_new_pairs_keep_geometry_and_self_mask_without_changing_old_features():
    x=fixture();x.pair[:,:,0]=1
    x.anchor[:,1,:12]=np.tile([0,0,1],4)
    y=np.array([[1,1,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]])
    out,added,_=complete(x,y,[0])
    np.testing.assert_array_equal(out.pair[:,:3],x.pair)
    np.testing.assert_array_equal(out.anchor,x.anchor)
    assert added==[(0,1)] and out.pair[0,3,0]==0 and out.self_mask[0,3]
    assert not out.self_mask[1,3] and out.pair[:,:,14:18][:,3].all()
    np.testing.assert_array_equal(out.membership.numpy().sum(1),np.ones(7))
