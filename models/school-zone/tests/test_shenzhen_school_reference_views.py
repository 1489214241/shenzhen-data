from types import SimpleNamespace
import numpy as np
from scripts.shenzhen_school_reference_views import plans,boundary_support,apply_view


def fixture():
    return SimpleNamespace(pair=np.arange(3*4*8,dtype=np.float32).reshape(3,4,8),anchor=np.ones((3,3,2)),school_n=3,
        groups=np.array(['same','same','other']),ref_sites=np.array([0,0,1,2]),ref_schools=np.array([0,1,0,2]),
        pair_blocks={'learned_boundary':slice(2,8)})


def test_hidden_multi_member_does_not_create_singleton_boundary_label():
    assert boundary_support({0:[0,1],1:[0],2:[1],3:[2]},np.array([False,True,False]))=={1:[0],3:[2]}


def test_group_weights_self_mask_and_pair_features_follow_physical_removal():
    x=fixture();out=apply_view(x,[0,2,3],np.zeros((3,3,6),np.float32))
    np.testing.assert_array_equal(out.pair[:,:,:2],x.pair[:,[0,2,3],:2]);np.testing.assert_array_equal(out.anchor,x.anchor)
    np.testing.assert_allclose(out.group_bias,[-np.log(2),-np.log(2),0])
    assert out.self_mask[:2,:2].all() and not out.self_mask[:2,2].any()
    assert out.membership.shape==(6,3) and (out.membership.sum(1)==1).all()
    np.testing.assert_array_equal(x.pair,np.arange(96,dtype=np.float32).reshape(3,4,8))


def test_no_reference_view_keeps_all_school_anchors():
    x=fixture();out=apply_view(x,[],np.zeros((3,0,6),np.float32))
    assert out.pair.shape==(3,0,8) and out.self_mask.shape==(3,0)
    np.testing.assert_array_equal(out.membership.numpy(),np.eye(3));np.testing.assert_array_equal(out.anchor,x.anchor)


def test_label_independent_complementary_fixed_plans():
    p=plans(7,42);again=plans(7,42)
    for k in p:np.testing.assert_array_equal(p[k],again[k])
    for k in [0,2]:np.testing.assert_array_equal(p[f'train_{k}'],~p[f'train_{k+1}'])
    np.testing.assert_array_equal(sum(p[f'train_{k}'].astype(int) for k in range(4)),np.full(7,2))
    assert not any(np.array_equal(p[s],p[t]) for s in ['stress_0','stress_1'] for t in ['train_0','train_1','train_2','train_3'])
