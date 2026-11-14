"""Replay all school-mask views, including geometry, without target answers."""
import numpy as np
import experiment_shenzhen_school_reference_mask as D
from shenzhen_batched_reference_boundaries import compute, LEGACY
from zoning import rules

F=D.F; R=D.R; X=D.X; V=D.V


def main():
    F.guard(); c=D.verify(); R.check(V.OUT,'manifest.json')
    dest=R.ROOT/'round18_input_replay_audit.json'
    if dest.exists():raise FileExistsError('Preserve input audit')
    xs,ys=X.load(c,'all_multi_training'); _,oldys=X.load(c,'control')
    cfg=F.read(X.B.CACHE/'manifest.json')['config']; report={}
    for di,d in enumerate(c['sources']):
        x=xs[d]; dist=X.B.source_district(d,{})
        assert not dist.answers and not dist.train_answers
        for key in ys[d]:np.testing.assert_array_equal(ys[d][key],oldys[d][key])
        # Generate the three partitions directly, independently of plans().
        rng=np.random.default_rng(20261118+di); plans={}
        for partition in range(3):
            order=rng.permutation(x.school_n); h=np.zeros(x.school_n,bool)
            h[order[:x.school_n//2]]=True
            prefix='train' if partition<2 else 'stress'; i=2*partition if partition<2 else 0
            plans[f'{prefix}_{i}']=h; plans[f'{prefix}_{i+1}']=~h
        known=rules.run(dist,**cfg['rules']).known
        values=compute(dist,known,x.ref_sites,plans)
        np.testing.assert_array_equal(values[0],x.pair[:,:,x.pair_blocks['learned_boundary']])
        details={}
        for i,(key,hidden) in enumerate(plans.items(),1):
            z=dict(np.load(V.OUT/d/f'{key}.npz')); cols=np.flatnonzero(~hidden[x.ref_schools])
            np.testing.assert_array_equal(z['hidden_schools'],hidden)
            np.testing.assert_array_equal(z['columns'],cols)
            np.testing.assert_array_equal(z['boundary'],values[i][:,cols])
            a=D.view(x,d,key)
            assert not hidden[a.ref_schools].any()
            np.testing.assert_array_equal(a.ref_sites,x.ref_sites[cols])
            np.testing.assert_array_equal(a.ref_schools,x.ref_schools[cols])
            for field in ['anchor','y','supervised','groups']:
                np.testing.assert_array_equal(getattr(a,field),getattr(x,field))
            assert a.school_n==x.school_n
            np.testing.assert_array_equal(a.membership.cpu().numpy(),np.eye(x.school_n,dtype=np.float32)[np.r_[a.ref_schools,np.arange(x.school_n)]])
            np.testing.assert_array_equal(a.self_mask,x.groups[:,None]==x.groups[a.ref_sites][None])
            _,inv,cnt=np.unique(x.groups[a.ref_sites],return_inverse=True,return_counts=True)
            np.testing.assert_array_equal(a.group_bias,-np.log(cnt[inv]).astype(np.float32))
            for block,sl in x.pair_blocks.items():
                expected=values[i][:,cols] if block=='learned_boundary' else x.pair[:,cols,sl]
                np.testing.assert_array_equal(a.pair[:,:,sl],expected)
            details[key]=dict(reference_pairs=len(cols),hidden_schools=int(hidden.sum()),
                changed_boundary_cells=int((values[i][:,cols]!=values[0][:,cols]).sum()))
        report[d]=details; print('INPUT_REPLAYED',d,flush=True)
    for d in c['targets']:
        assert not (ys[d]['labels']==0).any() and len(ys[d]['valid_ids'])==0
        assert (xs[d].y==-1).all() and len(xs[d].supervised)==0
    F.write(dest,dict(districts=report,views=48,full_and_masked_geometry_bitwise_replayed=True,
        school_mask_plans_recomputed=True,source_labels_unchanged=True,target_positive_only=True,
        target_gold_read=False,target_polygon_read=False,anchors_groups_self_masks_and_all_pair_blocks_checked=True,
        historical_geometry_used_without_setup_or_labels=True,
        dependencies={str(p):F.sha(p) for p in [__file__,LEGACY,V.OUT/'manifest.json',D.OUT/'input_freeze.json']}))
    print('ALL48_INPUT_VIEWS_REPLAYED',flush=True)


if __name__=='__main__':main()
