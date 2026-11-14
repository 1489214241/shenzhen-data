"""Rebuild one episode per source and independently verify cached context math."""
import numpy as np
import experiment_low_reference_training as R
from zoning.low_reference_training import rebuild_view


def main():
    R.guard();c=R.verify();audit=R.read(R.OUT/'input_audit.json');out={}
    for d in c['sources']:
        old=R.E.pkl(R.M.OLD/'inputs'/f'{d}.pkl')
        labels=dict(np.load(R.M.OLD/'inputs'/f'{d}_labels.npz'))
        arr=np.load(R.REPO/'data'/d/'arrays.npz');raw=np.load(R.REPO/'data'/d/'extra_evidence.npz')['name_match']
        meta=R.read(R.REPO/'data'/d/'district.json');roads=R.E.EV.lines(meta.get('lines') or meta['city'])
        selection=next(v for v in audit['districts'][d]['source_views'] if v['view']=='episode0')
        ix=np.array(selection['relations'],int)
        assert len(set(old.ref_schools[ix]))==len(ix)
        assert not set(old.groups[old.ref_sites[ix]])&set(old.groups[labels['valid_ids']])
        v=rebuild_view(old,ix,arr['coords'],roads,raw)
        t=R.E.pkl(R.OUT/'tasks'/d/'episode0.pkl');q=t['ids'];valid=~v.self_mask[q]
        np.testing.assert_array_equal(valid,t['valid'].numpy())
        pair=v.pair[q].astype(float);anchor=v.anchor[q].astype(float);mem=v.membership[:len(ix)].numpy()
        mass=np.exp(v.group_bias.astype(float))[None]*valid
        dist=pair[:,:,v.pair_blocks['distance'].start]*300
        a=mass*np.exp(-dist/500);b=mass*np.exp(-dist/1000);a=a@mem;b=b@mem
        cr=(pair[:,:,v.crossing_cols].sum(-1)>0).astype(float)
        cross=(mass*np.exp(-dist/500)*cr)@mem
        ctx=np.stack([a/np.maximum(a.sum(1,keepdims=True),1e-12),b/np.maximum(b.sum(1,keepdims=True),1e-12),
            np.log1p(a)/4,np.log1p(b)/4,cross/np.maximum(a,1e-12),(valid@mem>0).astype(float)],-1)
        expected_a=np.concatenate([anchor,ctx],-1);expected_a[:,:,0]/=10
        expected_p=np.concatenate([pair,anchor[:,v.ref_schools],ctx[:,v.ref_schools]],-1)
        expected_p[:,:,0]/=10;expected_p[:,:,41]/=10
        delta=0.
        for k,expected in [('px',expected_p),('ax',expected_a)]:
            np.testing.assert_allclose(t[k].numpy(),expected,atol=2e-6,rtol=2e-6)
            delta=max(delta,float(np.max(np.abs(t[k].numpy()-expected))))
        out[d]=dict(training_rows=len(q),visible_relations=len(ix),max_feature_error=delta,
            boundary_rebuilt=True,context_independently_recomputed=True,source_validation_groups_excluded=True)
    R.write(R.OUT/'feature_reconstruction_audit.json',dict(passed=True,districts=out))
    print('FEATURE RECONSTRUCTION AUDIT PASSED',out,flush=True)


if __name__=='__main__':main()
