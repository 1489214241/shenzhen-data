"""Independent eligibility, geometry, masking and reference-support audit."""
from collections import defaultdict
import numpy as np
import torch
import experiment_shenzhen_source_multireferences as X
from zoning import evidence as E

F=X.F


def support(x,y,split):
    refs=defaultdict(set)
    for q,s in zip(x.ref_sites,x.ref_schools):refs[str(x.groups[q])].add(int(s))
    ids=y['train_ids'] if split=='train' else y['valid_ids']
    labels=y['labels'][ids] if split=='train' else y['valid_labels']
    n=individual=together=0
    for q,row in zip(ids,labels):
        if not (row>=0).all() or (row==1).sum()<2:continue
        truth=set(np.flatnonzero(row==1));available=[ss for g,ss in refs.items() if g!=str(x.groups[q])]
        n+=1;individual+=int(truth<=set().union(*available));together+=int(any(truth<=ss for ss in available))
    return dict(n=n,individual=individual,together=together)


def main():
    F.guard();c=X.verify();bx,by=X.load(c,'control');report={};links={}
    for arm in X.ARMS:
        xs,ys=X.load(c,arm);details={}
        for d in c['sources']:
            x,y=xs[d],ys[d];old=bx[d];nold=len(old.ref_sites)
            np.testing.assert_array_equal(x.pair[:,:nold],old.pair);np.testing.assert_array_equal(x.anchor,old.anchor)
            dist=X.B.source_district(d,links);ri=x.ref_sites[nold:];rs=x.ref_schools[nold:]
            orig_refs=defaultdict(set)
            for q,s in zip(old.ref_sites,old.ref_schools):orig_refs[int(q)].add(int(s))
            legal=[]
            for q in y['train_ids']:
                row=y['labels'][q]
                if not (row>=0).all() or (row==1).sum()<2 or q in dist.not_reference:continue
                truth=set(np.flatnonzero(row==1))
                if not orig_refs[int(q)]<=truth:continue
                if arm=='complete_existing' and q not in set(old.ref_sites):continue
                if arm!='control':legal.extend((int(q),int(s)) for s in sorted(truth-orig_refs[int(q)]))
            assert sorted(zip(ri,rs))==sorted(legal)
            valid_groups=set(x.groups[y['valid_ids']]);assert not set(x.groups[ri])&valid_groups
            assert not set(ri)&dist.not_reference
            for q,s in zip(ri,rs):assert y['labels'][q,s]==1 and q in y['train_ids']
            np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
            expected_mem=np.eye(x.school_n,dtype=np.float32)[np.r_[x.ref_schools,np.arange(x.school_n)]]
            np.testing.assert_array_equal(x.membership.cpu().numpy(),expected_mem)
            _,inv,count=np.unique(x.groups[x.ref_sites],return_inverse=True,return_counts=True)
            np.testing.assert_array_equal(x.group_bias,-np.log(count[inv]).astype(np.float32))
            if len(ri):
                # Recompute geometry from raw projected coordinates, not any target school polygon.
                delta=dist.coords[ri][None]-dist.coords[:,None];distance=np.linalg.norm(delta,axis=-1)
                encoded=np.stack([np.minimum(distance/300,50),np.log1p(distance/100),np.clip(delta[...,0]/1000,-5,5),np.clip(delta[...,1]/1000,-5,5),np.exp(-distance/500)],-1).astype(np.float32)
                np.testing.assert_array_equal(x.pair[:,nold:,x.pair_blocks['distance']],encoded)
                np.testing.assert_array_equal(x.pair[:,nold:,x.pair_blocks['policy']],x.anchor[:,rs,x.anchor_blocks['policy']])
                assert not x.pair[:,nold:,x.pair_blocks['named_reference']].any()
                st=x.anchor[:,:,x.anchor_blocks['policy']][...,:12].reshape(x.n,x.school_n,4,3).argmax(-1)-1
                a,b=st[:,rs],st[ri,rs][None]
                for block,v in [('agreement',(a>=0)&(b>=0)&(a==b)),('disagreement',(a>=0)&(b>=0)&(a!=b))]:np.testing.assert_array_equal(x.pair[:,nold:,x.pair_blocks[block]],v.astype(np.float32))
                new=sorted(set(ri)-set(old.ref_sites));positions=[i for i,q in enumerate(ri) if q in new]
                if new:
                    ls=E.lines(dist.lines_key or dist.city);cross,within=E.crossing(ls,dist.coords,dist.coords[ri[positions]])
                    np.testing.assert_array_equal(x.pair[:,nold+np.array(positions),x.pair_blocks['crossing']],cross)
                    np.testing.assert_array_equal(x.pair[:,nold+np.array(positions),x.pair_blocks['within_2km']],within)
                cn=dist.community_name;strict=dist.community_strict;same=(cn[:,None]==cn[ri][None])&(cn[:,None]!='');both=strict[:,None]&strict[ri][None]
                np.testing.assert_array_equal(x.pair[:,nold:,x.pair_blocks['same_community']],np.stack([same&both,(~same)&both],-1).astype(np.float32))
            details[d]=dict(added_pairs=len(ri),new_reference_sites=len(set(ri)-set(old.ref_sites)),
                train=support(x,y,'train'),valid=support(x,y,'valid'),eligibility_and_geometry_verified=True)
        totals={split:{k:sum(v[split][k] for v in details.values()) for k in ['n','individual','together']} for split in ['train','valid']}
        report[arm]=dict(districts=details,totals=totals);print(arm,totals,flush=True)
    dest=X.OUT/'independent_input_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    F.write(dest,dict(reports=report,target_gold_read=False,target_polygon_read=False,
        source_validation_labels_used_only_for_diagnostic_counts=True,no_validation_labels_added_to_inputs=True,
        caveat='Coobserved reference positives remain partial observations, not proof of complete school membership.',
        source_hashes={str(p):F.sha(p) for p in [X.OUT/'input_freeze.json',__file__]}))


if __name__=='__main__':main()
