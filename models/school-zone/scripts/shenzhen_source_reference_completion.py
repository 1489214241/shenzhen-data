"""Add legal source-training co-labels only at existing reference locations."""
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch


def eligible_pairs(x,labels,train_ids):
    allowed=set(map(int,train_ids));by_site=defaultdict(set)
    for q,s in zip(x.ref_sites,x.ref_schools):by_site[int(q)].add(int(s))
    added=[];conflicts=[]
    for q,known in sorted(by_site.items()):
        if q not in allowed:continue
        y=labels[q]
        if not (y>=0).all() or (y==1).sum()<2:continue
        true=set(map(int,np.flatnonzero(y==1)))
        if not known<=true:
            conflicts.append(dict(site=q,old_positive_schools=sorted(known),source_training_schools=sorted(true)))
            continue
        added.extend((q,s) for s in sorted(true-known))
    return added,conflicts


def complete(x,labels,train_ids):
    added,conflicts=eligible_pairs(x,labels,train_ids);out=deepcopy(x)
    if not added:return out,added,conflicts
    reference_column={int(q):j for j,q in enumerate(x.ref_sites)}
    states=x.anchor[:,:,x.anchor_blocks['policy']][...,:12].reshape(x.n,x.school_n,4,3).argmax(-1)-1
    columns=[]
    for q,s in added:
        a=x.pair[:,reference_column[q],:].copy()
        a[:,x.pair_blocks['named_reference']]=0
        a[:,x.pair_blocks['policy']]=x.anchor[:,s,x.anchor_blocks['policy']]
        qst=states[:,s];rst=states[q,s]
        a[:,x.pair_blocks['agreement']]=((qst>=0)&(rst>=0)&(qst==rst)).astype(np.float32)
        a[:,x.pair_blocks['disagreement']]=((qst>=0)&(rst>=0)&(qst!=rst)).astype(np.float32)
        columns.append(a)
    out.pair=np.concatenate([x.pair,np.stack(columns,axis=1)],axis=1)
    out.ref_sites=np.r_[x.ref_sites,[q for q,s in added]].astype(int)
    out.ref_schools=np.r_[x.ref_schools,[s for q,s in added]].astype(int)
    _,inv,count=np.unique(out.groups[out.ref_sites],return_inverse=True,return_counts=True)
    out.group_bias=-np.log(count[inv]).astype(np.float32)
    out.self_mask=out.groups[:,None]==out.groups[out.ref_sites][None]
    out.membership=torch.nn.functional.one_hot(torch.tensor(np.r_[out.ref_schools,np.arange(out.school_n)],dtype=torch.long),out.school_n).float()
    return out,added,conflicts
