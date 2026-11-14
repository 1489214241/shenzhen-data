"""No-label school-set decoder for an already frozen school posterior.

This is a deterministic pushforward through existing policy rules, not a
calibrated joint multilabel posterior. Identical whole sets merge; intersecting
sets do not merge. Unknown household/geometry conditions are not invented.
"""
import numpy as np


def set_partition(root_sets):
    groups={}
    for school,labels in enumerate(root_sets):
        key=tuple(sorted(set(labels)))
        if not key:raise ValueError('Empty policy root set')
        groups.setdefault(key,[]).append(school)
    keys=list(groups)
    membership=np.zeros((len(root_sets),len(keys)),np.float64)
    for k,key in enumerate(keys):membership[groups[key],k]=1
    assert np.all(membership.sum(1)==1)
    return keys,membership


def decode(probabilities,root_sets):
    p=np.asarray(probabilities,np.float64)
    if p.ndim!=2 or p.shape[1]!=len(root_sets):raise ValueError('School order mismatch')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid frozen probabilities')
    keys,membership=set_partition(root_sets)
    masses=p@membership
    old_roots=p.argmax(1)
    old_sets=[tuple(sorted(set(root_sets[j]))) for j in old_roots]
    best=[]
    for q,row in enumerate(masses):
        best_index=int(row.argmax())
        # Exact/numerical tie retains old decision, never favors a larger set.
        prior=keys.index(old_sets[q])
        if np.isclose(row[prior],row[best_index],atol=1e-12,rtol=0):best_index=prior
        best.append(keys[best_index])
    return best,masses,keys
