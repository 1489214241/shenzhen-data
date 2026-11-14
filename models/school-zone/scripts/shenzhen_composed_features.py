"""Expose the native geographic-clause union without discarding unknowns.

Keeps K1--K8 and X2. Pure named-list clauses never become positive area
features. Negative area union is unknown if non-area alternatives exist.
No model labels, school IDs or target polygons are inputs to this computation.
"""
from collections import defaultdict
import numpy as np
from zoning import rules


def composed_scope(O,A,S):
    n,m=O['n'],O['school_n'];result=np.full((n,m),-1,np.int8)
    by=defaultdict(list)
    for c in O['clauses']:by[c['school']].append(c)
    audit=[]
    for school in range(m):
        cls=by[school];area=[c for c in cls if c['kind']=='area']
        values=[rules.clause(c,O,A,S)[0] for c in area]
        unknown=[np.full(n,-1,np.int8)] if S['nonarea_as_unknown'] and len(area)<len(cls) else []
        union=rules.OR(values+unknown,n)
        own=[q for q,ss in O['name_known'].items() if school in ss]
        explicit_alt=any(c['kind']!='area' for c in cls)
        blocked=bool([q for q in own if union[q]==0 and not O['generic'][q]] and not explicit_alt) and rules.on(S,'X2')
        if blocked:union[:]=-1
        # Outside listed area branches is not outside an independent named branch.
        if explicit_alt:union[union==0]=-1
        result[:,school]=union
        audit.append(dict(school_index=school,area_clauses=len(area),nonarea_alternatives=explicit_alt,blocked=blocked,
            positive=int((union==1).sum()),negative=int((union==0).sum()),unknown=int((union<0).sum())))
    return result,audit


def encode(states):
    if not np.isin(states,[-1,0,1]).all():raise ValueError('Invalid three-valued scope state')
    return (states[...,None]==np.array([-1,0,1])).astype(np.float32)
