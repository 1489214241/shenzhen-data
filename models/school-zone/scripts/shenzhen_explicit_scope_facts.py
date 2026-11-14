"""Three-valued scope reasoning from affirmative exact name matches only.

Non-match remains unknown. No geographical, predicted-school or label inputs.
An explicit exclusion can make a branch false; a school is false only if the
entire Boolean expression is false. Unknown alternatives block exclusion.
"""
import numpy as np
from sympy import Symbol, And, Or, Not
from sympy.logic.inference import satisfiable


def evaluate(ast, matches, n):
    op=ast['op']
    if op=='named':return matches[ast['text']].copy()
    if op not in ('and','or','minus'):return np.full(n,-1,np.int8)
    values=[evaluate(c,matches,n) for c in ast['children']]
    if op=='minus':
        if len(values)!=2:raise ValueError('Malformed exclusion')
        values[1]=np.where(values[1]<0,-1,1-values[1]).astype(np.int8);op='and'
    v=np.stack(values)
    if op=='and':return np.where((v==0).any(0),0,np.where((v==1).all(0),1,-1)).astype(np.int8)
    return np.where((v==1).any(0),1,np.where((v==0).all(0),0,-1)).astype(np.int8)


def prove(ast,true_terms,wanted):
    """Independent SAT proof under only affirmative name membership facts."""
    symbols={}
    def atom(key):
        if key not in symbols:symbols[key]=Symbol('atom_'+str(len(symbols)))
        return symbols[key]
    def convert(a,path=()):
        op=a['op']
        if op=='and':return And(*(convert(c,path+(i,)) for i,c in enumerate(a['children'])))
        if op=='or':return Or(*(convert(c,path+(i,)) for i,c in enumerate(a['children'])))
        if op=='minus':return And(convert(a['children'][0],path+(0,)),Not(convert(a['children'][1],path+(1,))))
        return atom(('named',a['text'])) if op=='named' else atom(('opaque','/'.join(map(str,path))))
    expression=convert(ast)
    premises=And(*(atom(('named',t)) for t in true_terms))
    assert satisfiable(premises) is not False
    counterexample=And(premises,Not(expression) if wanted==1 else expression)
    assert satisfiable(counterexample) is False,'Claim is not entailed by observed names'
    return dict(valid=True,conclusion='inside' if wanted==1 else 'outside',
                true_named_terms=sorted(true_terms),unsatisfiable_counterexample=True,
                unmatched_names_are_unknown=True,solver='sympy.logic.satisfiable')
