"""Independent numerical replay plus exhaustive Boolean scope-proof checks."""
import itertools
import numpy as np
import experiment_modernnca_scope_cells as X
from audit_modernnca_conditional_sets import independent_pick


def frozen(x):return tuple(frozen(v) for v in x) if isinstance(x,(list,tuple)) else x


def atoms(expr):
    if expr[0] in ('and','or','minus'):return set().union(*(atoms(e) for e in expr[1:]))
    return {frozen(expr)}


def truth(expr,assignment):
    if expr[0]=='or':return any(truth(e,assignment) for e in expr[1:])
    if expr[0]=='and':return all(truth(e,assignment) for e in expr[1:])
    if expr[0]=='minus':return truth(expr[1],assignment) and not truth(expr[2],assignment)
    return assignment[frozen(expr)]


def main():
    c=X.verify();X.check('candidate_freeze.json');X.check('prediction_freeze.json')
    assert X.read(X.OUT/'protocol_receipt.json')['protocol_sha256']==X.sha(X.OUT/'protocol.json')
    checked_proofs=assignments=0;cached=set()
    for d in X.E.M.TARGETS:
        cat=X.read(X.OUT/'catalogues'/f'{d}.json')
        universe={s['school_id'] for s in cat['all_official_school_scopes']}
        restricted={s for r in cat['all_policy_relations'] if r['kind'] in ('university','eligibility','pending') for s in r['school_ids']}
        for cell in cat['new_scope_cells']:
            assert set(cell['school_ids'])<=universe and not(set(cell['school_ids'])&restricted)
            assert set(cell['school_ids'])=={p['school_id'] for p in cell['evidence']}
            for p in cell['evidence']:
                assert p['evidence'] and p['source_text']
                a=p['condition'];b=p['sufficient_scope_branch'];key=(frozen(a),frozen(b))
                if key in cached:continue
                cached.add(key);leaves=sorted(atoms(a)|atoms(b))
                assert len(leaves)<=16,'Large proof needs another verifier'
                for bits in itertools.product([False,True],repeat=len(leaves)):
                    state=dict(zip(leaves,bits));assert not truth(a,state) or truth(b,state),(a,b)
                    assignments+=1
                checked_proofs+=1
    pp=X.E.probabilities();mean={d:np.stack([p[d] for p in pp]).mean(0) for d in pp[0]};checked=0
    for label,probs in [('ensemble',mean)]+list(zip(map(str,c['seeds']),pp)):
        for d in X.E.M.TARGETS:
            meta=X.read(X.E.M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
            sites=X.read(X.OUT/'site_candidates'/f'{d}.json');previous=X.read(X.E.OUT/'site_candidates'/f'{d}.json')
            for arm in X.ARMS:
                rows=X.read(X.OUT/'predictions'/arm/label/f'{d}.json')
                old=None if arm=='scope_cell_sets' else X.read(X.E.OUT/'predictions'/arm/label/f'{d}.json')
                for q,(r,s) in enumerate(zip(rows,sites)):
                    assert r['site_id']==s['site_id']==previous[q]['site_id']
                    assert s['positive_lower_bound']==previous[q]['positive_lower_bound']
                    assert s['exclusions']==previous[q]['exclusions']
                    expected=independent_pick(probs[d][q],s['scope_cell_candidates'],ids,c['beta_fixed']) if old is None else old[q]['after']
                    assert r['after']==expected,(arm,label,d,q)
                    checked+=1
    result=X.read(X.OUT/'results.json')
    for arm in X.ARMS:
        rows=X.read(X.OUT/'scored_rows'/f'{arm}.json')
        assert len(rows)==1761 and sum(r['semantic_pending'] for r in rows)==242
        correct=fp=fn=0
        for r in rows:
            g=set(r['truth']);p=set(r['prediction']);assert r['correct']==(p==g)
            correct+=int(p==g);fp+=len(p-g);fn+=len(g-p)
        m=result['metrics'][arm]['全部'];assert (correct,fp,fn)==(m['correct'],m['fp'],m['fn'])
    for d,h in result['gold_sha256'].items():assert X.sha(X.E.M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    X.write(X.OUT/'audit.json',dict(passed=True,output_rows_independently_replayed=checked,
        unique_scope_implications_exhaustively_checked=checked_proofs,boolean_assignments_checked=assignments,
        same_weights_probabilities_and_positive_inputs=True,unchanged_gold=True,semantic_pending_retained=242,
        limitation='Boolean verification validates implications of parsed expressions, not correctness of geographic identity or parser interpretation.'))
    print('AUDIT PASSED',checked,'outputs',checked_proofs,'proofs',assignments,'assignments',flush=True)


if __name__=='__main__':main()
