"""All-site policy set hypotheses. No target labels, polygons or graph closure.

Observed membership is a lower bound. A ranked union is an estimated output,
never a verified complete school set. Unknown syntax stays unknown.
"""
import itertools
import numpy as np
from .conditional_school_sets import exact_identity, normalize


def admin_names(ledger_rows):
    """Only official-profile membership records bound to the current site."""
    result=set()
    def walk(value):
        if isinstance(value,dict):
            if value.get('evidence_kind')=='official_profile_and_official_school_scope':
                if any(s.get('quote') and s.get('source',{}).get('url')
                       for s in value.get('membership_sources',[])) and value.get('community'):
                    result.add(normalize(value['community']))
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    for row in ledger_rows:
        if row['positive'] and row['routes']['hard_constraint']:walk(row['evidence'])
    return result


def condition_state(condition,site,admins):
    op=condition[0]
    if op=='named':return 1 if exact_identity(site,[condition[1]]) else -1
    if op in ('admin','membership'):return 1 if normalize(condition[1]) in admins else -1
    if op=='or':
        vals=[condition_state(x,site,admins) for x in condition[1:]]
        return 1 if 1 in vals else (0 if all(v==0 for v in vals) else -1)
    if op=='and':
        vals=[condition_state(x,site,admins) for x in condition[1:]]
        return 0 if 0 in vals else (1 if all(v==1 for v in vals) else -1)
    if op=='minus':
        a,b=(condition_state(x,site,admins) for x in condition[1:])
        return 1 if a==1 and b==0 else (0 if a==0 or b==1 else -1)
    return -1


def one_pass(roots,policy,graphs):
    roots=set(roots);out=set(roots)
    for graph in graphs:
        out.update(e['implied_school_id'] for e in graph['edges'] if e['trigger_school_id'] in roots)
    for rule in policy['relations']:
        if rule['kind']=='shared' and roots&set(rule.get('trigger_school_ids',[])):
            out.update(rule['school_ids'])
    return out


def full_prior(site,policy,graphs,cells,positive,negative,ledger_rows,scope_positive):
    negative=set(negative)
    text=one_pass(positive,policy,graphs);matched=[]
    admins=admin_names(ledger_rows)
    for cell in cells:
        if condition_state(cell['condition'],site,admins)==1:
            if not set(cell['school_ids'])&negative:
                text.update(cell['school_ids']);matched.append(cell['candidate_id'])
    # All whole-scope positives are explicitly estimates, never administrative facts.
    estimated=text|one_pass(scope_positive,policy,graphs)
    return dict(verified_members=sorted(text-negative),estimated_members=sorted(estimated-negative),
                matched_cells=matched,official_admin_names=sorted(admins),
                positive_conflicts=sorted(text&negative),complete_school_set=False,
                exact_count_verified=False,enrollment_eligibility=False)


def candidate_pool(records,prior,baseline,raw,scope_positive,excluded,compose):
    """All school hypotheses, plus explicit two-condition overlap hypotheses.

The union is considered by the model; adjacency NEVER proves applicability.
No generated member is recursively expanded into a new school relation.
"""
    lower=frozenset(prior['verified_members']);excluded=set(excluded)
    pool={};atoms=[]
    def add(schools,basis):
        key=frozenset(schools)|lower
        if not key or key&excluded:return
        if key not in pool:pool[key]=basis
    for r in records:
        key=frozenset(r['school_ids'])
        if key&excluded:continue
        add(key,[r['candidate_id']]);atoms.append((key,r['candidate_id']))
    if compose:
        # Deduplication avoids repeated equal-scope campus/branch templates.
        unique={key:rid for key,rid in atoms}
        for (a,aid),(b,bid) in itertools.combinations(unique.items(),2):
            add(a|b,[aid,bid])
    add(lower,['text_positive_lower_bound'])
    add(baseline,['current_baseline_hypothesis'])
    add(raw,['independent_model_hypothesis'])
    add(scope_positive,['whole_scope_estimate_hypothesis'])
    return sorted(pool.items(),key=lambda kv:(len(kv[0]),sorted(kv[0])))


def choose(pool,probability,ids,threshold,university_groups=(),shared_groups=()):
    """Frozen, source-calibrated independent Bernoulli membership utility.

Subtract the source threshold log odds; no target calibration or size prior.
Independent sharing evidence is needed to retain multiple university members.
"""
    q=np.clip(np.asarray(probability,dtype=float),1e-7,1-1e-7)
    t=float(threshold);utility=np.log(q/(1-q))-np.log(t/(1-t))
    byid={s:j for j,s in enumerate(ids)};universities=list(map(set,university_groups))
    shared=list(map(set,shared_groups));best=None
    for members,basis in pool:
        if any(len(hit:=set(members)&g)>1 and not any(hit<=s for s in shared) for g in universities):continue
        value=float(utility[[byid[s] for s in members]].sum())
        key=(-value,len(members),sorted(members))
        if best is None or key<best[0]:best=(key,members,basis,value)
    if best is None:return [],dict(status='no_admissible_candidate',candidate_count=len(pool))
    return sorted(best[1]),dict(status='model_estimated_school_set',basis=best[2],score=best[3],
                               candidate_count=len(pool),complete_school_set=False)


def compile_pool(pool,ids,university_groups=(),shared_groups=()):
    byid={s:j for j,s in enumerate(ids)}
    legal=[]
    for members,basis in pool:
        if any(len(hit:=set(members)&set(g))>1 and not any(hit<=set(s) for s in shared_groups)
               for g in university_groups):continue
        legal.append((members,basis))
    matrix=np.zeros((len(legal),len(ids)),dtype=np.float64)
    for i,(members,_) in enumerate(legal):matrix[i,[byid[s] for s in members]]=1
    return legal,matrix


def choose_compiled(compiled,probability,threshold):
    pool,matrix=compiled
    if not pool:return [],dict(status='no_admissible_candidate',candidate_count=0)
    q=np.clip(np.asarray(probability,dtype=float),1e-7,1-1e-7)
    t=float(threshold);u=np.log(q/(1-q))-np.log(t/(1-t))
    # einsum avoids per-row BLAS thread launch overhead for small matrices.
    score=np.einsum('ij,j->i',matrix,u,optimize=False)
    best=int(np.argmax(score))  # Pool already sorted by size and stable IDs.
    return sorted(pool[best][0]),dict(status='model_estimated_school_set',basis=pool[best][1],
        score=float(score[best]),candidate_count=len(pool),complete_school_set=False)
