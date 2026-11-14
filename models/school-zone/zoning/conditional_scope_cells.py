"""Compile sufficient official scope branches into conditional set hypotheses.

Every school in a cell has its own condition -> original scope proof. This is
not propagation across school edges and does not read residential Gold/maps.
"""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_scope_implication import canonical, implies, Unsupported


def sufficient_branches(node):
    """Split only outer ORs. AND, minus, ring, and qualified identities stay whole.

    An unsupported OR branch does not invalidate a separate sufficient branch,
    but an unsupported child within an AND or exclusion cannot be dropped.
    """
    if node['op']=='or':
        return [expr for child in node['children'] for expr in sufficient_branches(child)]
    try:return [canonical(node)]
    except Unsupported:return []


def build_cells(observations,policy):
    restricted={s for r in policy['relations'] if r['kind'] in ('university','eligibility','pending') for s in r['school_ids']}
    byid={r['school_id']:r for r in observations};branches={};audit=[]
    for row in observations:
        sid=row['school_id'];why=None
        if sid in restricted:why='university_eligibility_or_pending_member'
        elif row['scope_status']!='individual_scope_requires_site_applicability':why=row['scope_status']
        else:
            branches[sid]=sufficient_branches(row['ast'])
            if not branches[sid]:why='no_supported_sufficient_branch'
        audit.append(dict(school_id=sid,school_name=row['school_name'],source_text=row['source_text'],
            admitted_branches=branches.get(sid,[]),skip_reason=why,evidence=row['evidence']))
    conditions=sorted({b for bs in branches.values() for b in bs})
    records=[]
    for condition in conditions:
        proofs=[]
        for sid,bs in branches.items():
            matches=[b for b in bs if implies(condition,b)]
            if matches:
                proofs.append(dict(school_id=sid,school_name=byid[sid]['school_name'],
                    condition=condition,sufficient_scope_branch=matches[0],
                    proof='direct_boolean_condition_implies_original_scope_branch',
                    source_text=byid[sid]['source_text'],evidence=byid[sid]['evidence']))
        if len(proofs)<2:continue
        key=json.dumps(condition,ensure_ascii=False,separators=(',',':'))
        records.append(dict(candidate_id='scope_cell:'+hashlib.sha256(key.encode()).hexdigest()[:16],
            school_ids=sorted(p['school_id'] for p in proofs),root_school_ids=[],
            kind='joint_sufficient_scope_condition',condition=condition,
            exact_site_names=[condition[1]] if condition[0]=='named' else [],evidence=proofs,
            scope_applicability='unknown',complete_school_set=False,enrollment_eligibility=False,
            year=policy['year'],stage=policy['stage']))
    return records,audit
