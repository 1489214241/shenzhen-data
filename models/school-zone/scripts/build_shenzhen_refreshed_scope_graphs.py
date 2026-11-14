"""Rebuild scope implication after parser repairs; no target labels or geometry."""
from pathlib import Path
from itertools import product
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_scope_implication import build
from zoning import shenzhen_policy as SP

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
INPUT=F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
OLD=ROOT/'round03_partial_labels/source_graphs'
OUT=ROOT/'scope_graph_refresh_review/graphs'
PDF=F.SZ.DATA/'原始资料/年度政策/附件1：龙岗区2026年义务教育阶段小学一年级学位申请指南.pdf_dd40f1d9af.pdf'


def atoms(expr):
    if expr[0] in ('and','or','minus'):
        return set().union(*(atoms(c) for c in expr[1:]))
    return {tuple(expr)}


def truth(expr,assignment):
    op=expr[0]
    if op=='and':return all(truth(c,assignment) for c in expr[1:])
    if op=='or':return any(truth(c,assignment) for c in expr[1:])
    if op=='minus':return truth(expr[1],assignment) and not truth(expr[2],assignment)
    return assignment[tuple(expr)]


def exhaustive_implication(a,b):
    leaves=sorted(atoms(a)|atoms(b))
    if len(leaves)>16:raise ValueError('Unexpected large new proof; requires separate review')
    checked=0;antecedent_true=0
    for bits in product([False,True],repeat=len(leaves)):
        assignment=dict(zip(leaves,bits));left=truth(a,assignment);right=truth(b,assignment)
        assert not left or right,(a,b,assignment)
        checked+=1;antecedent_true+=int(left)
    return dict(independent_boolean_assignments=checked,antecedent_true_assignments=antecedent_true,
                atoms=leaves,valid=True,geographic_atoms_not_evaluated=True)


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source graph audit')
    sources={str(p):F.sha(p) for p in [Path(__file__),PDF,
        F.REPO/'scripts/shenzhen_scope_implication.py',F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py',
        F.REPO/'scripts/shenzhen_resident_group_scope.py',F.REPO/'scripts/shenzhen_qualified_scope.py',
        F.REPO/'scripts/shenzhen_aligned_policy.py',F.REPO/'scripts/build_shenzhen_aligned_inputs.py',
        F.REPO/'zoning/shenzhen_policy.py',OUT.parent/'longgang_page9.png']}
    summary={}
    for d in F.SZ.DISTRICTS:
        path=INPUT/d/'reviewed_policy.json';inv=F.read(path)['school_scope_inventory'];sources[str(path)]=F.sha(path)
        parser=BoundaryConjunctionParserV2(inv,parse_scope)
        rows=[dict(school_id=r['school_id'],school_name=r['school_name'],source_text=r['base_scope_text'],
            evidence=r['evidence'],scope_status=r['status'],ast=parser.parse(r['base_scope_text'])) for r in inv]
        policy=SP.load_policy(d);policy_path=F.SZ.DATA/d/'政策规则/规则包.json';sources[str(policy_path)]=F.sha(policy_path)
        graph=build(rows,policy);old=F.read(OLD/f'{d}.json');sources[str(OLD/f'{d}.json')]=F.sha(OLD/f'{d}.json')
        key=lambda e:(e['trigger_school_id'],e['implied_school_id'])
        before={key(e):e for e in old['edges']};after={key(e):e for e in graph['edges']}
        added=[dict(after[k],independent_proof=exhaustive_implication(after[k]['trigger_expression'],after[k]['implied_expression'])) for k in sorted(after.keys()-before.keys())]
        removed=[before[k] for k in sorted(before.keys()-after.keys())]
        # This intervention is a graph refresh, not permission to drop old
        # predictions. Any future removal needs its own controlled replay.
        assert not removed,'Removal requires separately declared experiment'
        F.write(OUT/f'{d}.json',graph)
        F.write(OUT/f'{d}_changes.json',dict(added=added,removed=removed,school_inventory=rows))
        summary[d]=dict(schools=len(rows),old_edges=len(before),new_edges=len(after),added=len(added),removed=len(removed))
    F.write(OUT/'summary.json',dict(districts=summary,official_pdf_page9_visually_reviewed=True,
        scope_status_and_university_exclusions_unchanged=True,target_gold_read=False,target_polygon_read=False,
        interpretation='Direct inclusion between parsed original scopes; correctness of the predicted trigger school remains uncertain.',adopted=False))
    F.write(OUT/'manifest.json',dict(sources=sources,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*.json')}))
    print('GRAPH_REFRESH',summary,flush=True)


if __name__=='__main__':main()
