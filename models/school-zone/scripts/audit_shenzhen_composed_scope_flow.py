"""Trace non-cardinal scope values through native rules and model channels.

Counts are input estimates, including legacy office proxies, not confirmed truth.
No target answers or target polygons are read. No rule is changed or adopted.
"""
from copy import deepcopy
import numpy as np
import shenzhen_full_inputs as F
from zoning import rules

INPUT=F.REPO/'data/深圳市/居民小组约束输入_20261112'
OUT=F.REPO/'outputs/shenzhen_resolution_20261112/composed_scope_flow_audit'


def ops(ast):
    return {ast['op']}|{op for c in ast.get('children',[]) for op in ops(c)}


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve audit')
    for rel,h in F.read(INPUT/'manifest.json')['files'].items():assert F.sha(INPUT/rel)==h,rel
    summary={}
    for d in F.SZ.DISTRICTS:
        O=F.read(INPUT/d/'aligned_observations.json');O['name_known']={int(q):ss for q,ss in O['name_known'].items()}
        A=dict(np.load(INPUT/d/'observation_arrays.npz'));meta=F.read(INPUT/d/'model_metadata.json')
        stored=np.load(INPUT/d/'model_arrays.npz');policy=F.read(INPUT/d/'reviewed_policy.json')
        S=dict(rules.PRESETS['unified_20261106']['芙蓉区']);relaxed=deepcopy(S);relaxed['off']=list(S['off'])+['K8']
        reports=[];rows=[]
        for j,school in enumerate(meta['schools']):
            cls=[c for c in O['clauses'] if c['school']==j]
            before,st,negative=rules.school(j,cls,O,A,S)
            after,st2,negative2=rules.school(j,cls,O,A,relaxed)
            # Named-phase guards may further withdraw states after this native call.
            changes=np.flatnonzero(after&~before)
            composed=rules.OR([A['side_values'][:,c['sides'][0]['column']] for c in cls],O['n'])
            unknown=(stored['statuses'][:,j]<0).all(1)
            no_reference=np.array([j not in meta['known'].get(str(q),[]) for q in range(O['n'])])
            hidden=np.flatnonzero((composed==1)&unknown&no_reference)
            clause_ops=sorted({v for c in cls for v in ops(c['ast'])})
            reports.append(dict(school_id=school['school_id'],school_name=school['school_name'],operators=clause_ops,
                composed_positive=int((composed==1).sum()),positive_not_in_cardinal_channels_and_not_reference=len(hidden),
                regained_hits_without_K8=len(changes),native_positive_hits=int(before.sum())))
            for q in sorted(set(hidden.tolist())|set(changes.tolist())):
                rows.append(dict(site_id=meta['sites'][q]['coordinate_group_id'],names=meta['sites'][q]['coordinate_group_building_names'],
                    school_id=school['school_id'],school_name=school['school_name'],operators=clause_ops,
                    hidden_composed=bool(q in hidden),regained_without_K8=bool(q in changes),
                    composed_value=int(composed[q]),cardinal_status=stored['statuses'][q,j].tolist(),
                    native_known_school_indices=O['name_known'].get(q,[]),
                    reference_school_indices=meta['known'].get(str(q),[])))
        summary[d]=dict(schools=len(reports),sites=O['n'],
            positive_composed_without_cardinal_or_reference=sum(r['positive_not_in_cardinal_channels_and_not_reference'] for r in reports),
            schools_with_hidden_composed=sum(r['positive_not_in_cardinal_channels_and_not_reference']>0 for r in reports),
            regained_hits_without_K8=sum(r['regained_hits_without_K8'] for r in reports),
            schools_with_K8_withdrawal=sum(r['regained_hits_without_K8']>0 for r in reports))
        F.write(OUT/d/'school_flow.json',reports);F.write(OUT/d/'affected_pairs.json',rows)
        print(d,summary[d],flush=True)
    F.write(OUT/'summary.json',dict(districts=summary,target_gold_read=False,target_polygon_read=False,
        interpretation='Only a flow audit. Composed estimates may use office proxies; cardinal/reference absence does not imply all model features lack information.',
        adopted=False))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        input_manifest=F.sha(INPUT/'manifest.json'),code=F.sha(__file__),native_rules=F.sha(F.REPO/'zoning/rules.py')))


if __name__=='__main__':main()
