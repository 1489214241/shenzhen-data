"""Audit staged resident-group guard against all202 official school scopes.

Uses the frozen membership/road estimate audit as the counterfactual baseline.
No school polygon or evaluation labels. Named leaves are separately verified.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
import shenzhen_aligned_policy as AP
from shenzhen_scoped_ring import ScopedRingParser
from shenzhen_resident_group_scope import ResidentGroupParser
from audit_shenzhen_community_geometry import SourceGeometryEvaluator,SOURCES,INPUT
from build_shenzhen_aligned_inputs import parse_scope,strict_line

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'resident_group_scope_audit'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve audit')
    for folder in [SOURCES/'review_v1',INPUT]:
        for rel,h in F.read(folder/'manifest.json')['files'].items():assert F.sha(folder/rel)==h
    roadpath=F.OUT/'road_geometry.json'
    assert F.sha(roadpath)==F.read(F.OUT/'manifest.json')['files']['road_geometry.json']
    byname=defaultdict(list)
    for f in F.read(roadpath)['features']:
        for key in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
            for name in str(f['properties'].get(key,'')).split(';'):
                if name:byname[name].append(f)
    facts=F.read(SOURCES/'review_v1/membership_facts.json')['facts']
    profiles=F.read(SOURCES/'profile_inventory.json')['profiles'];summary={}
    for d in F.SZ.DISTRICTS:
        meta=F.read(ROOT/'round03_partial_labels/inputs'/f'{d}.json');sites=meta['sites'];schools=meta['schools']
        policy=F.read(INPUT/d/'reviewed_policy.json');index={r['school_id']:r for r in policy['school_scope_inventory']}
        inv=[index[s['school_id']] for s in schools]
        asts=[[p(inv,parse_scope).parse(r['base_scope_text']) for r in inv] for p in [ScopedRingParser,ResidentGroupParser]]
        coords=np.array([s['representative_point_wgs84'] for s in sites]);n=len(sites);sides={}
        def unknown(*args,**kwargs):return np.full(n,-1,np.int8)
        def side(name,direction,alias=None):
            key=(name,direction,alias)
            if key not in sides:
                fs={f['properties']['osm_id']:f for f in byname.get(alias or name,[])}
                sides[key]=strict_line(dict(status='ready',predicate='line_side',inward=direction,
                    geometries=[f['geometry'] for f in fs.values()]),coords) if fs else unknown()
            return sides[key].copy()
        evs=[]
        for aa in asts:
            ev=SourceGeometryEvaluator(sites,schools,inv,policy,byname,parse_scope,side,unknown,F.project,strict_line)
            ev.asts=aa;ev.install_membership(facts,profiles,True);evs.append(ev)
        allrows=[];withdrawn=[];changes=0
        for j,row in enumerate(inv):
            changed=asts[0][j]!=asts[1][j];changes+=changed
            named=[ev.positive_terms(aa[j]) for ev,aa in zip(evs,asts)]
            assert named[0]==named[1],row['school_name']
            vals=[ev.evaluate(aa[j],j) if not row['status'].endswith('_pending') else unknown() for ev,aa in zip(evs,asts)]
            assert not ((vals[1]==1)&(vals[0]!=1)).any(),row['school_name']
            removed=np.flatnonzero((vals[0]==1)&(vals[1]!=1))
            for q in removed:
                withdrawn.append(dict(school_id=row['school_id'],school_name=row['school_name'],site_id=sites[q]['coordinate_group_id'],
                    building_names=sites[q]['coordinate_group_building_names'],school_scope_text=row['base_scope_text'],
                    evidence=row['evidence'],before=int(vals[0][q]),after=int(vals[1][q]),
                    reason='resident_group_identity_unconfirmed'))
            allrows.append(dict(school_id=row['school_id'],school_name=row['school_name'],scope=row['base_scope_text'],evidence=row['evidence'],
                ast_changed=changed,before_ast=asts[0][j],after_ast=asts[1][j],named_terms=named[1],
                before_counts={str(k):int((vals[0]==k).sum()) for k in [-1,0,1]},
                after_counts={str(k):int((vals[1]==k).sum()) for k in [-1,0,1]}))
        summary[d]=dict(schools=len(inv),changed_school_asts=changes,withdrawn_composed_positive_pairs=len(withdrawn),
                        explicit_named_terms_preserved=True,new_positive_pairs=0)
        F.write(OUT/f'{d}.json',dict(all_schools=allrows,withdrawn_composed_estimates=withdrawn,
            target_gold_read=False,target_polygon_read=False,changed_frozen_inputs=False))
        print(d,summary[d],flush=True)
    F.write(OUT/'summary.json',dict(summary=summary,adopted=False,
        limitation='Staged semantic repair; withdrawn estimates may retain independent named evidence. Existing frozen training inputs unchanged.'))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        sources={str(p):F.sha(p) for p in [Path(__file__),roadpath,SOURCES/'review_v1/manifest.json',INPUT/'manifest.json',
                 F.REPO/'scripts/shenzhen_resident_group_scope.py',F.REPO/'scripts/audit_shenzhen_community_geometry.py']}))


if __name__=='__main__':main()
