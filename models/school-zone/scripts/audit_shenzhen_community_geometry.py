"""Source-only coverage audit for membership AND independent road conditions.

Does not change the frozen round08 experiment. Uses the exact Changsha line-side
predicate already present in the adapter and independently archived OSM ways.
No office proxy, target answer, school polygon, fuzzy name, or unknown-as-false.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
import shenzhen_aligned_policy as AP
from build_shenzhen_aligned_inputs import parse_scope,strict_line
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_scoped_ring import ScopedRingParser

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCES=ROOT/'community_membership_sources'
INPUT=F.REPO/'data/深圳市/局部条件修复输入_20261112'
OUT=ROOT/'community_geometry_source_audit'


class SourceGeometryEvaluator(AP.Evaluator):
    def install_membership(self,facts,profiles,enabled=True):
        names=defaultdict(set)
        for p in profiles:
            if p['district']==self.policy['district']:names[p['community']].add((p['street'],p['community']))
        ids={s['coordinate_group_id']:q for q,s in enumerate(self.sites)}
        self.membership_values={};self.street_values={};self.source_facts={}
        for f in facts:
            if f['district']!=self.policy['district']:continue
            q=ids[f['site_id']]
            self.source_facts[q]=f
            if not enabled:continue
            if names[f['community']]=={(f['street'],f['community'])}:
                self.membership_values.setdefault(f['community'],self.unknown())[q]=1
            self.street_values.setdefault(f['street'],self.unknown())[q]=1

    def evaluate(self,a,j,stack=(),exclude=False,partial=False):
        if a['op']=='membership':return self.membership_values.get(a['text'],self.unknown()).copy()
        if a['op']=='admin':return self.street_values.get(a['text'],self.unknown()).copy()
        # Isolate membership/geography. A missing exact name in an exclusion
        # list does not prove non-membership, so these leaves remain unknown.
        if a['op'] in ['named','address']:return self.unknown()
        return super().evaluate(a,j,stack,exclude,partial)


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source coverage audit')
    for folder,manifest in [(SOURCES/'review_v1','manifest.json'),(INPUT,'manifest.json')]:
        for rel,h in F.read(folder/manifest)['files'].items():assert F.sha(folder/rel)==h,(folder,rel)
    road_path=F.OUT/'road_geometry.json'
    assert F.sha(road_path)==F.read(F.OUT/'manifest.json')['files']['road_geometry.json']
    byname=defaultdict(list)
    for f in F.read(road_path)['features']:
        for k in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
            for name in str(f['properties'].get(k,'')).split(';'):
                if name:byname[name].append(f)
    facts=F.read(SOURCES/'review_v1/membership_facts.json')['facts']
    profiles=F.read(SOURCES/'profile_inventory.json')['profiles'];summary={}
    for d in F.SZ.DISTRICTS:
        meta=F.read(ROOT/'round03_partial_labels/inputs'/f'{d}.json')
        sites,schools=meta['sites'],meta['schools'];policy=F.read(INPUT/d/'reviewed_policy.json')
        indexed={r['school_id']:r for r in policy['school_scope_inventory']};inv=[indexed[s['school_id']] for s in schools]
        coords=np.array([s['representative_point_wgs84'] for s in sites]);n=len(sites);sides={};side_audit=[]
        def unknown(*args,**kwargs):return np.full(n,-1,np.int8)
        def side(name,direction,alias=None):
            key=(name,direction,alias)
            if key not in sides:
                fs={f['properties']['osm_id']:f for f in byname.get(alias or name,[])}
                sides[key]=strict_line(dict(status='ready',predicate='line_side',inward=direction,
                    geometries=[f['geometry'] for f in fs.values()]),coords) if fs else unknown()
                side_audit.append(dict(name=name,direction=direction,alias=alias,source_osm_ids=sorted(fs),
                    counts={str(k):int((sides[key]==k).sum()) for k in [-1,0,1]}))
            return sides[key].copy()
        old_pairs={(f['site_id'],f['school_id']) for f in F.read(ROOT/'round07_community_facts/facts'/f'{d}.json')['facts']}
        summary[d]={}
        for version,parser in [('qualified',QualifiedParser),('scoped_ring',ScopedRingParser)]:
            asts=[parser(inv,parse_scope).parse(r['base_scope_text']) for r in inv]
            evs=[]
            for enabled in [False,True]:
                ev=SourceGeometryEvaluator(sites,schools,inv,policy,byname,parse_scope,side,unknown,F.project,strict_line)
                ev.asts=asts;ev.install_membership(facts,profiles,enabled);evs.append(ev)
            added=[];school_audit=[]
            for j,(row,ast) in enumerate(zip(inv,asts)):
                if row['status'].endswith('_pending'):continue
                before,after=[ev.evaluate(ast,j) for ev in evs]
                hit=np.flatnonzero((after==1)&(before!=1))
                for q in hit:
                    f=evs[1].source_facts[int(q)]
                    added.append(dict(district=d,site_id=sites[q]['coordinate_group_id'],building_names=sites[q]['coordinate_group_building_names'],
                        school_id=row['school_id'],school_name=row['school_name'],ast=ast,school_scope_text=row['base_scope_text'],
                        school_sources=row['evidence'],membership_source=f,novel_vs_round07=(sites[q]['coordinate_group_id'],row['school_id']) not in old_pairs,
                        kind='reviewed_membership_and_independent_road_estimate',complete_school_set=False,enrollment_eligibility=False))
                school_audit.append(dict(school_id=row['school_id'],school_name=row['school_name'],
                    source=row['base_scope_text'],ast=ast,before={str(k):int((before==k).sum()) for k in [-1,0,1]},
                    after={str(k):int((after==k).sum()) for k in [-1,0,1]},additional_membership_supported_sites=len(hit)))
            F.write(OUT/d/f'{version}.json',dict(facts=added,all_school_audit=school_audit,road_side_audit=side_audit,
                ring_geometry=evs[1].geometry_audit,target_gold_read=False,target_polygon_read=False,office_proxy_used=False,adopted=False))
            summary[d][version]=dict(schools=len(inv),source_supported_pairs=len(added),
                source_supported_sites=len({r['site_id'] for r in added}),novel_pairs_vs_round07=sum(r['novel_vs_round07'] for r in added),
                novel_sites_vs_round07=len({r['site_id'] for r in added if r['novel_vs_round07']}))
        print(d,summary[d],flush=True)
    F.write(OUT/'summary.json',dict(summary=summary,
        caution='Independent geometry estimate only; no current training or scoring change; road identity, geographic predicates and source dates remain auditable limitations.',
        target_gold_read=False,target_polygon_read=False,adopted=False))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        source_inputs={str(p):F.sha(p) for p in [Path(__file__),road_path,SOURCES/'review_v1/manifest.json',SOURCES/'profile_inventory.json',
            INPUT/'manifest.json',F.REPO/'scripts/shenzhen_scoped_ring.py',F.REPO/'scripts/shenzhen_aligned_policy.py']}))


if __name__=='__main__':main()
