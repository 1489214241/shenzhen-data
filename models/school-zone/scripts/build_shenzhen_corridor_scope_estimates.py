"""Freeze conservative independent-road scope estimates, never target labels."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import LineString,shape
from shapely.ops import unary_union
import shenzhen_full_inputs as F
from shenzhen_boundary_conjunction import BoundaryConjunctionParser
from build_shenzhen_aligned_inputs import parse_scope

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
AUDIT=ROOT/'road_ring_topology_audit'
RAW=F.REPO/'data/深圳市/居民小组约束输入_20261112'
OUT=ROOT/'road_corridor_scope_estimates'


def unconditional_rings(ast):
    if ast['op']=='ring':yield tuple(sorted(ast['names']))
    if ast['op']=='or':
        for child in ast['children']:yield from unconditional_rings(child)


def consensus_inside(points,lines,regions,margin=50.):
    # Requiring every independently fixed radius to enclose the point avoids
    # choosing a radius using target performance. Boundary-near points stay unknown.
    far=shapely.distance(unary_union(lines),points)>margin
    votes=np.stack([shapely.contains(p,points) for p in regions])
    return np.where(far&votes.all(0),1,-1).astype(np.int8)


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source estimates')
    m=F.read(AUDIT/'manifest.json')
    for rel,h in m['files'].items():assert F.sha(AUDIT/rel)==h,rel
    for p,h in m['sources'].items():assert F.sha(p)==h,p
    road_path=F.OUT/'road_geometry.json';roads={str(f['properties']['osm_id']):f for f in F.read(road_path)['features']}
    chosen={};geometry_audit=[]
    for case in F.read(AUDIT/'report.json')['cases']:
        radii=case.get('radii',{})
        if not radii or radii['0']['supported']!=0 or not all(radii[str(r)]['supported']==1 for r in [10,25,50]):continue
        regions=[shape(radii[str(r)]['candidates'][0]['geometry']) for r in [10,25,50]]
        assert regions[0].covers(regions[1]) and regions[1].covers(regions[2])
        lines=[unary_union([LineString(F.project(roads[i]['geometry']['coordinates'])) for i in case['source_osm_ids'][name]]) for name in case['names']]
        chosen[tuple(sorted(case['names']))]=(case,regions,lines)
    for d in F.SZ.DISTRICTS:
        meta=F.read(RAW/d/'model_metadata.json');inv=F.read(RAW/d/'reviewed_policy.json')['school_scope_inventory']
        parser=BoundaryConjunctionParser(inv,parse_scope);sites=meta['sites']
        coords=F.project([s['representative_point_wgs84'] for s in sites]);points=shapely.points(coords)
        values={key:consensus_inside(points,lines,regions) for key,(_,regions,lines) in chosen.items()}
        facts=[]
        for row in inv:
            if row['status'].endswith('_pending'):continue
            ast=parser.parse(row['base_scope_text'])
            for key in set(unconditional_rings(ast))&set(chosen):
                case,regions,lines=chosen[key]
                for q in np.flatnonzero(values[key]==1):
                    facts.append(dict(district=d,site_id=sites[q]['coordinate_group_id'],site_names=sites[q]['coordinate_group_building_names'],
                        school_id=row['school_id'],school_name=row['school_name'],source_text=row['base_scope_text'],evidence=row['evidence'],
                        geometry_case_id=case['case_id'],road_names=case['names'],source_osm_ids=case['source_osm_ids'],
                        independent_coordinate_crs=32650,min_boundary_distance_m=float(shapely.distance(unary_union(lines),points[q])),
                        all_fixed_radius_votes=[bool(p.contains(points[q])) for p in regions],
                        kind='independent_road_corridor_geographic_estimate',complete_school_set=False,enrollment_eligibility=False))
        assert len({(r['site_id'],r['school_id']) for r in facts})==len(facts)
        F.write(OUT/d/'estimates.json',dict(facts=facts,used_for_training=False,target_gold_read=False,target_polygon_read=False))
        geometry_audit.append(dict(district=d,source_scope_estimates=len(facts),sites=len({r['site_id'] for r in facts}),
            schools=len({r['school_id'] for r in facts}),cases=[dict(case_id=c['case_id'],inside_points=int((values[k]==1).sum())) for k,(c,_,_) in chosen.items()]))
        print(d,geometry_audit[-1],flush=True)
        for r in facts:print(r['site_names'],r['school_name'],round(r['min_boundary_distance_m'],1),flush=True)
    source_review=dict(official_pdf='https://www.lg.gov.cn/attachment/1/1722/1722313/12839768.pdf',pages=[9,10],
        reviewed_images=['scope_map_consistency_audit/official_page-09.png','policy_review_sources/longgang_page-10.png'],
        conclusion='Visual review confirms standalone OR ring branches for Ailian, SISU Longgang and Wenli. Other school-scope branches remain unchanged.',
        geometry_review='Plots01 and05 show gaps of16.97m and15.93m respectively; no radius selected using target labels.',
        caution='OSM road positions are independent estimates and may differ from2026policy geometry; unanimity across buffers does not verify official boundary identity.')
    F.write(OUT/'review.json',source_review)
    F.write(OUT/'summary.json',dict(districts=geometry_audit,selected_case_ids=[v[0]['case_id'] for v in chosen.values()],
        radii_m=[10,25,50],boundary_margin_m=50,negative_labels_generated=0,target_gold_read=False,target_polygon_read=False,adopted=False))
    links={str(AUDIT/'manifest.json'):F.sha(AUDIT/'manifest.json'),str(road_path):F.sha(road_path),str(Path(__file__)):F.sha(__file__)}
    for p in [ROOT/r for r in source_review['reviewed_images']]:links[str(p)]=F.sha(p)
    for d in F.SZ.DISTRICTS:
        for rel in ['model_metadata.json','reviewed_policy.json']:
            p=RAW/d/rel;links[str(p)]=F.sha(p)
    F.write(OUT/'manifest.json',dict(sources=links,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
