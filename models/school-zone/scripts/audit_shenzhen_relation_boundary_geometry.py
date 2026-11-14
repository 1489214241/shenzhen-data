"""Check whether exact-name OSM route members repair community geometry gaps."""
from pathlib import Path
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union
import shenzhen_full_inputs as F
from audit_shenzhen_road_topology import corridor_candidates

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'community_boundary_network_gaps'
OUT = ROOT / 'community_boundary_relation_geometry'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve geometry replay')
    manifest = F.read(SOURCE / 'manifest.json')
    for path, expected in manifest['sources'].items():
        assert F.sha(path) == expected, path
    for path, expected in manifest['files'].items():
        assert F.sha(SOURCE / path) == expected, path
    old = F.read(ROOT / 'community_boundary_geometry_audit/report.json')['cases']
    reports = F.read(SOURCE / 'report.json')['cases']
    roads = {f['properties']['osm_id']: f for f in F.read(F.OUT / 'road_geometry.json')['features']}
    result = []
    for c, review in zip(old, reports):
        assert c['case_id'] == review['case_id'] and c['names'] == review['names']
        resolved = {side: set(ids) for side, ids in c['source_osm_ids'].items()}
        added = {}
        for side, relations in review['exact_name_relation_candidates'].items():
            for r in relations:
                resolved[side].update(r['local_way_ids'])
            added[side] = sorted(resolved[side] - set(c['source_osm_ids'][side]))
        if not any(added.values()):
            continue
        row = dict(case_id=c['case_id'], community=c['community'], names=c['names'],
                   added_osm_ids=added, resolved_osm_ids={s: sorted(v) for s, v in resolved.items()},
                   missing_names=[c['names'][s] for s, v in resolved.items() if not v],
                   source_relations=review['exact_name_relation_candidates'], radii={})
        if not row['missing_names']:
            lines = [unary_union([LineString(F.project(roads[i]['geometry']['coordinates']))
                                 for i in resolved[s]]) for s in '东南西北']
            row['corner_gaps_m'] = [lines[i].distance(lines[(i + 1) % 4]) for i in range(4)]
            for radius in [0, 10, 25, 50]:
                faces, supported = corridor_candidates(lines, radius)
                row['radii'][str(radius)] = dict(total_faces=len(faces), supported_faces=len(supported),
                    candidates=[dict(area_km2=p.area / 1e6, geometry=mapping(p)) for p in supported])
        result.append(row)
        print(row['community'], 'added ways', sum(map(len, added.values())),
              'supported', {r: v['supported_faces'] for r, v in row['radii'].items()},
              'corner gaps', row.get('corner_gaps_m'), flush=True)
    F.write(OUT / 'report.json', dict(cases=result, target_gold_read=False,
        target_school_polygons_read=False, target_sites_used=False, generated_memberships=0,
        model_inputs_changed=False, criterion='Unchanged four-side >=50m boundary support at radii 0/10/25/50m.'))
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in [Path(__file__),
        SOURCE / 'manifest.json', F.REPO / 'scripts/audit_shenzhen_road_topology.py']},
        files={'report.json': F.sha(OUT / 'report.json')}))


if __name__ == '__main__':
    main()
