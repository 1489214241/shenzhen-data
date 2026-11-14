"""Audit community corners and recover road identity from the frozen OSM PBF.

No school labels, school polygons or target property points are read. Relation
names generate review candidates only; they are not automatically road aliases.
"""
from pathlib import Path
import sys
from collections import Counter
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union, nearest_points
from pyproj import Transformer
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'community_boundary_network_gaps'
NAMES = ('name', 'name:zh', 'name:zh-Hans', 'official_name', 'alt_name', 'old_name')


def names(tags):
    return {v for key in NAMES for v in str(tags.get(key, '')).split(';') if v}


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed network gap audit')
    roadpath = F.OUT / 'road_geometry.json'
    casepath = ROOT / 'community_boundary_geometry_audit/report.json'
    pbfpath = F.REPO / 'tmp/shenzhen_guangdong.osm.pbf'
    expected = F.read(F.OUT / 'road_geometry_audit.json')['tiles'][0]['source_sha256']
    assert F.sha(pbfpath) == expected
    features = F.read(roadpath)['features']
    byid = {f['properties']['osm_id']: f for f in features}
    cases = F.read(casepath)['cases']
    required = {n for c in cases for n in c['names'].values()}
    selected_ids = set(byid)
    sys.path.insert(0, str(F.REPO / 'tmp/shenzhen_osmium_vendor'))
    import osmium

    class Relations(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.scanned = 0
            self.selected = []

        def relation(self, rel):
            self.scanned += 1
            tags = dict(rel.tags)
            kind = tags.get('type')
            if not (kind in {'associatedStreet', 'street', 'waterway'} or
                    kind == 'route' and tags.get('route') in {'road', 'railway'}):
                return
            members = [dict(type=m.type, ref=m.ref, role=m.role) for m in rel.members]
            local = [m for m in members if m['type'] == 'w' and m['ref'] in selected_ids]
            if local or names(tags) & required:
                self.selected.append(dict(id=rel.id, tags=tags, members=members,
                                          local_way_ids=sorted({m['ref'] for m in local})))

    handler = Relations()
    handler.apply_file('tmp/shenzhen_guangdong.osm.pbf')
    F.write(OUT / 'local_route_relations.json', dict(
        scanned_relations=handler.scanned, selected_relations=handler.selected,
        source_sha256=expected, school_relations_generated=0))
    reverse = Transformer.from_crs(32650, 4326, always_xy=True)
    reports = []
    for c in cases:
        ids = c['source_osm_ids']
        geoms = {side: unary_union([LineString(F.project(byid[i]['geometry']['coordinates']))
                                   for i in ways]) for side, ways in ids.items() if ways}
        corners = []
        for a, b in [('东', '南'), ('南', '西'), ('西', '北'), ('北', '东')]:
            if a not in geoms or b not in geoms:
                corners.append(dict(sides=[a, b], status='missing_named_geometry'))
                continue
            x, y = nearest_points(geoms[a], geoms[b])
            distance = geoms[a].distance(geoms[b])
            corners.append(dict(sides=[a, b], names=[c['names'][a], c['names'][b]],
                status='intersects' if distance < 1e-6 else 'disconnected',
                min_distance_m=distance,
                nearest_points_wgs84=[list(reverse.transform(p.x, p.y)) for p in (x, y)],
                warning='Minimum over full named ways; not proof of the intended corner.'))
        relation_candidates = {}
        for side, name in c['names'].items():
            matches = []
            for r in handler.selected:
                if name not in names(r['tags']):
                    continue
                # Non-street member roles are never silently treated as boundary segments.
                local = [m['ref'] for m in r['members'] if m['type'] == 'w' and
                         m['ref'] in byid and (r['tags'].get('type') not in
                         {'associatedStreet', 'street'} or m['role'] == 'street')]
                matches.append(dict(relation_id=r['id'], tags=r['tags'],
                    local_way_ids=sorted(set(local)),
                    new_way_ids=sorted(set(local) - set(ids[side])),
                    admitted=False))
            relation_candidates[side] = matches
        reports.append(dict(case_id=c['case_id'], community=c['community'],
            names=c['names'], missing_names=c['missing_names'], corners=corners,
            exact_name_relation_candidates=relation_candidates,
            sources=c['sources'], school_relations_generated=0))
        print(c['case_id'], c['community'], [(p['sides'], round(p['min_distance_m'], 1))
              for p in corners if 'min_distance_m' in p],
              'relation_candidates', sum(map(len, relation_candidates.values())), flush=True)
    F.write(OUT / 'report.json', dict(cases=reports,
        scanned_relations=handler.scanned, local_relation_count=len(handler.selected),
        relation_types=dict(Counter(r['tags'].get('type') for r in handler.selected)),
        target_gold_read=False, target_school_polygons_read=False, target_sites_used=False,
        changed_model_inputs=False, generated_memberships=0,
        method='Adjacent directional boundary minimum distances; exact road names on frozen PBF relations. No gap filling.'))
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in
        [Path(__file__), Path(F.__file__), roadpath, casepath, pbfpath]},
        files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
