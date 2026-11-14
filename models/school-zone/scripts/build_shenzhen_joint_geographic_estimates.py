"""Source-only intervention: actual community identity AND finite road sides.

Reuse the exact Changsha finite-line predicate and the frozen independent OSM
geometry. Do not use office proxies, target answers, or catchment polygons.
"""
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope, strict_line, SIDE_PATH
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_administrative_identity import ReviewedAdministrativeScopeProof
from shenzhen_joint_geographic_proof import JointGeographicProof, contains_witness

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'joint_geographic_estimates_v1'
POLICY = F.REPO / 'data/深圳市/边界合取单项修复输入_20261112'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve source-only estimates')
    fact_paths = [ROOT / sub / 'membership_facts.json' for sub in
        ['administrative_identity_review_v1', 'yuanshan_profile_review_v1']]
    facts = [f for p in fact_paths for f in F.read(p)['facts']]
    assert len(facts) == len({(f['district'], f['site_id']) for f in facts}) == 229
    identities_path = ROOT / 'administrative_identity_review_v1/identities.json'
    identities = F.read(identities_path)['identities']
    profiles_path = ROOT / 'community_membership_sources/profile_inventory.json'
    profiles = F.read(profiles_path)['profiles']
    road_path = F.OUT / 'road_geometry.json'
    byname = defaultdict(dict)
    for feature in F.read(road_path)['features']:
        for key in ['name', 'name:zh', 'name:zh-Hans', 'official_name', 'alt_name', 'old_name']:
            for name in str(feature['properties'].get(key, '')).split(';'):
                if name:
                    byname[name][feature['properties']['osm_id']] = feature
    sources = {str(p): F.sha(p) for p in [Path(__file__), SIDE_PATH, identities_path, profiles_path,
        road_path, F.REPO / 'scripts/build_shenzhen_aligned_inputs.py',
        F.REPO / 'scripts/shenzhen_joint_geographic_proof.py',
        F.REPO / 'scripts/shenzhen_administrative_identity.py',
        F.REPO / 'scripts/shenzhen_community_facts.py',
        F.REPO / 'scripts/shenzhen_boundary_conjunction_v2.py'] + fact_paths}
    for sub in ['administrative_identity_review_v1', 'yuanshan_profile_review_v1']:
        manifest = F.read(ROOT / sub / 'manifest.json')
        for p, h in manifest['sources'].items():
            assert F.sha(p) == h, p
            sources[p] = h
        for rel, h in manifest['files'].items():
            assert F.sha(ROOT / sub / rel) == h, rel
            sources[str(ROOT / sub / rel)] = h
    coverage = {}
    for district in F.SZ.DISTRICTS:
        meta_path = POLICY / district / 'model_metadata.json'
        policy_path = POLICY / district / 'reviewed_policy.json'
        sources.update({str(p): F.sha(p) for p in [meta_path, policy_path]})
        sites = F.read(meta_path)['sites']
        coords = np.array([r['representative_point_wgs84'] for r in sites])
        indexes = {r['coordinate_group_id']: i for i, r in enumerate(sites)}
        road_cache = {}; road_audit = []

        def road(name, direction, alias, fact):
            key = (name, direction, alias)
            if key not in road_cache:
                features = byname.get(alias or name, {})
                values = np.full(len(sites), -1, np.int8)
                if features:
                    values = strict_line(dict(status='ready', predicate='line_side', inward=direction,
                        geometries=[f['geometry'] for f in features.values()]), coords)
                record = dict(policy_name=name, direction=direction, explicit_alias=alias,
                    source_osm_ids=sorted(features), predicate='unchanged_Changsha_strict_line',
                    states=dict(Counter(str(int(v)) for v in values)),
                    missing_geometry=not bool(features))
                road_cache[key] = (values, record); road_audit.append(record)
            values, record = road_cache[key]
            q = indexes[fact['site_id']]; value = int(values[q])
            witness = dict(kind='independent_road_side_estimate', site_id=fact['site_id'],
                policy_name=name, direction=direction, explicit_alias=alias,
                source_osm_ids=record['source_osm_ids'], value=value,
                point_wgs84=coords[q].tolist(), road_source_sha256=sources[str(road_path)],
                predicate='Changsha strict_line: finite cross-section agreement, 30m < side distance < 10km')
            return value, [witness] if value >= 0 else []

        inv = F.read(policy_path)['school_scope_inventory']
        parser = BoundaryConjunctionParserV2(inv, parse_scope)
        asts = [parser.parse(r['base_scope_text']) for r in inv]
        local = [f for f in facts if f['district'] == district]
        baseline = ReviewedAdministrativeScopeProof(inv, asts, profiles, district, identities).derive(local)
        joint = JointGeographicProof(inv, asts, profiles, district, identities, road).derive(local)
        base_keys = {(r['site_id'], r['school_id']) for r in baseline}
        added = [r for r in joint if (r['site_id'], r['school_id']) not in base_keys
                 and contains_witness(r['proof'], 'official_community_membership')
                 and contains_witness(r['proof'], 'independent_road_side_estimate')]
        for row in added:
            row['evidence_kind'] = 'official_membership_with_independent_road_geographic_estimate'
            row['is_geographic_estimate'] = True
            row['verified_enrollment_eligibility'] = False
        # Full status ledger makes unresolved clauses visible without scoring them.
        engine = JointGeographicProof(inv, asts, profiles, district, identities, road)
        ledger = []
        for fact in local:
            for j, (school, ast) in enumerate(zip(inv, asts)):
                if school['status'].endswith('_pending'):
                    continue
                value, evidence = engine.evaluate(ast, fact, (j,))
                ledger.append(dict(site_id=fact['site_id'], school_id=school['school_id'],
                    state=value, proof=evidence))
        F.write(OUT / district / 'estimates.json', dict(facts=added, target_gold_read=False,
            target_polygons_read=False, new_negative_training_labels=0,
            limitations='Road observations are independent geographical estimates, not official school polygons.'))
        F.write(OUT / district / 'scope_audit.json', dict(rows=ledger, roads=road_audit,
            school_asts=[dict(school_id=r['school_id'], school_name=r['school_name'],
                source_text=r['base_scope_text'], evidence=r['evidence'], ast=a) for r, a in zip(inv, asts)]))
        coverage[district] = dict(verified_memberships=len(local), schools=len(inv),
            community_only_relations=len(baseline), added_geographic_estimates=len(added),
            added_sites=len({r['site_id'] for r in added}),
            added_schools=len({r['school_id'] for r in added}),
            full_scope_states=dict(Counter(str(r['state']) for r in ledger)))
        print('JOINT', district, coverage[district], flush=True)
        for row in added:
            print('ESTIMATE', row['building_names'], row['school_name'], row['school_scope_text'], flush=True)
    F.write(OUT / 'summary.json', dict(coverage=coverage, input_memberships=229,
        target_gold_read=False, target_polygons_read=False, office_proxies_used=False,
        unknown_is_negative=False, current_property_registry_used=False, adopted=False))
    F.write(OUT / 'manifest.json', dict(sources=sources, files={
        str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*.json')}))


if __name__ == '__main__':
    main()
