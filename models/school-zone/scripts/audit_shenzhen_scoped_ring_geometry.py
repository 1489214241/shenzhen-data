"""Independent geography/lexical audit of six reviewed syntax changes.

No community-office proxy, target labels or school catchment geometry. This
does not regenerate references or change the running training experiment.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pdfplumber
import shenzhen_full_inputs as F
import shenzhen_aligned_policy as AP
from build_shenzhen_aligned_inputs import parse_scope, strict_line
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_scoped_ring import ScopedRingParser

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
AUDIT = ROOT / 'scoped_ring_source_audit_v2'
INPUT = F.REPO / 'data/深圳市/局部条件修复输入_20261112'


def main():
    if (AUDIT / 'geometry_review.json').exists():
        raise FileExistsError('Keep diagnostic snapshot')
    src = F.read(AUDIT / 'report.json')
    road_path = F.OUT / 'road_geometry.json'
    old_manifest = F.read(F.OUT / 'manifest.json')
    assert F.sha(road_path) == old_manifest['files']['road_geometry.json']
    byname = defaultdict(list)
    for f in F.read(road_path)['features']:
        for k in ['name', 'name:zh', 'name:zh-Hans', 'official_name', 'alt_name', 'old_name']:
            for name in str(f['properties'].get(k, '')).split(';'):
                if name:
                    byname[name].append(f)
    district = '龙岗区'
    policy = F.read(INPUT / district / 'reviewed_policy.json')
    meta = F.read(ROOT / 'round03_partial_labels/inputs' / (district + '.json'))
    schools, sites = meta['schools'], meta['sites']
    indexed = {r['school_id']:r for r in policy['school_scope_inventory']}
    inventory = [indexed[s['school_id']] for s in schools]
    unknown = lambda *args, **kwargs: np.full(len(sites), -1, np.int8)
    def side(name, direction, alias=None):
        features = byname.get(alias or name, [])
        if not features:
            return unknown()
        return strict_line(dict(status='ready', predicate='line_side', inward=direction,
            geometries=[f['geometry'] for f in features]),
            np.array([s['representative_point_wgs84'] for s in sites]))
    evaluators = []
    for parser in [QualifiedParser, ScopedRingParser]:
        ev = AP.Evaluator(sites, schools, inventory, policy, byname, parse_scope,
            side, unknown, F.project, strict_line,
            admin_features=F.read(INPUT/'admin_geometry.json')['features'])
        ev.asts = [parser(inventory, parse_scope).parse(r['base_scope_text']) for r in inventory]
        evaluators.append(ev)
    reviews = []
    for r in src['changes']:
        j = next(j for j, s in enumerate(schools) if s['school_id'] == r['school_id'])
        values = [ev.evaluate(ev.asts[j], j) for ev in evaluators]
        changed = np.flatnonzero(values[0] != values[1])
        reviews.append(dict(school_name=r['school_name'], school_id=r['school_id'],
            source_text=r['source_text'], evidence=r['evidence'],
            before_counts={str(k):int((values[0]==k).sum()) for k in [-1,0,1]},
            after_counts={str(k):int((values[1]==k).sum()) for k in [-1,0,1]},
            changed_sites=[dict(site_id=sites[q]['coordinate_group_id'],
                names=sites[q]['coordinate_group_building_names'],
                before=int(values[0][q]),after=int(values[1][q])) for q in changed]))
    pdf_path = F.SZ.DATA / src['changes'][0]['evidence'][0]['document_local_path']
    with pdfplumber.open(pdf_path) as pdf:
        pages = {str(n):pdf.pages[n-1].extract_text() for n in [9,12]}
    F.write(AUDIT/'official_page_text.json', dict(path=str(pdf_path),sha256=F.sha(pdf_path),pages=pages))
    F.write(AUDIT/'geometry_review.json', dict(reviews=reviews,
        geometry=[dict(version=name,rings=ev.geometry_audit,admin=ev.admin_audit,names=ev.name_audit)
                  for name,ev in zip(['before','after'],evaluators)],
        target_gold_read=False,school_polygon_used=False,office_proxy_used=False,
        complete_school_set=False,adopted=False))
    F.write(AUDIT/'geometry_manifest.json',dict(files={str(p):F.sha(p) for p in [
        Path(__file__),road_path,INPUT/'admin_geometry.json',AUDIT/'report.json',
        AUDIT/'geometry_review.json',AUDIT/'official_page_text.json']}))
    for r in reviews:
        print(r['school_name'],r['before_counts'],'->',r['after_counts'],flush=True)
    for r in evaluators[1].geometry_audit:
        print(r['names'],r['status'],r.get('missing',''),flush=True)


if __name__ == '__main__':
    main()
