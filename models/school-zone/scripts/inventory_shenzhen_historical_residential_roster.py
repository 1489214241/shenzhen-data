"""Inventory geographical columns of the two-page official 2022 roster.

Resolve a merged street only when its original cell geometrically covers the
current row. Do not promote the document's reporting month to publication date.
"""
from pathlib import Path
from collections import Counter
import pdfplumber
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'residential_roster_followup'
OUT = ROOT / 'historical_residential_roster_inventory'
URL = 'https://www.lg.gov.cn/attachment/1/1230/1230273/10343789.pdf'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve roster inventory')
    path = SOURCE / 'longgang_residential_roster.pdf'
    sitespath = ROOT / 'round03_partial_labels/inputs/龙岗区.json'
    profilespath = ROOT / 'community_membership_sources/profile_inventory.json'
    priorpaths = [ROOT / 'community_membership_sources/review_v1/membership_facts.json',
                  ROOT / 'municipal_residential_membership_sources/review_v1/membership_facts.json']
    sites = F.read(sitespath)['sites']
    prior = [r for p in priorpaths for k in ['facts', 'pending'] for r in F.read(p).get(k, [])]
    profiles = F.read(profilespath)['profiles']
    records = []
    with pdfplumber.open(path) as pdf:
        assert len(pdf.pages) == 2 and '2022' in pdf.pages[0].extract_text()
        for number, page in enumerate(pdf.pages, 1):
            tables = page.find_tables()
            assert len(tables) == 1
            table = tables[0]
            values = table.extract()
            assert values[0] == ['街道', '社区', '住宅区']
            assert len(values) == (13 if number == 1 else 22)
            for row_index, row in enumerate(values[1:], 1):
                street, community, residential = row
                street_cell = table.rows[row_index].cells[0]
                anchor = row_index
                if street is None:
                    y0 = table.rows[row_index].cells[1][1]
                    y1 = table.rows[row_index].cells[1][3]
                    anchors = [i for i in range(1, row_index) if table.rows[i].cells[0] and
                        table.rows[i].cells[0][1] <= y0 + .01 and table.rows[i].cells[0][3] >= y1 - .01]
                    assert len(anchors) == 1, (number, row_index)
                    anchor = anchors[0]
                    street = values[anchor][0]
                    street_cell = table.rows[anchor].cells[0]
                assert street and community and residential
                record = dict(page=number, table_row_including_header=row_index + 1,
                    street_anchor_row_including_header=anchor + 1, street_cell_bbox=street_cell,
                    district='龙岗区', street=norm(street) + '街道', community=norm(community) + '社区',
                    residential_name=norm(residential), source_url=URL, source_sha256=F.sha(path),
                    reporting_month='2022-10', publication_date=None,
                    date_status='Original reporting month verified; original web publication date not verified.',
                    quote=dict(street=street, community=community, residential=residential),
                    exact_coordinate_group_matches=[], ambiguous_name_candidates=[],
                    admitted=False, generated_school_relations=0)
                for site in sites:
                    ns = site['coordinate_group_building_names']
                    exact = all(norm(n) == record['residential_name'] for n in ns)
                    partial = any(norm(n) in record['residential_name'] or record['residential_name'] in norm(n) for n in ns)
                    if not (exact or partial):
                        continue
                    sid = site['coordinate_group_id']
                    old = [p for p in prior if p['district'] == '龙岗区' and p['site_id'] == sid]
                    entry = dict(site_id=sid, site_names=ns,
                        prior_memberships=[{k: p.get(k) for k in ['street', 'community', 'status', 'reason']} for p in old])
                    record['exact_coordinate_group_matches' if exact else 'ambiguous_name_candidates'].append(entry)
                record['profile_identities'] = sorted({(p['street'], p['community']) for p in profiles
                    if p['district'] == '龙岗区' and p['community'] == record['community']})
                records.append(record)
    assert len(records) == 33
    summary = dict(rows=len(records), rows_per_page=dict(Counter(r['page'] for r in records)),
        exact_matched_sites=len({x['site_id'] for r in records for x in r['exact_coordinate_group_matches']}),
        exact_without_prior=len({x['site_id'] for r in records for x in r['exact_coordinate_group_matches'] if not x['prior_memberships']}),
        target_gold_read=False, target_school_polygons_read=False, admitted=False, generated_school_relations=0)
    F.write(OUT / 'review_queue.json', dict(summary=summary, statements=records))
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in
        [Path(__file__), path, sitespath, profilespath] + priorpaths},
        files={'review_queue.json': F.sha(OUT / 'review_queue.json')},
        visual_review='Both PDF pages rendered and read before extraction; merged street cells retained.'))
    print(summary, flush=True)
    for r in records:
        if r['exact_coordinate_group_matches']:
            print(r['page'], r['street'], r['community'], r['residential_name'],
                  'prior', [x['prior_memberships'] for x in r['exact_coordinate_group_matches']], flush=True)


if __name__ == '__main__':
    main()
