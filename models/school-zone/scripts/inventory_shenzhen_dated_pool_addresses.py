"""Inventory an entire official address catalogue, never school outcomes.

Only facility names and geographical addresses are retained in the review queue.
Address mentions are candidates, not automatic whole-property membership facts.
"""
from collections import Counter
from pathlib import Path
import re
import unicodedata
import pdfplumber
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'dated_address_sources'
OUT = ROOT / 'dated_address_inventory_v1'
NOTICE = SOURCE / 'swimming_notice_20250929.html'
NOTICE_URL = 'https://wjw.sz.gov.cn/gkmlpt/content/12/12415/post_12415968.html'
SPECS = [
    ('swimming_catalog_a.pdf', 54, 'https://wjw.sz.gov.cn/attachment/1/1630/1630056/12415968.pdf'),
    ('swimming_catalog_b.pdf', 848, 'https://wjw.sz.gov.cn/attachment/1/1630/1630055/12415968.pdf'),
    ('swimming_catalog_c_current.pdf', 257, 'https://wjw.sz.gov.cn/attachment/1/1757/1757899/12415968.pdf'),
]


def norm(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s or ''))


def extract(path, expected, url):
    rows = []
    with pdfplumber.open(path) as doc:
        metadata = {k: doc.metadata.get(k) for k in ['CreationDate', 'ModDate']}
        dates = [str(v)[2:10] for v in metadata.values() if v and str(v).startswith('D:')]
        # A 2025 page title cannot backdate an attachment modified in October 2026.
        temporal = 'post_cutoff_attachment_version_pending' if any(d > '20260601' for d in dates) else 'dated_source_candidate'
        for page_no, page in enumerate(doc.pages, 1):
            tables = page.find_tables()
            assert len(tables) == 1, (path, page_no, len(tables))
            table = tables[0]
            extracted = table.extract()
            assert len(extracted) == len(table.rows)
            for row_index, (cells, geometry) in enumerate(zip(extracted, table.rows)):
                assert len(cells) == 5, (path, page_no, row_index)
                number = norm(cells[0])
                if number == '序号':
                    assert norm(cells[1]) == '单位名称' and norm(cells[2]) == '单位地址'
                    continue
                fragment = dict(page=page_no, table_row=row_index + 1,
                    row_bbox=list(geometry.bbox), name_bbox=list(geometry.cells[1]),
                    address_bbox=list(geometry.cells[2]), unit_name_fragment=norm(cells[1]),
                    address_fragment=norm(cells[2]))
                if number.isdigit():
                    assert int(number) == len(rows) + 1, (path, page_no, number, len(rows))
                    rows.append(dict(row=int(number), unit_name=norm(cells[1]), address=norm(cells[2]),
                        fragments=[fragment], source_pdf=str(path), source_url=url, source_sha256=F.sha(path),
                        notice_url=NOTICE_URL, publication_date='2025-09-29', temporal_status=temporal))
                else:
                    assert not number and rows and row_index == 0
                    assert rows[-1]['fragments'][-1]['page'] == page_no - 1
                    assert cells[1] or cells[2], (path, page_no, cells)
                    rows[-1]['unit_name'] += norm(cells[1])
                    rows[-1]['address'] += norm(cells[2])
                    rows[-1]['fragments'].append(fragment)
        assert len(rows) == expected
        return rows, dict(file=path.name, expected_rows=expected, rows=len(rows), pages=len(doc.pages),
                         metadata=metadata, temporal_status=temporal,
                         continued_rows=sum(len(r['fragments']) > 1 for r in rows))


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve catalogue inventory')
    soup = BeautifulSoup(NOTICE.read_bytes(), 'html.parser')
    assert soup.find('meta', attrs={'name': 'PubDate'})['content'] == '2025-09-29 08:55'
    links = {a['href'] for a in soup.select('a[href]')}
    assert all(url in links for _, _, url in SPECS)
    assert '1159' in soup.get_text('', strip=True)
    rows = []; catalogs = []
    for name, expected, url in SPECS:
        local, report = extract(SOURCE / name, expected, url)
        rows.extend(local); catalogs.append(report)
    assert len(rows) == 1159
    sites = []
    site_paths = []
    for district in ['龙华区', '龙岗区']:
        path = ROOT / 'round03_partial_labels/inputs' / f'{district}.json'
        site_paths.append(path)
        sites.extend(dict(district=district, site_id=r['coordinate_group_id'],
                          site_names=r['coordinate_group_building_names']) for r in F.read(path)['sites'])
    prior_paths = [ROOT / sub / 'review_v1/membership_facts.json' for sub in
                  ['community_membership_sources', 'municipal_residential_membership_sources']]
    prior = {(f['district'], f['site_id']): f for p in prior_paths for f in F.read(p)['facts']}
    candidates = []; local_rows = []
    for row in rows:
        district = next((d for d in ['龙华区', '龙岗区'] if row['address'].startswith('深圳市' + d)), None)
        if district is None:
            continue
        suffix = row['address'][len('深圳市' + district):]
        match = re.match(r'(?P<street>[\u4e00-\u9fff]{2,6}街道)(?P<community>[\u4e00-\u9fff]{2,8}社区)', suffix)
        geography = match.groupdict() if match else {}
        local_rows.append(dict(**row, district=district, explicit_geography=geography))
        # Full literal occurrence is only a screening step, not entity proof.
        for site in sites:
            if site['district'] != district:
                continue
            names = {norm(n) for n in site['site_names']}
            if len(names) != 1:
                continue
            name = next(iter(names))
            if name not in row['address'] and name not in row['unit_name']:
                continue
            old = prior.get((district, site['site_id']))
            candidates.append(dict(**site, source_row=row, explicit_geography=geography,
                literal_in_address=name in row['address'], literal_in_unit=name in row['unit_name'],
                previous_accepted_identity=dict(street=old['street'], community=old['community']) if old else None,
                historical_admission=False, decision='pending_manual_entity_and_address_extent_review',
                limitations=['Facility address may cover only a phase or part of a development.',
                    'A substring hit is not proof that the full coordinate-group entity is identical.',
                    'Source version and administrative identity must be checked before deriving school relations.']))
    F.write(OUT / 'catalogue_address_rows.json', dict(rows=rows, catalogs=catalogs,
        rating_as_feature=False, target_gold_read=False, target_polygons_read=False))
    F.write(OUT / 'review_queue.json', dict(rows=candidates, local_address_rows=local_rows,
        summary=dict(total_catalogue_rows=1159, district_address_rows=len(local_rows),
            literal_match_pairs=len(candidates), matched_sites=len({(r['district'], r['site_id']) for r in candidates}),
            matched_sites_without_previous_accepted=len({(r['district'], r['site_id']) for r in candidates if not r['previous_accepted_identity']}),
            dated_explicit_address_sites_without_previous_accepted=len({(r['district'], r['site_id']) for r in candidates
                if not r['previous_accepted_identity'] and r['explicit_geography'] and r['literal_in_address']
                and r['source_row']['temporal_status'] == 'dated_source_candidate'})),
        no_school_relations_generated=True, model_inputs_changed=False))
    paths = [Path(__file__), NOTICE] + [SOURCE / name for name, _, _ in SPECS] + site_paths + prior_paths
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in paths}, files={
        str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*.json')}))
    print('ADDRESS_CATALOGUE', F.read(OUT / 'review_queue.json')['summary'], flush=True)
    print('VERSIONS', catalogs, flush=True)


if __name__ == '__main__':
    main()
