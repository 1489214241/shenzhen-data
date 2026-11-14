"""Independent inventory replay and current-only membership review.

This does not change the historical input admission boundary or school outputs.
"""
from pathlib import Path
import argparse
from collections import Counter
import re
import unicodedata
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'public_property_memberships_current'
OUT = ROOT / 'public_property_memberships_review'
DISTRICTS = ('龙华区', '龙岗区')
FIELDS = {'EaAreaName', 'EaAreaCode', 'PropertyCantonName', 'StreetOfficeName',
          'ResidentsCommitteeName', 'EBuildingTypeName', 'EaAreaExternalCode',
          'EastScope', 'WestScope', 'NorthScope', 'SouthScope'}


def normalize(value):
    return re.sub(r'[\s\u3000·•・]', '', unicodedata.normalize('NFKC', value))


def inventory():
    F.guard()
    dest = OUT / 'inventory_audit.json'
    if dest.exists():
        raise FileExistsError('Preserve completed audit')
    plan = F.read(SOURCE / 'plan.json')
    assert F.sha(F.REPO / 'scripts/acquire_shenzhen_public_property_memberships.py') == plan['script_sha256']
    for p, h in plan['provenance'].items():
        assert F.sha(p) == h
    oldstatus = F.read(SOURCE / 'inventory_status.json')
    assert oldstatus['all_lists_complete']
    for p, h in oldstatus['source_inputs'].items():
        assert F.sha(p) == h
    replay = {}
    all_files = [SOURCE / 'plan.json', SOURCE / 'inventory_status.json', SOURCE / 'exact_match_review_queue.json']
    summary = {}
    for district in DISTRICTS:
        page_files = sorted((SOURCE / 'lists' / district).glob('*.json'))
        data = [F.read(p) for p in page_files]
        totals = {r['total_count'] for r in data}
        assert len(totals) == 1
        total = totals.pop()
        assert [r['parameters']['startIndex'] for r in data] == list(range(0, total, 20))
        items = []
        for page in data:
            assert page['parameters'] == dict(orgName='', areaName='', cantonName=district,
                startIndex=len(items), maxSize=20)
            assert page['endpoint'].endswith('/GetPrejectDetailList')
            assert page['http_status'] == '200' and page['curl_exit_code'] == 0
            assert not page['admitted_to_frozen_2026_inputs']
            assert len(page['items']) == min(20, total - len(items))
            assert all(set(x) == FIELDS for x in page['items'])
            assert all(x['PropertyCantonName'] == district for x in page['items'])
            items.extend(page['items'])
        assert len(items) == total == len({x['EaAreaCode'] for x in items})
        sitespath = ROOT / 'round03_partial_labels/inputs' / f'{district}.json'
        sites = F.read(sitespath)['sites']
        name_index = {}
        for item in items:
            name_index.setdefault(normalize(item['EaAreaName'] or ''), []).append(item['EaAreaCode'])
        for site in sites:
            names = {normalize(n) for n in site['coordinate_group_building_names']}
            if len(names) == 1 and next(iter(names)) in name_index:
                replay[(district, site['coordinate_group_id'])] = sorted(name_index[next(iter(names))])
        all_files += page_files + [sitespath]
        summary[district] = dict(pages=len(data), rows=total, unique_project_codes=total,
            exact_matched_sites=sum(k[0] == district for k in replay))
    queue = F.read(SOURCE / 'exact_match_review_queue.json')
    frozen = {(r['district'], r['site_id']): sorted(p['EaAreaCode'] for p in r['projects']) for r in queue}
    assert len(frozen) == len(queue) and frozen == replay
    sources = {str(p): F.sha(p) for p in all_files + [Path(__file__)]}
    F.write(dest, dict(status='passed', summary=summary, exact_matched_sites=len(queue),
        target_gold_read=False, source_files=sources,
        limitation='Checks minimized response consistency and matching; original raw responses were hashed but not retained. No historical validity claim.'))
    print('INVENTORY_AUDIT', summary, 'matched', len(queue), flush=True)


def details():
    F.guard()
    dest = OUT / 'current_membership_review.json'
    if dest.exists():
        raise FileExistsError('Preserve current membership review')
    inventory_audit = F.read(OUT / 'inventory_audit.json')
    for p, h in inventory_audit['source_files'].items():
        assert F.sha(p) == h
    status = F.read(SOURCE / 'detail_status.json')
    assert status['complete']
    queue = F.read(SOURCE / 'exact_match_review_queue.json')
    expected = {p['EaAreaCode'] for r in queue for p in r['projects']}
    files = [p for p in (SOURCE / 'details').glob('*.json') if not p.name.endswith('.failure.json')]
    assert {p.stem for p in files} == expected
    details_by_code = {}
    for path in files:
        receipt = F.read(path)
        assert receipt['parameters'] == {'projectDetailsID': path.stem}
        assert receipt['endpoint'].endswith('/GetProjectDetails')
        assert receipt['http_status'] == '200' and receipt['curl_exit_code'] == 0
        assert not receipt['admitted_to_frozen_2026_inputs']
        assert set(receipt['item']) == FIELDS and receipt['item']['EaAreaCode'] == path.stem
        details_by_code[path.stem] = receipt
    priorpaths = [ROOT / 'community_membership_sources/review_v1/membership_facts.json',
                 ROOT / 'municipal_residential_membership_sources/review_v1/membership_facts.json']
    prior = [(section, r) for path in priorpaths for section in ['facts', 'pending']
             for r in F.read(path).get(section, [])]
    rows = []
    for entry in queue:
        candidates = [details_by_code[p['EaAreaCode']] for p in entry['projects']]
        identity_changed = any(r['item']['PropertyCantonName'] != entry['district'] or
            not all(normalize(n) == normalize(r['item']['EaAreaName'] or '') for n in entry['site_names'])
            for r in candidates)
        reason = 'pending_multiple_projects' if len(candidates) != 1 else 'current_identity_supported'
        if identity_changed:
            reason = 'pending_list_detail_identity_change'
        item = candidates[0]['item']
        if not item['StreetOfficeName'] or not item['ResidentsCommitteeName']:
            reason = 'missing_street_or_community'
        historical = [dict(section=section, street=r.get('street'), community=r.get('community'),
            status=r.get('status'), reason=r.get('reason')) for section, r in prior
            if r['district'] == entry['district'] and r['site_id'] == entry['site_id']]
        same = [r for r in historical if (r['street'], r['community']) ==
                (item['StreetOfficeName'], item['ResidentsCommitteeName'])]
        differing = [r for r in historical if r['street'] and r['community'] and
            (r['street'], r['community']) != (item['StreetOfficeName'], item['ResidentsCommitteeName'])]
        rows.append(dict(district=entry['district'], site_id=entry['site_id'], site_names=entry['site_names'],
            current_query=candidates, decision=reason, previous_review=historical,
            agrees_with_previous=bool(same), differs_from_previous=bool(differing),
            historical_admission=False, school_relations_generated=0))
    summary = dict(total_sites=len(rows), decisions=dict(Counter(r['decision'] for r in rows)),
        by_district={d: dict(Counter(r['decision'] for r in rows if r['district'] == d)) for d in DISTRICTS},
        agrees_with_previous=sum(r['agrees_with_previous'] for r in rows),
        differs_from_previous=sum(r['differs_from_previous'] for r in rows),
        without_previous_membership=sum(not r['previous_review'] for r in rows),
        historical_admission=False, school_relations_generated=0, model_inputs_changed=False)
    F.write(dest, dict(summary=summary, rows=rows,
        interpretation='Current official property registry evidence. Differences may reflect time, semantics or identity; no historical overwrite.'))
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in
        [Path(__file__), OUT / 'inventory_audit.json', SOURCE / 'detail_status.json'] + files + priorpaths},
        files={'current_membership_review.json': F.sha(dest)}))
    print('CURRENT_MEMBERSHIP_REVIEW', summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['inventory', 'details'])
    globals()[p.parse_args().phase]()
