"""Read the public ZJJ property catalogue, retaining geography fields only.

The two methods are invoked by the public, login-exempt PropertyProjectInformation
pages linked from the ZJJ website. No account, token, private API or mutation.
Current-source evidence is never backdated to the frozen enrollment cutoff.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import time
from urllib.parse import urlencode
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'public_property_memberships_current'
ENDPOINT = 'https://zjj.sz.gov.cn/szwyweb/szwyweb/PropertyProjectControllerSvr.assx/'
FIELDS = ['EaAreaName', 'EaAreaCode', 'PropertyCantonName', 'StreetOfficeName',
          'ResidentsCommitteeName', 'EBuildingTypeName', 'EaAreaExternalCode',
          'EastScope', 'WestScope', 'NorthScope', 'SouthScope']
PAGE_SIZE = 20  # The public application's default page size.
DISTRICTS = ('龙华区', '龙岗区')


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request(method, parameters, destination):
    assert method in {'GetPrejectDetailList', 'GetProjectDetails'}
    if destination.exists():
        return F.read(destination)
    time.sleep(1)
    started = stamp()
    run = subprocess.run(['curl.exe', '--silent', '--show-error', '--noproxy', '*',
        '--connect-timeout', '8', '--max-time', '30', ENDPOINT + method,
        '-H', 'Content-Type: application/x-www-form-urlencoded',
        '--data-raw', urlencode(parameters), '--write-out', '\n%{http_code}'],
        capture_output=True, timeout=40)
    content, _, status = run.stdout.rpartition(b'\n')
    receipt = dict(endpoint=ENDPOINT + method, parameters=parameters,
        fetched_utc=started, completed_utc=stamp(), http_status=status.decode('ascii', 'replace'),
        curl_exit_code=run.returncode, response_bytes=len(content),
        raw_response_sha256=hashlib.sha256(content).hexdigest(),
        raw_response_retained=False, retained_fields=FIELDS,
        temporal_status='current_query_no_historical_membership_timestamp',
        admitted_to_frozen_2026_inputs=False)
    if run.returncode or status != b'200':
        F.write(destination.with_suffix('.failure.json'), receipt)
        raise RuntimeError('Public query failed; receipt saved, no automatic retry')
    try:
        response = json.loads(content.decode('utf-8-sig'))
        assert isinstance(response, dict) and 'error_respone' not in response
        if method == 'GetPrejectDetailList':
            assert isinstance(response['TotalCount'], int) and isinstance(response['_Items'], list)
            receipt.update(total_count=response['TotalCount'],
                items=[{k: row.get(k) for k in FIELDS} for row in response['_Items']])
            assert len(receipt['items']) <= parameters['maxSize']
        else:
            assert response['EaAreaCode'] == parameters['projectDetailsID']
            receipt['item'] = {k: response.get(k) for k in FIELDS}
    except (ValueError, KeyError, AssertionError):
        F.write(destination.with_suffix('.failure.json'), receipt)
        raise RuntimeError('Unexpected public response; stopped without retaining personal fields')
    F.write(destination, receipt)
    return receipt


def prepare():
    path = OUT / 'plan.json'
    if path.exists():
        plan = F.read(path)
        assert plan['script_sha256'] == F.sha(__file__)
        return
    pages = ROOT / 'residential_roster_followup'
    F.write(path, dict(created_utc=stamp(), districts=list(DISTRICTS), page_size=PAGE_SIZE,
        source_page='https://zjj.sz.gov.cn/ztfw/wyrq/wycx/',
        application='https://zjj.sz.gov.cn/szwyweb/#/PropertyProjectInformationInquiryList',
        script_sha256=F.sha(__file__),
        provenance={str(p): F.sha(p) for p in [pages / 'property_query_catalog.html',
            pages / 'property_query.html', pages / 'property_app.html', pages / 'property_app.js']},
        data_policy='Geography fields only; omit people, phones, apartment addresses. No credentials.',
        matching='Exact normalized names, same district; retain ambiguous project matches for review.',
        cutoff_policy='Current lookup only. Not evidence of pre-2026-06-01 membership. No automatic historical promotion.',
        target_gold_read=False, target_polygons_read=False))


def inventory(budget):
    used = 0
    for district in DISTRICTS:
        offset = 0
        while True:
            dest = OUT / 'lists' / district / f'{offset:05d}.json'
            if not dest.exists() and used >= budget:
                return
            fresh = not dest.exists()
            row = request('GetPrejectDetailList', dict(orgName='', areaName='', cantonName=district,
                startIndex=offset, maxSize=PAGE_SIZE), dest)
            used += int(fresh)
            assert all(x['PropertyCantonName'] == district for x in row['items']), district
            if fresh:
                print('PAGE', district, offset, len(row['items']), 'total', row['total_count'], flush=True)
            if offset + PAGE_SIZE >= row['total_count']:
                break
            assert len(row['items']) == PAGE_SIZE
            offset += PAGE_SIZE


def review_queue():
    reports = {}
    matches = []
    for district in DISTRICTS:
        pages = sorted((OUT / 'lists' / district).glob('*.json'))
        rows = [F.read(p) for p in pages]
        totals = {r['total_count'] for r in rows}
        offsets = [r['parameters']['startIndex'] for r in rows]
        expected = list(range(0, next(iter(totals)), PAGE_SIZE)) if len(totals) == 1 else []
        items = [r for page in rows for r in page['items']]
        codes = [r['EaAreaCode'] for r in items]
        complete = bool(rows) and len(totals) == 1 and offsets == expected and len(items) == next(iter(totals)) and len(set(codes)) == len(codes)
        reports[district] = dict(pages=len(pages), reported_totals=sorted(totals),
            rows=len(items), unique_project_codes=len(set(codes)), complete=complete)
        sites = F.read(ROOT / 'round03_partial_labels/inputs' / f'{district}.json')['sites']
        for site in sites:
            site_names = site['coordinate_group_building_names']
            hit = [r for r in items if all(norm(n) == norm(r['EaAreaName'] or '') for n in site_names)]
            if hit:
                matches.append(dict(district=district, site_id=site['coordinate_group_id'],
                    site_names=site_names, projects=hit,
                    status='pending_current_detail_and_identity_review', historical_admission=False))
    F.write(OUT / 'inventory_status.json', dict(districts=reports,
        exact_matched_sites=len(matches), exact_matched_sites_by_district={d: sum(r['district'] == d for r in matches) for d in DISTRICTS},
        all_lists_complete=all(v['complete'] for v in reports.values()),
        source_inputs={str(ROOT / 'round03_partial_labels/inputs' / f'{d}.json'):
            F.sha(ROOT / 'round03_partial_labels/inputs' / f'{d}.json') for d in DISTRICTS},
        target_gold_read=False, changed_model_inputs=False))
    F.write(OUT / 'exact_match_review_queue.json', matches)
    print('INVENTORY', reports, 'matched sites', len(matches), flush=True)


def details(budget):
    status = F.read(OUT / 'inventory_status.json')
    assert status['all_lists_complete'], 'Finish and audit district lists before querying matches'
    queue = F.read(OUT / 'exact_match_review_queue.json')
    # Query every exact candidate, including ambiguous candidates; no school outcomes.
    codes = sorted({p['EaAreaCode'] for r in queue for p in r['projects']})
    used = 0
    for code in codes:
        dest = OUT / 'details' / f'{code}.json'
        if dest.exists():
            continue
        if used >= budget:
            break
        r = request('GetProjectDetails', dict(projectDetailsID=code), dest)
        used += 1
        x = r['item']
        print('DETAIL', code, x['EaAreaName'], x['StreetOfficeName'], x['ResidentsCommitteeName'], flush=True)
    loaded = [F.read(p) for p in (OUT / 'details').glob('*.json') if not p.name.endswith('.failure.json')]
    F.write(OUT / 'detail_status.json', dict(expected_project_codes=len(codes), downloaded=len(loaded),
        with_community=sum(bool(r['item']['ResidentsCommitteeName']) for r in loaded),
        complete=len(loaded) == len(codes), historical_admission=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'inventory', 'queue', 'details'])
    parser.add_argument('--budget', type=int, default=25)
    args = parser.parse_args()
    F.guard()
    prepare()
    if args.phase == 'inventory':
        inventory(args.budget)
        review_queue()
    elif args.phase == 'queue':
        review_queue()
    elif args.phase == 'details':
        details(args.budget)
