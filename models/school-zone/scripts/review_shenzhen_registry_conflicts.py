"""Review every registry conflict without replacing dated historical inputs.

The current registry is a discovery source, not a historical membership oracle.
No school answers, school polygons, or model scores are read here.
"""
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'registry_conflict_review_v1'
NEWS = ROOT / 'registry_conflict_sources/jinyun_20251021.html'
NEWS_URL = 'https://www.lg.gov.cn/bmzz/bjjdb/jddt/bjxw/content/post_12445830.html'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed conflict review')
    review_path = ROOT / 'public_property_memberships_review/current_membership_review.json'
    manifest_path = review_path.parent / 'manifest.json'
    manifest = F.read(manifest_path)
    for p, h in manifest['sources'].items():
        assert F.sha(p) == h, p
    assert F.sha(review_path) == manifest['files'][review_path.name]
    review = F.read(review_path)
    prior_paths = [ROOT / sub / 'review_v1/membership_facts.json' for sub in
                   ['community_membership_sources', 'municipal_residential_membership_sources']]
    prior = [f for p in prior_paths for f in F.read(p)['facts']]
    profiles_path = ROOT / 'community_membership_sources/profile_inventory.json'
    profiles = F.read(profiles_path)['profiles']
    soup = BeautifulSoup(NEWS.read_bytes(), 'html.parser')
    assert soup.find('meta', attrs={'name': 'PubDate'})['content'] == '2025-10-21'
    news_quote = '深圳市龙岗区布吉街道文景社区金运花园'
    assert news_quote in soup.get_text('', strip=True)
    sources = {str(p): F.sha(p) for p in
               [Path(__file__), review_path, manifest_path, profiles_path, NEWS] + prior_paths}
    rows = []
    for row in review['rows']:
        if not row['differs_from_previous']:
            continue
        matches = [f for f in prior if (f['district'], f['site_id']) ==
                   (row['district'], row['site_id'])]
        identities = {(f['street'], f['community']) for f in matches}
        assert len(identities) == 1 and matches
        evidence = [e for f in matches for e in f['pre_enrollment_evidence']]
        assert all(e['source']['page_dates'] and
                   all(date[:10] <= '2026-06-01' for date in e['source']['page_dates'])
                   for e in evidence)
        assert all(any(name in e['quote'] for name in row['site_names']) for e in evidence)
        identity = next(iter(identities))
        context = []
        for p in profiles:
            if (p['district'], p['street'], p['community']) != (row['district'], *identity):
                continue
            raw = ROOT / 'community_membership_sources' / p['source']['raw_file']
            assert F.sha(raw) == p['source']['sha256']
            sources[str(raw)] = F.sha(raw)
            raw_soup = BeautifulSoup(raw.read_bytes(), 'html.parser')
            raw_text = raw_soup.get_text('', strip=True)
            for e in evidence:
                if e['source']['url'] == p['source']['url']:
                    assert e['quote'] in raw_text
            for paragraph in p['paragraphs']:
                if any(word in paragraph for word in ['成立', '分设']):
                    assert paragraph in raw_text
                    context.append(dict(quote=paragraph, source=p['source'],
                        interpretation='Administrative history context; does not prove the old registry label is an exact alias.'))
        corroboration = []
        if row['site_names'] == ['金运花园']:
            assert identity == ('布吉街道', '文景社区')
            corroboration.append(dict(url=NEWS_URL, publication_date='2025-10-21',
                source_publisher='龙岗政府在线', original_content_source='南方+',
                quote=news_quote, local_file=str(NEWS), sha256=F.sha(NEWS)))
        rows.append(dict(district=row['district'], site_id=row['site_id'], site_names=row['site_names'],
            registry_identity={k: row['current_query'][0]['item'][k] for k in
                               ['StreetOfficeName', 'ResidentsCommitteeName']},
            dated_identity=dict(street=identity[0], community=identity[1]),
            dated_evidence=evidence, administrative_context=context,
            additional_corroboration=corroboration,
            historical_decision='retain_existing_dated_fact_do_not_overwrite_from_undated_registry',
            current_identity_conflict_resolved=False,
            cause='Unestablished: registry age, administrative level, version or identity must be checked.',
            new_school_relations=0))
    assert len(rows) == review['summary']['differs_from_previous'] == 12
    # Inventory the entire new population without selecting school mistakes.
    queue = []
    for row in review['rows']:
        if row['previous_review']:
            continue
        item = row['current_query'][0]['item']
        queue.append(dict(district=row['district'], site_id=row['site_id'], site_names=row['site_names'],
            street=item['StreetOfficeName'], community=item['ResidentsCommitteeName'],
            project_code=item['EaAreaCode'], historical_admission=False,
            required_evidence='Dated independent official property-to-community statement and administrative version check.'))
    queue.sort(key=lambda r: (r['district'], r['street'], r['community'], r['site_names']))
    assert len(queue) == 358
    F.write(OUT / 'conflicts.json', dict(rows=rows, summary=dict(reviewed=12,
        retained_existing_dated_facts=12, new_dated_corroborations=1,
        registry_overwrites=0, conflicts_with_proven_cause=0),
        target_gold_read=False, target_polygons_read=False, model_inputs_changed=False))
    F.write(OUT / 'new_membership_queue.json', dict(rows=queue, summary=dict(
        total=len(queue), by_district=dict(Counter(r['district'] for r in queue))),
        selection='All 358 previously unrecorded exact matches; no school labels or scores used.'))
    F.write(OUT / 'manifest.json', dict(sources=sources, files={
        str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*.json')}))
    print('REVIEWED 12; retained dated facts 12; new dated corroboration 1; discovery queue 358')


if __name__ == '__main__':
    main()
