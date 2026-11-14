"""Review all six newly located official community pages and explicit properties."""
from pathlib import Path
from collections import Counter
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from shenzhen_community_facts import MembershipScopeProof
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'dated_address_followup_sources'
OUT = ROOT / 'yuanshan_profile_review_v1'
PAGES = [('xikeng', '西坑社区', '12709490'), ('anliang', '安良社区', '12709489'),
         ('dakang', '大康社区', '12709488'), ('baoan', '保安社区', '12709487'),
         ('yinhe', '银荷社区', '12709486'), ('heao', '荷坳社区', '12709485')]


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve reviewed source')
    directory = SOURCE / 'yuanshan_profile_directory.html'
    links = {a['href'] for a in BeautifulSoup(directory.read_bytes(), 'html.parser').select('a[href]')}
    paths = [Path(__file__), directory]
    reviewed = []
    for stem, community, ident in PAGES:
        p = SOURCE / f'{stem}_profile.html'; paths.append(p)
        url = f'https://www.lg.gov.cn/bmzz/ysjdb/wmfw/ggfw/sqgzz/content/post_{ident}.html'
        assert url in links
        soup = BeautifulSoup(p.read_bytes(), 'html.parser')
        assert soup.find('meta', attrs={'name': 'PubDate'})['content'] == '2026-03-30'
        paragraphs = [n.get_text('', strip=True) for n in soup.select('p')]
        relevant = [text for text in paragraphs if text.startswith(community)]
        assert len(relevant) == 1
        reviewed.append(dict(district='龙岗区', street='园山街道', community=community,
            paragraph=relevant[0], source=dict(url=url, raw_file=str(p), sha256=F.sha(p),
                page_dates=['2026-03-30']), school_locations_not_used=True))
    baoan, = [p for p in reviewed if p['community'] == '保安社区']
    names = ['星河智荟', '星河盛境', '卓弘星辰', '名居山河里', '信义金御山居']
    quote = '和' + '、'.join(names) + '5个花园小区'
    assert quote in baoan['paragraph']
    site_path = ROOT / 'round03_partial_labels/inputs/龙岗区.json'; paths.append(site_path)
    sites = F.read(site_path)['sites']
    prior_paths = [ROOT / sub / 'review_v1/membership_facts.json' for sub in
                  ['community_membership_sources', 'municipal_residential_membership_sources']]
    paths += prior_paths
    priors = [f for p in prior_paths for part in ['facts', 'pending'] for f in F.read(p).get(part, [])]
    facts = []; ledger = []
    for name in names:
        matches = [s for s in sites if set(s['coordinate_group_building_names']) == {name}]
        if len(matches) != 1:
            ledger.append(dict(name=name, decision='no_unique_exact_dataset_entity', matches=len(matches)))
            continue
        site = matches[0]; sid = site['coordinate_group_id']
        prior = [f for f in priors if f['district'] == '龙岗区' and f['site_id'] == sid]
        if any(f.get('reason') or (f.get('street'), f.get('community')) != ('园山街道', '保安社区') for f in prior):
            ledger.append(dict(name=name, site_id=sid, decision='pending_prior_identity_conflict', prior=prior))
            continue
        if any(f.get('status') == 'source_supported_pre_enrollment' for f in prior):
            ledger.append(dict(name=name, site_id=sid, decision='duplicate_accepted_membership'))
            continue
        evidence = dict(kind='official_profile_geographical_membership', district='龙岗区', street='园山街道',
            community='保安社区', site_id=sid, site_names=[name], source=baoan['source'], quote=baoan['paragraph'])
        facts.append(dict(district='龙岗区', street='园山街道', community='保安社区', site_id=sid,
            site_names=[name], evidence=[evidence], pre_enrollment_evidence=[evidence], source_dates=['2026-03-30'],
            status='source_supported_pre_enrollment', complete_school_set=False, enrollment_eligibility=False,
            membership_absence_is_negative=False, source_age_limitation=True))
        ledger.append(dict(name=name, site_id=sid, decision='accepted_dated_explicit_membership',
            basis='Official profile explicitly lists the exact named garden community under its jurisdiction.'))
    profile_path = ROOT / 'community_membership_sources/profile_inventory.json'; paths.append(profile_path)
    policy_path = F.REPO / 'data/深圳市/边界合取单项修复输入_20261112/龙岗区/reviewed_policy.json'; paths.append(policy_path)
    inv = F.read(policy_path)['school_scope_inventory']
    parser = BoundaryConjunctionParserV2(inv, parse_scope)
    asts = [parser.parse(row['base_scope_text']) for row in inv]
    relations = MembershipScopeProof(inv, asts, F.read(profile_path)['profiles'], '龙岗区').derive(facts)
    F.write(OUT / 'membership_facts.json', dict(facts=facts, cutoff='2026-06-01', target_gold_read=False,
        target_polygons_read=False, unknown_is_negative=False, adopted=False))
    F.write(OUT / 'review.json', dict(profiles=reviewed, property_review=ledger,
        summary=dict(profiles_reviewed=6, explicit_garden_properties=5, new_memberships=len(facts),
            decisions=dict(Counter(r['decision'] for r in ledger)), derivable_school_relations=len(relations)),
        explanation='Community membership alone does not resolve resident-group exclusions. No excluded subgroup is silently marked absent.'))
    F.write(OUT / 'school_facts.json', dict(facts=relations,
        school_asts=[dict(school_id=r['school_id'], school_name=r['school_name'],
            source_text=r['base_scope_text'], ast=a) for r,a in zip(inv, asts)]))
    paths += [F.REPO / 'scripts/shenzhen_community_facts.py', F.REPO / 'scripts/shenzhen_boundary_conjunction_v2.py',
              F.REPO / 'scripts/build_shenzhen_aligned_inputs.py']
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in paths}, files={
        str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*.json')}))
    print('YUANSHAN_PROFILE_REVIEW', F.read(OUT / 'review.json')['summary'], flush=True)


if __name__ == '__main__':
    main()
