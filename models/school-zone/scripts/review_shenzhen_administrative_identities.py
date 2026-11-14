"""Freeze a complete unmatched-term review and derive source-only relations."""
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_community_facts import MembershipScopeProof
from shenzhen_administrative_identity import ReviewedAdministrativeScopeProof

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'administrative_identity_review_v1'
POLICY = F.REPO / 'data/深圳市/边界合取单项修复输入_20261112'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed identity review')
    profile_path = ROOT / 'community_membership_sources/profile_inventory.json'
    term_path = ROOT / 'community_identity_version_audit/report.json'
    profiles = F.read(profile_path)['profiles']
    terms = F.read(term_path)['districts']
    sources = {str(p): F.sha(p) for p in [Path(__file__), profile_path, term_path,
        F.REPO / 'scripts/shenzhen_administrative_identity.py',
        F.REPO / 'scripts/shenzhen_community_facts.py',
        F.REPO / 'scripts/shenzhen_boundary_conjunction_v2.py',
        F.REPO / 'scripts/build_shenzhen_aligned_inputs.py']}
    identities = []
    # Both literal names and one-station/one-committee status occur in these profiles.
    for community in ['龙胜社区', '浪口社区']:
        p, = [p for p in profiles if p['district'] == '龙华区' and
              p['street'] == '大浪街道' and p['community'] == community]
        paragraph, = [s for s in p['paragraphs'] if '一站一居' in s and community + '居委会' in s]
        raw = ROOT / 'community_membership_sources' / p['source']['raw_file']
        assert F.sha(raw) == p['source']['sha256']
        soup = BeautifulSoup(raw.read_bytes(), 'html.parser')
        assert paragraph in soup.get_text('', strip=True)
        published = soup.find('meta', attrs={'name': 'PubDate'})['content'][:10]
        assert published == p['source']['page_dates'][0][:10] <= '2026-06-01'
        sources[str(raw)] = F.sha(raw)
        identities.append(dict(district='龙华区', street='大浪街道', community=community,
            scope_term=community + '居委会', relation='community_and_sole_committee_coextensive',
            decision='accepted_dated_identity', version_not_before=published,
            evidence=[dict(url=p['source']['url'], publication_date=published,
                quote=paragraph, local_file=str(raw), sha256=F.sha(raw))]))
    split_path = ROOT / 'community_identity_version_sources/dalang_adjustment_20230524.html'
    soup = BeautifulSoup(split_path.read_bytes(), 'html.parser')
    assert soup.find('meta', attrs={'name': 'PubDate'})['content'][:10] == '2023-05-24'
    split_url = 'https://www.szlhq.gov.cn/lhdljdb/gkmlpt/content/10/10610/post_10610579.html'
    paragraphs = [p.get_text('', strip=True) for p in soup.select('p')]
    for base in ['水围', '赖屋山', '上横朗']:
        phrase = base + '社区下辖' + base + '居委会'
        matches = [s for s in paragraphs if phrase in s]
        assert len(matches) == 1
        # Restrict the evidence to this community's clause, preserving its sole status.
        clause = matches[0].split(phrase, 1)[1].split('。', 1)[0]
        quote = phrase + clause + '。'
        assert '属一站一居' in quote and quote in matches[0]
        identities.append(dict(district='龙华区', street='大浪街道', community=base + '社区',
            scope_term=base + '社区居委会', relation='community_and_sole_committee_coextensive',
            decision='accepted_dated_identity', version_not_before='2023-05-24',
            evidence=[dict(url=split_url, publication_date='2023-05-24', quote=quote,
                local_file=str(split_path), sha256=F.sha(split_path))]))
    sources[str(split_path)] = F.sha(split_path)
    byterm = {(r['district'], r['scope_term']): r for r in identities}
    ledger = []
    for district, report in terms.items():
        for term, schools in report['unmatched'].items():
            mapping = byterm.get((district, term))
            reason = 'No reviewed dated equivalence; parent, child, shorthand and version remain distinct.'
            if term == '同胜社区居委会':
                reason = 'Current Tongsheng contains Xiahenglang committee; old Tongsheng split into Huarong/Laiwushan. No same-name rewrite.'
            if term == '陶元社区居委会':
                reason = 'Profile lists Taoxia and Yuanfen committees; school wording is not an established sole-committee identity.'
            ledger.append(dict(district=district, scope_term=term, school_count=len(schools),
                decision=mapping['decision'] if mapping else 'pending_no_proven_equivalence',
                reason='Explicit same-territory sole committee, with dated version guard.' if mapping else reason,
                official_school_texts=schools))
    assert len(ledger) == 29 and len(identities) == 5
    fact_paths = [ROOT / sub / 'review_v1/membership_facts.json' for sub in
                  ['community_membership_sources', 'municipal_residential_membership_sources']]
    facts = [f for p in fact_paths for f in F.read(p)['facts']]
    assert len(facts) == len({(f['district'], f['site_id']) for f in facts}) == 227
    sources.update({str(p): F.sha(p) for p in fact_paths})
    coverage = {}
    F.write(OUT / 'identities.json', dict(identities=identities, ledger=ledger,
        decisions=dict(Counter(r['decision'] for r in ledger)), cutoff='2026-06-01',
        unknown_is_negative=False, target_gold_read=False, target_polygon_read=False))
    F.write(OUT / 'membership_facts.json', dict(facts=facts, new_property_memberships=0,
        current_registry_memberships_used=0, cutoff='2026-06-01'))
    for district in F.SZ.DISTRICTS:
        policy_path = POLICY / district / 'reviewed_policy.json'
        sources[str(policy_path)] = F.sha(policy_path)
        inv = F.read(policy_path)['school_scope_inventory']
        parser = BoundaryConjunctionParserV2(inv, parse_scope)
        asts = [parser.parse(r['base_scope_text']) for r in inv]
        local = [f for f in facts if f['district'] == district]
        old = MembershipScopeProof(inv, asts, profiles, district).derive(local)
        new = ReviewedAdministrativeScopeProof(inv, asts, profiles, district, identities).derive(local)
        keys = lambda rels: {(r['site_id'], r['school_id']) for r in rels}
        assert keys(old) <= keys(new)
        additional = [r for r in new if (r['site_id'], r['school_id']) not in keys(old)]
        assert all(any(w['kind'] == 'reviewed_administrative_identity' for w in r['proof']) for r in additional)
        F.write(OUT / 'school_facts' / f'{district}.json', dict(facts=additional,
            baseline_relations=old, reviewed_relations=new,
            all_school_ast_audit=[dict(school_id=r['school_id'], school_name=r['school_name'],
                source_text=r['base_scope_text'], ast=a, evidence=r['evidence']) for r, a in zip(inv, asts)],
            target_gold_read=False, target_polygon_read=False))
        coverage[district] = dict(memberships=len(local), old_relations=len(old),
            reviewed_relations=len(new), added_relations=len(additional),
            added_sites=len({r['site_id'] for r in additional}))
        for r in additional:
            print('ADDED', district, r['building_names'], r['school_name'], flush=True)
    F.write(OUT / 'summary.json', dict(coverage=coverage, no_new_model_training=True,
        current_registry_used=False, adopted=False))
    F.write(OUT / 'manifest.json', dict(sources=sources, files={
        str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*.json')}))
    print('IDENTITY_REVIEW', coverage, flush=True)


if __name__ == '__main__':
    main()
