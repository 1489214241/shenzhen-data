"""Inventory unresolved administrative identities and an official split notice.

This does not collapse committees into communities or infer membership from
office locations. No output-school labels or property predictions are inputs.
"""
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
OUT = ROOT / 'community_identity_version_audit'
SOURCE = ROOT / 'community_identity_version_sources/dalang_adjustment_20230524.html'
URL = 'https://www.szlhq.gov.cn/lhdljdb/gkmlpt/content/10/10610/post_10610579.html'


def terms(ast):
    if ast['op'] == 'membership':
        yield ast['text']
    for child in ast.get('children', []):
        yield from terms(child)


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve identity version audit')
    profilespath = ROOT / 'community_membership_sources/profile_inventory.json'
    profiles = F.read(profilespath)['profiles']
    soup = BeautifulSoup(SOURCE.read_bytes(), 'html.parser')
    published = soup.find('meta', attrs={'name': 'PubDate'})['content']
    assert published == '2023-05-24 16:02'
    text = soup.get_text('', strip=True)
    clauses = [
        ('浪口社区', ['浪口社区', '水围社区'],
         '将原浪口社区分设为浪口社区和水围社区'),
        ('同胜社区', ['华荣社区', '赖屋山社区'],
         '将原同胜社区分设为华荣社区和赖屋山社区（原社区范围不再使用同胜社区名称）'),
        ('横朗社区', ['上横朗社区', '同胜社区'],
         '将原横朗社区分设为上横朗社区和同胜社区（不再使用横朗社区名称）'),
    ]
    splits = []
    for old, new, quote in clauses:
        assert quote in text
        splits.append(dict(old_community=old, new_communities=new, source_quote=quote,
            publication_date=published, url=URL,
            use='Version review only. Old parent membership cannot select a new child.'))
    committees = [('浪口社区', ['浪口居委会']), ('水围社区', ['水围居委会']),
                  ('华荣社区', ['三合居委会', '谭罗居委会']), ('赖屋山社区', ['赖屋山居委会']),
                  ('上横朗社区', ['上横朗居委会']), ('同胜社区', ['下横朗居委会'])]
    committee_evidence = []
    for community, children in committees:
        quote = community + '下辖' + '和'.join(children)
        assert quote in text
        committee_evidence.append(dict(community=community, committees=children, quote=quote,
                                       automatic_scope_rewrite=False))
    reports = {}
    paths = [Path(__file__), SOURCE, profilespath]
    for district in ['龙华区', '龙岗区']:
        path = ROOT / 'municipal_residential_membership_sources/review_v1/school_facts' / f'{district}.json'
        rows = F.read(path)['all_school_ast_audit']
        paths.append(path)
        known = {p['community'] for p in profiles if p['district'] == district}
        byterm = defaultdict(list)
        for row in rows:
            for name in sorted(set(terms(row['ast']))):
                byterm[name].append(dict(school_name=row['school_name'], school_id=row['school_id'],
                    official_scope_text=row['source_text'], official_sources=row['evidence']))
        missing = {name: values for name, values in byterm.items() if name not in known}
        reports[district] = dict(schools=len(rows), distinct_membership_terms=len(byterm),
            unmatched_terms=len(missing), unmatched=missing,
            interpretation='Literal identifier mismatch only: committees, old names, shorthand or missing profiles. Not proof that official data are absent.')
        print(district, 'membership terms', len(byterm), 'unmatched', len(missing), flush=True)
    priorpath = ROOT / 'community_membership_sources/review_v1/membership_facts.json'
    municipalpath = ROOT / 'municipal_residential_membership_sources/review_v1/membership_facts.json'
    paths += [priorpath, municipalpath]
    affected = []
    for p in [priorpath, municipalpath]:
        for section in ['facts', 'pending']:
            for fact in F.read(p).get(section, []):
                if fact['district'] == '龙华区' and fact.get('street') == '大浪街道' and fact.get('community') in {'浪口社区', '同胜社区', '横朗社区'}:
                    affected.append(dict(source_file=str(p), section=section, site_id=fact['site_id'],
                        names=fact['site_names'], community=fact['community'], source_dates=fact.get('source_dates'),
                        finding='Requires version context, not automatically a wrong fact.'))
    F.write(OUT / 'report.json', dict(districts=reports, official_source=dict(url=URL,
        publication_date=published, sha256=F.sha(SOURCE)), splits=splits, committee_evidence=committee_evidence,
        existing_membership_review=affected,
        observed_hazard='Old Tongsheng and new Tongsheng are different named administrative areas per official notice.',
        model_inputs_changed=False, target_gold_read=False, target_polygons_read=False,
        adopted_aliases=[], new_school_relations=0))
    F.write(OUT / 'manifest.json', dict(sources={str(p): F.sha(p) for p in paths},
        files={'report.json': F.sha(OUT / 'report.json')}))


if __name__ == '__main__':
    main()
