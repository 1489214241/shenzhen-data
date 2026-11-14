"""Review exact residential identities before deriving any school relations."""
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import re
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm
from inventory_shenzhen_residential_table import extract
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_community_facts import MembershipScopeProof

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'municipal_residential_membership_sources'
OUT=SOURCE/'review_v1'
POLICY=F.REPO/'data/深圳市/边界合取单项修复输入_20261112'


def inspect():
    F.guard()
    old=F.read(ROOT/'community_membership_sources/review_v1/membership_facts.json')
    profiles=F.read(ROOT/'community_membership_sources/profile_inventory.json')['profiles']
    for d,file in [('龙华区','longhua_review_queue.json'),('龙岗区','longgang_review_queue.json')]:
        for r in F.read(SOURCE/file)['statements']:
            if not r['exact_coordinate_group_matches']:continue
            for hit in r['exact_coordinate_group_matches']:
                prior=[f for f in old['facts']+old['pending'] if f['district']==d and f['site_id']==hit['site_id']]
                print('REVIEW',d,r.get('row'),r['street'],r['community'],r['residential_name'],flush=True)
                print('PRIOR',[(p.get('street'),p.get('community'),p.get('status'),p.get('reason')) for p in prior],flush=True)
                if any('reason' in p for p in prior):
                    for p in prior:
                        if p.get('reason'):print('CONFLICT_EVIDENCE',[(e.get('street'),e.get('community')) for e in p['evidence']],flush=True)
                matching=[p for p in profiles if p['district']==d and p['community']==r['community']]
                print('IDENTITIES',sorted({(p['street'],p['community']) for p in matching}),flush=True)
                for p in profiles:
                    if p['district']!=d:continue
                    for paragraph in p['paragraphs']:
                        if norm(r['residential_name']) in norm(paragraph):
                            print('PROFILE',p['street'],p['community'],p['source']['page_dates'],paragraph,flush=True)


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve reviewed source facts')
    for manifest,base in [(SOURCE/'inventory_manifest.json',SOURCE),
                          (SOURCE/'longgang_inventory_manifest.json',None)]:
        for rel,h in F.read(manifest)['files'].items():
            assert F.sha(base/rel if base else rel)==h,rel
    # Re-read the table's merged cells and original HTML text, not search excerpts.
    table=extract()
    lg=F.read(SOURCE/'longgang_review_queue.json')['statements']
    for a,b in zip(table,lg):assert all(a[k]==b[k] for k in a)
    lh=F.read(SOURCE/'longhua_review_queue.json')['statements']
    html=BeautifulSoup((SOURCE/'longhua2022.html').read_text('utf8'),'html.parser').get_text('',strip=True)
    assert all(r['quote'] in html for r in lh)
    old=F.read(ROOT/'community_membership_sources/review_v1/membership_facts.json')
    profiles=F.read(ROOT/'community_membership_sources/profile_inventory.json')['profiles']
    facts=[];ledger=[];all_inputs=[]
    for d,records in [('龙华区',lh),('龙岗区',lg)]:
        inp=ROOT/'round03_partial_labels/inputs'/f'{d}.json';all_inputs.append(inp)
        sites=F.read(inp)['sites'];byid={s['coordinate_group_id']:s for s in sites}
        for r in records:
            hits=r['exact_coordinate_group_matches'];ident=(r['street'],r['community'])
            # Exact district-unique entity names only. Do not enlarge phases or aliases.
            if len(hits)!=1:
                ledger.append(dict(district=d,name=r['residential_name'],source=r,
                    decision='pending_entity_identity' if hits or r['partial_name_matches'] else 'no_exact_dataset_entity',
                    reason='Only one coordinate group with all names exactly equal can be admitted. No fuzzy alias or phase expansion.'))
                continue
            h=hits[0];sid=h['site_id'];names=byid[sid]['coordinate_group_building_names']
            assert all(norm(n)==norm(r['residential_name']) for n in names)
            assert len([s for s in sites if all(norm(n)==norm(r['residential_name']) for n in s['coordinate_group_building_names'])])==1
            identities={(p['street'],p['community']) for p in profiles if p['district']==d and p['community']==r['community']}
            prior=[f for f in old['facts']+old['pending'] if f['district']==d and f['site_id']==sid]
            conflicts=[p for p in prior if p.get('reason') or (p['street'],p['community'])!=ident]
            if conflicts or identities!={ident}:
                ledger.append(dict(district=d,name=r['residential_name'],site_id=sid,source=r,decision='pending_identity_conflict',prior=prior))
                continue
            assert re.fullmatch(r'\d{4}-\d{2}-\d{2}',r['publication_date'])
            assert r['publication_date']<='2026-06-01'
            existing=any(p['site_id']==sid and p['district']==d for p in old['facts'])
            evidence=dict(kind='official_residential_notice_geographical_membership',
                source=dict(url=r['source_url'],sha256=r['source_sha256'],page_dates=[r['publication_date']],
                    notice_url=r.get('notice_url',r['source_url']),sheet=r.get('sheet'),cells=r.get('cells')),
                quote=r['quote'],district=d,street=r['street'],community=r['community'],site_id=sid,site_names=names)
            f=dict(district=d,site_id=sid,site_names=names,street=r['street'],community=r['community'],
                evidence=[evidence],pre_enrollment_evidence=[evidence],source_dates=[r['publication_date']],
                complete_school_set=False,enrollment_eligibility=False,membership_absence_is_negative=False,
                source_age_limitation=True,status='source_supported_pre_enrollment',novel_vs_profile_review=not existing)
            ledger.append(dict(district=d,name=r['residential_name'],site_id=sid,source=r,
                decision='duplicate_existing_fact' if existing else 'accepted_dated_membership',
                reason='Official residential entry explicitly pairs street, community and exact residential entity; no contrary accepted/pending identity in profile review.',
                temporal_limitation='Dated pre-enrollment evidence; unchanged 2026 boundaries are not guaranteed. Later-only profiles not used to establish this fact.'))
            if not existing:facts.append(f)
    assert len({(f['district'],f['site_id']) for f in facts})==len(facts)
    F.write(OUT/'membership_facts.json',dict(facts=facts,cutoff='2026-06-01',target_gold_read=False,target_polygon_read=False,
        unknown_is_negative=False,adopted=False,source_age_limitation=True))
    F.write(OUT/'review_ledger.json',dict(rows=ledger,longhua_missing_community=F.read(SOURCE/'longhua_review_queue.json')['unresolved']))
    coverage={}
    for d in F.SZ.DISTRICTS:
        p=POLICY/d/'reviewed_policy.json';all_inputs.append(p)
        policy=F.read(p);inv=policy['school_scope_inventory']
        parser=BoundaryConjunctionParserV2(inv,parse_scope);asts=[parser.parse(r['base_scope_text']) for r in inv]
        local=[f for f in facts if f['district']==d]
        relations=MembershipScopeProof(inv,asts,profiles,d).derive(local)
        for relation in relations:relation['evidence_kind']='official_residential_notice_and_official_school_scope'
        F.write(OUT/'school_facts'/f'{d}.json',dict(facts=relations,all_school_ast_audit=[dict(school_id=r['school_id'],school_name=r['school_name'],
            source_text=r['base_scope_text'],ast=a,evidence=r['evidence']) for r,a in zip(inv,asts)],target_gold_read=False,target_polygon_read=False))
        coverage[d]=dict(new_membership_sites=len(local),school_relations=len(relations),
            applicable_sites=len({r['site_id'] for r in relations}),supported_schools=len({r['school_id'] for r in relations}))
        for r in relations:print('PROVED',d,r['building_names'],r['school_name'],r['school_scope_text'],flush=True)
    F.write(OUT/'summary.json',dict(coverage=coverage,decisions=dict(Counter(r['decision'] for r in ledger)),
        scope='Exact entity source membership supplement only; no new model training or complete labels.'))
    paths=[Path(__file__),F.REPO/'scripts/inventory_shenzhen_residential_table.py',
        F.REPO/'scripts/shenzhen_community_facts.py',F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py',
        SOURCE/'inventory_manifest.json',SOURCE/'longgang_inventory_manifest.json',
        ROOT/'community_membership_sources/review_v1/membership_facts.json',ROOT/'community_membership_sources/profile_inventory.json']+all_inputs
    links={str(p):F.sha(p) for p in paths}
    for manifest,base in [(SOURCE/'inventory_manifest.json',SOURCE),(SOURCE/'longgang_inventory_manifest.json',None)]:
        for rel,h in F.read(manifest)['files'].items():links[str(base/rel if base else Path(rel))]=h
    F.write(OUT/'manifest.json',dict(sources=links,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False,target_polygon_read=False))
    print('SOURCE_REVIEW',coverage,Counter(r['decision'] for r in ledger),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inspect','prepare']);globals()[p.parse_args().phase]()
