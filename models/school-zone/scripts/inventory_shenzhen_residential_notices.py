"""Archive dated residential-community statements; no admission or school labels."""
import re
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from inventory_shenzhen_community_profiles import norm

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'municipal_residential_membership_sources'


def main():
    F.guard();dest=OUT/'longhua_review_queue.json'
    if dest.exists():raise FileExistsError('Preserve source inventory')
    receipt=next(r for r in F.read(OUT/'curl_host_receipts.json') if r['file']=='longhua2022.html')
    assert receipt['status']=='200' and F.sha(OUT/receipt['file'])==receipt['sha256']
    soup=BeautifulSoup((OUT/receipt['file']).read_text(encoding='utf8'),'html.parser')
    ps=[p.get_text('',strip=True) for p in soup.find_all('p')]
    starts=[i for i,p in enumerate(ps) if p=='“生活垃圾分类绿色小区”'];assert len(starts)==1
    end=next(i for i in range(starts[0]+1,len(ps)) if ps[i]=='“生活垃圾分类好家庭”')
    paragraphs=ps[starts[0]+1:end]
    sites=F.read(ROOT/'round03_partial_labels/inputs/龙华区.json')['sites']
    profiles=F.read(ROOT/'community_membership_sources/profile_inventory.json')['profiles']
    statements=[];unresolved=[]
    for paragraph in paragraphs:
        if '街道' not in paragraph and '福城' not in paragraph:continue
        heading,listing=re.split(r'[:：]',paragraph,maxsplit=1)
        for item in re.split(r'[、。]',listing):
            item=item.strip()
            if not item:continue
            m=re.fullmatch(r'(.+?街道)(.+?社区)(.+)',item)
            if not m:
                unresolved.append(dict(heading=heading,text=item,reason='No explicit community in this residential entry'));continue
            street,community,name=m.groups();identity=(street,community)
            related=[p for p in profiles if p['district']=='龙华区' and (p['street'],p['community'])==identity]
            exact=[];partial=[]
            for site in sites:
                names=site['coordinate_group_building_names']
                entry=dict(site_id=site['coordinate_group_id'],site_names=names)
                if all(norm(n)==norm(name) for n in names):exact.append(entry)
                elif any(norm(n) in norm(name) or norm(name) in norm(n) for n in names):partial.append(entry)
            # Profile corroboration is a review clue, not automated approval:
            # a building name may occur in an office address or a boundary clause.
            witnesses=[dict(source=p['source'],paragraph=q) for p in related for q in p['paragraphs'] if norm(name) in norm(q)]
            statements.append(dict(district='龙华区',street=street,community=community,residential_name=name,quote=item,
                source_url=receipt['url'],publication_date='2022-12-07',source_sha256=receipt['sha256'],
                exact_coordinate_group_matches=exact,partial_name_matches=partial,profile_mentions=witnesses,
                admitted=False,status='pending_current_identity_and_temporal_review',complete_school_set=False))
    assert len(statements)==26 and len(unresolved)==7
    summary=dict(explicit_community_statements=len(statements),no_community_entries=len(unresolved),
        exact_coordinate_groups=len({h['site_id'] for r in statements for h in r['exact_coordinate_group_matches']}),
        statements_with_profile_mentions=sum(bool(r['profile_mentions']) for r in statements),
        school_relations_generated=0,target_labels_read=False,target_polygons_read=False)
    F.write(dest,dict(summary=summary,statements=statements,unresolved=unresolved,
        caveat='2022 administrative membership is dated evidence, not automatic proof of 2026 scope. Header total36 conflicts with detailed33; retain all33 detailed entries and do not invent three rows.'))
    F.write(OUT/'inventory_manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.iterdir() if p.is_file()},
        source_code_sha256=F.sha(__file__),target_gold_read=False,admitted=False))
    print('NOTICE_INVENTORY',summary,flush=True)


if __name__=='__main__':main()
