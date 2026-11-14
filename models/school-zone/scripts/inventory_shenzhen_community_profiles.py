"""Source-only community profile inventory and exhaustive literal-hit review queue.

Hits are proposals, not membership facts. Context can mention neighboring
communities, exclusions, phases, institutions, or historical subdivisions.
"""
from pathlib import Path
import re
import unicodedata
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F
from shenzhen_qualified_scope import QualifiedParser
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_aligned_policy import walk

OUT=F.REPO/'outputs/shenzhen_resolution_20261112/community_membership_sources'
INPUT=F.REPO/'data/深圳市/局部条件修复输入_20261112'
STREETS=dict(lcjdb='龙城街道',bjjdb='布吉街道',jhjdb='吉华街道',btjdb='坂田街道',phjdb='平湖街道',
             hgjdb='横岗街道',bljdb='宝龙街道',pdjdb='坪地街道',lgjdb='龙岗街道',nwjdb='南湾街道',ysjdb='园山街道',
             lhjdb='龙华街道',mzjdb='民治街道',ghjdb='观湖街道',gljdb='观澜街道',fcjdb='福城街道',dljdb='大浪街道')


def norm(t):
    return re.sub(r'[\s\u3000·•・]','',unicodedata.normalize('NFKC',t))


def source(rec):
    path=OUT/rec['body'];assert F.sha(path)==rec['sha256']
    h=BeautifulSoup(path.read_text(encoding='utf8'),'html.parser')
    dates=[m.get('content','') for m in h.find_all('meta') if (m.get('name') or '').lower() in ['pubdate','publishtime','date','articledate']]
    if not dates:
        match=re.search(r'发布时间[：:]\s*(\d{4}[-年]\d{1,2}[-月]\d{1,2})',h.get_text(' ',strip=True))
        if match:dates=[match[1]]
    return h,dict(url=rec['url'],raw_file=rec['body'],sha256=rec['sha256'],page_dates=dates,
                  fetched_utc=rec['fetched_utc'],title=rec['title'])


def main():
    F.guard()
    if (OUT/'profile_inventory.json').exists():raise FileExistsError('Preserve inventory snapshot')
    profiles=[];coverage={};excluded=[]
    for d,catalog in [('龙华区','catalog_longhua.json'),('龙岗区','catalog_longgang.json')]:
        cat=F.read(OUT/catalog)
        assert len(cat['directories'])==(6 if d=='龙华区' else 11)
        for rec in cat['articles']:
            h,src=source(rec);m=re.fullmatch(r'(.+?社区)(?:概况|简介|工作站)?(?:-龙华政府在线)?',rec['title'])
            if not m or '街道' in m[1]:excluded.append(src);continue
            body=h.select_one('#zoomcon') or h.select_one('.article_box')
            assert body is not None,rec['url']
            paragraphs=[p.get_text(' ',strip=True) for p in body.find_all(['p','tr']) if p.get_text(strip=True)]
            if not paragraphs:paragraphs=[body.get_text(' ',strip=True)]
            code=re.search(r'/(?:bmzz|jdbxxgkml)/([^/]+)/',rec['url'])[1]
            profiles.append(dict(district=d,street=STREETS[code],community=m[1],paragraphs=list(dict.fromkeys(paragraphs)),source=src))
        if d=='龙岗区':
            for code in ['lcjdb','ysjdb']:
                rec=next(r for r in cat['directories'] if '/'+code+'/' in r['url']);h,src=source(rec)
                count=0
                for p in h.find_all('p'):
                    text=p.get_text(' ',strip=True);m=re.match(r'^(?:[（(][一二三四五六七八九十]+[）)])?([^，。；、()（）]{1,8}社区)',text)
                    if not m:continue
                    profiles.append(dict(district=d,street=STREETS[code],community=m[1],paragraphs=[text],source=src));count+=1
                assert count==(11 if code=='lcjdb' else 6),(code,count)
        policy=F.read(INPUT/d/'reviewed_policy.json');inv=policy['school_scope_inventory'];parser=QualifiedParser(inv,parse_scope)
        units={a['text'] for r in inv for a in walk(parser.parse(r['base_scope_text'])) if a['op']=='membership'}
        names={p['community'] for p in profiles if p['district']==d}
        coverage[d]=dict(profiles=sum(p['district']==d for p in profiles),unique_named_communities=len(names),
            policy_membership_units=len(units),exact_name_units_present=sorted(units&names),missing_exact_name_units=sorted(units-names),
            caution='Name coverage only. District/street identity, dates and local restrictions still need review.')
    F.write(OUT/'profile_inventory.json',dict(profiles=profiles,coverage=coverage,excluded_broad_pages=excluded,
        input_snapshot_changed=False,target_gold_read=False))
    proposals=[]
    for d in F.SZ.DISTRICTS:
        sites=F.read(F.REPO/'outputs/shenzhen_resolution_20261112/round03_partial_labels/inputs'/f'{d}.json')['sites']
        for profile in [p for p in profiles if p['district']==d]:
            for i,paragraph in enumerate(profile['paragraphs']):
                text=norm(paragraph)
                hits=[]
                for s in sites:
                    matched=[name for name in s['coordinate_group_building_names'] if len(norm(name))>=3 and norm(name) in text]
                    if matched:hits.append(dict(site_id=s['coordinate_group_id'],site_names=s['coordinate_group_building_names'],literal_matched_names=matched))
                if hits:proposals.append(dict(district=d,street=profile['street'],community=profile['community'],source=profile['source'],
                    paragraph_index=i,quote=paragraph,hits=hits,admitted=False))
    F.write(OUT/'literal_hit_review_queue.json',dict(proposals=proposals,
        warning='Literal occurrence is not membership. Must reject institutional, neighboring, exclusion and phase ambiguity.',target_gold_read=False))
    F.write(OUT/'inventory_manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.name!='inventory_manifest.json'},code_sha256=F.sha(__file__),target_gold_read=False))
    print('COVERAGE',coverage,flush=True)
    print('REVIEW_PARAGRAPHS',len(proposals),flush=True)
    for d in F.SZ.DISTRICTS:
        print(d,len({h['site_id'] for r in proposals if r['district']==d for h in r['hits']}),flush=True)


if __name__=='__main__':main()
