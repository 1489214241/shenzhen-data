"""Archive public community profiles only, with provenance and bounded fetches."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/shenzhen_resolution_20261112/community_membership_sources'
SEEDS=[
    'https://www.lg.gov.cn/bmzz/lcjdb/jdgk/sqgk/index.html',
    'https://www.lg.gov.cn/bmzz/bjjdb/jdgk/sqgk/index.html',
    'https://www.lg.gov.cn/bmzz/jhjdb/jhgk/sqjj/index.html',
    'https://www.szlhq.gov.cn/jdbxxgkml/lhjdb/jgzn_124619/sqgzz_124624/index.html',
    'https://www.szlhq.gov.cn/jdbxxgkml/mzjdb/jgzn_124588/sqgzz_124593/index.html',
]


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def fetch(url):
    parsed=urlparse(url)
    assert parsed.scheme=='https' and parsed.netloc in ['www.lg.gov.cn','www.szlhq.gov.cn']
    ident=hashlib.sha256(url.encode()).hexdigest()[:20]
    body=OUT/'raw'/f'{ident}.html';receipt=OUT/'receipts'/f'{ident}.json'
    if receipt.exists():
        rec=json.loads(receipt.read_text(encoding='utf8'))
        if rec.get('status')==200 and body.exists() and hashlib.sha256(body.read_bytes()).hexdigest()==rec.get('sha256'):return rec
    body.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['curl.exe','--silent','--show-error','--location','--max-time','25',
        '--proto','=https','--output',str(body),'--write-out','%{http_code}',url],capture_output=True)
    rec=dict(url=url,fetched_utc=datetime.now(timezone.utc).isoformat(),
        returncode=result.returncode,status=int(result.stdout.decode(errors='replace') or '0'),
        error=result.stderr.decode('gb18030',errors='replace')[:500],body=str(body.relative_to(OUT)),usable=False)
    if body.exists():
        raw=body.read_bytes();rec['sha256']=hashlib.sha256(raw).hexdigest();rec['bytes']=len(raw)
        soup=BeautifulSoup(raw.decode('utf8',errors='replace'),'html.parser')
        rec['title']=soup.title.get_text(' ',strip=True) if soup.title else ''
        rec['usable']=rec['status']==200 and '社区' in rec['title'] and not any(w in rec['title'] for w in ['验证','拦截','403'])
        rec['text']=soup.get_text('\n',strip=True)
        rec['links']=[dict(text=a.get_text(' ',strip=True),url=urljoin(url,a['href']))
                      for a in soup.find_all('a',href=True)
                      if '社区' in a.get_text() and urlparse(urljoin(url,a['href'])).netloc==parsed.netloc]
    write(receipt,rec)
    print(rec['status'],rec.get('title',rec['error']),url,flush=True)
    return rec


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:seeds=list(pool.map(fetch,SEEDS))
    article_urls=sorted({a['url'] for r in seeds if r['usable'] for a in r.get('links',[])
                         if '/content/post_' in a['url'] and any(k in a['text'] for k in ['社区','工作站'])})
    assert len(article_urls)<200,'Review unexpectedly broad discovery before fetching'
    with ThreadPoolExecutor(max_workers=3) as pool:articles=list(pool.map(fetch,article_urls))
    write(OUT/'catalog.json',dict(seed_urls=SEEDS,seeds=seeds,articles=articles,
        scope='Public community membership research; no school catchments, Gold or household records.',
        admission='Discovery only. Every clause requires source/identity review before use.'))
    print('COMPLETE',len(seeds),len(articles),'USABLE',sum(r['usable'] for r in seeds+articles),flush=True)


def links(rec):
    if rec['status']!=200:return []
    soup=BeautifulSoup((OUT/rec['body']).read_text(encoding='utf8',errors='replace'),'html.parser')
    return [(a.get_text(' ',strip=True),urljoin(rec['url'],a['href'])) for a in soup.find_all('a',href=True)]


def expand():
    previous=json.loads((OUT/'catalog.json').read_text(encoding='utf8'))
    portal=fetch('https://www.lg.gov.cn/bmzz/')
    homes=set()
    for rec in [portal]+previous['seeds']:
        for title,url in links(rec):
            if '街道' in title and ('/jdbxxgkml/' in url or '/bmzz/' in url):
                if urlparse(url).netloc in ['www.lg.gov.cn','www.szlhq.gov.cn']:
                    homes.add(url)
    assert len(homes)<50
    with ThreadPoolExecutor(max_workers=3) as pool:home_records=list(pool.map(fetch,sorted(homes)))
    directories=set()
    for rec in home_records:
        for title,url in links(rec):
            if any(k in title for k in ['社区概况','社区简介','社区工作站']):
                if urlparse(url).netloc in ['www.lg.gov.cn','www.szlhq.gov.cn']:
                    directories.add(url)
    assert len(directories)<60
    with ThreadPoolExecutor(max_workers=3) as pool:dir_records=list(pool.map(fetch,sorted(directories)))
    article_urls=set()
    for rec in home_records+dir_records:
        for title,url in links(rec):
            if '社区' in title and '/content/post_' in url and urlparse(url).netloc in ['www.lg.gov.cn','www.szlhq.gov.cn']:
                article_urls.add(url)
    assert len(article_urls)<300
    with ThreadPoolExecutor(max_workers=3) as pool:articles=list(pool.map(fetch,sorted(article_urls)))
    write(OUT/'catalog_expanded.json',dict(portal=portal,homes=home_records,directories=dir_records,articles=articles,
        enumeration='Discovered from official district/street navigation, not target errors or school labels.'))
    print('EXPANDED',len(homes),len(directories),len(articles),flush=True)


def longhua():
    script=fetch('https://www.szlhq.gov.cn/newjs/sub.js')
    js=(OUT/script['body']).read_text(encoding='utf8')
    pairs=[]
    for name,chunk in re.findall(r"name: '(\w+街道办)'(.*?)(?=f2Col)",js,re.S):
        pairs.append((name,re.search(r'jgzn:\s*\{ id: (\d+)',chunk).group(1)))
    assert len(pairs)==6
    dirs=[];meta=[]
    for name,ident in pairs:
        rec=fetch(f'https://www.szlhq.gov.cn/postmeta/i/{ident}.json')
        data=json.loads((OUT/rec['body']).read_text(encoding='utf8'))
        children=[v for v in data['children'] if v['name']=='社区工作站']
        assert len(children)==1
        child=children[0];meta.append(dict(street=name,metadata=rec,community_category=child))
        dirs.append('https://www.szlhq.gov.cn/'+child['pub_point']+'/index.html')
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,dirs))
    article_urls=sorted({url for rec in records for title,url in links(rec)
                         if '社区' in title and '/content/post_' in url and urlparse(url).netloc=='www.szlhq.gov.cn'})
    assert len(article_urls)<100
    with ThreadPoolExecutor(max_workers=3) as pool:articles=list(pool.map(fetch,article_urls))
    write(OUT/'catalog_longhua.json',dict(streets=meta,directories=records,articles=articles,
        discovery='Six street offices and community category IDs from official sub.js and postmeta public navigation.'))
    print('LONGHUA',len(records),len(articles),sum(r['usable'] for r in articles),flush=True)


def longgang():
    # All listed paths were located in official community pages or public search.
    paths=['lcjdb/jdgk/sqgk','bjjdb/jdgk/sqgk','jhjdb/jhgk/sqjj',
           'btjdb/btgk/sqjj','phjdb/jdgk/sqgk','hgjdb/jdgk/sqgk',
           'bljdb/jdgk/sqgk','pdjdb/jdgk/sqgk','lgjdb/jdgk/sqgk']
    dirs={'https://www.lg.gov.cn/bmzz/'+p+'/index.html' for p in paths}
    homes=[]
    for code in ['nwjdb','ysjdb']:
        rec=fetch('https://www.lg.gov.cn/bmzz/'+code+'/');homes.append(rec)
        for title,url in links(rec):
            if (title in ['社区概况','社区简介'] or re.search(r'/(?:sqgk|sqjj)/(?:index.html)?$',url)) and urlparse(url).netloc=='www.lg.gov.cn':
                dirs.add(url.rstrip('/')+'/index.html' if url.endswith('/') else url)
    assert len(dirs)==11,'Require all eleven public street directories'
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,sorted(dirs)))
    article_urls=sorted({url for rec in records for title,url in links(rec)
        if '社区' in title and '/content/post_' in url and urlparse(url).netloc=='www.lg.gov.cn'})
    assert len(article_urls)<200
    with ThreadPoolExecutor(max_workers=3) as pool:articles=list(pool.map(fetch,article_urls))
    write(OUT/'catalog_longgang.json',dict(homes=homes,directories=records,articles=articles,
        discovery='Official community overview directories of all eleven street offices; Longcheng contains inline profiles.'))
    print('LONGGANG',len(records),len(articles),sum(r['usable'] for r in articles),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expand',action='store_true');p.add_argument('--longhua',action='store_true');p.add_argument('--longgang',action='store_true');a=p.parse_args()
    longgang() if a.longgang else longhua() if a.longhua else expand() if a.expand else main()
