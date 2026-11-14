"""Download only explicitly listed public evidence; retain bytes, URL and hash."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import requests, json, hashlib, sys
from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parents[1]/'data/深圳市/福田区/文字补查_20261111'

def fetch(url):
    key=hashlib.sha256(url.encode()).hexdigest()[:16]
    dest=ROOT/'网页原件';dest.mkdir(exist_ok=True)
    meta={'url':url,'fetched_on':'2026-11-11'}
    try:
        r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=18)
        meta.update(status_code=r.status_code,final_url=r.url,content_type=r.headers.get('Content-Type',''))
        if r.status_code==200:
            suffix='.pdf' if r.content.startswith(b'%PDF') else '.html'
            target=dest/(key+suffix);target.write_bytes(r.content)
            meta.update(local_file=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(r.content).hexdigest())
            if suffix=='.html':
                r.encoding='utf-8';s=BeautifulSoup(r.text,'html.parser')
                meta['title']=s.title.get_text(' ',strip=True) if s.title else ''
                meta['admission_links']=[{'title':a.get_text(' ',strip=True),'url':urljoin(r.url,a['href'])} for a in s.find_all('a',href=True) if any(w in a.get_text() for w in ['招生','插班','学区','入学','转学','简介'])]
    except requests.RequestException as e:meta['error']=str(e)
    (dest/(key+'.metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(fetch,sys.argv[1:]):print(json.dumps(result,ensure_ascii=False))
