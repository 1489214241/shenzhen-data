"""Cache public road identity evidence and bounded local OSM route relations."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import requests
from bs4 import BeautifulSoup
import shenzhen_full_inputs as F

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'independent_road_identity_sources'
URLS=[
    'https://jtys.sz.gov.cn/zwgk/ggqsydw/gsgl/',
    'https://jtys.sz.gov.cn/bsfw/wybs_122209/dllz/content/post_11017486.html',
    'https://ga.sz.gov.cn/gkmlpt/content/12/12703/post_12703089.html',
]


def fetch(url):
    key=hashlib.sha256(url.encode()).hexdigest()[:20];receipt=OUT/'official'/f'{key}.json'
    if receipt.exists():return F.read(receipt)
    s=requests.Session();s.trust_env=False
    s.headers['User-Agent']='SchoolCatchmentResearch/1.0 (public road identity audit)'
    rec=dict(url=url,fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        r=s.get(url,timeout=(8,35));rec['http_status']=r.status_code;rec['final_url']=r.url
        dest=receipt.with_suffix('.html');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content)
        rec['raw_file']=str(dest.relative_to(OUT));rec['sha256']=F.sha(dest)
        if r.status_code==200:
            r.encoding=r.apparent_encoding
            soup=BeautifulSoup(r.text,'html.parser')
            for tag in soup(['script','style']):tag.decompose()
            text=soup.get_text('\n',strip=True);dest.with_suffix('.txt').write_text(text,encoding='utf-8')
            rec['text_file']=str(dest.with_suffix('.txt').relative_to(OUT))
            terms=['机荷','沈海','水官','梅观','路线编码','路线编号']
            rec['matching_lines']=[line for line in text.splitlines() if any(t in line for t in terms)]
    except Exception as ex:rec['error_type']=type(ex).__name__
    finally:s.close()
    F.write(receipt,rec);return rec


def routes():
    path=OUT/'route_relations_receipt.json'
    if path.exists():return F.read(path)
    inv=ROOT/'community_boundary_geometry_audit/report.json'
    cases=F.read(inv)['cases'];ids=sorted({j for c in cases if not c['missing_names'] for v in c['source_osm_ids'].values() for j in v})
    query='[out:json][timeout:25];way(id:'+','.join(map(str,ids))+')->.w;(rel(bw.w)["type"="route"]["route"="road"];rel(bw.w)["type"="associatedStreet"];);out body;'
    attempts=[];rec=dict(seed_way_count=len(ids),seed_way_ids=ids,query=query,source_sha256=F.sha(inv),complete=False)
    for endpoint in ['https://overpass.kumi.systems/api/interpreter','https://overpass-api.de/api/interpreter']:
        s=requests.Session();s.trust_env=False
        try:
            r=s.get(endpoint,params={'data':query},timeout=(8,40));attempt=dict(endpoint=endpoint,http_status=r.status_code)
            if r.status_code==200:
                body=r.json();attempt['remark']=body.get('remark')
                if 'elements' in body and not body.get('remark'):
                    F.write(OUT/'route_relations.json',body)
                    rec.update(complete=True,relations=len(body['elements']),raw_sha256=F.sha(OUT/'route_relations.json'),osm_base=body.get('osm3s'))
            attempts.append(attempt)
            if rec['complete']:break
        except Exception as ex:attempts.append(dict(endpoint=endpoint,error_type=type(ex).__name__))
        finally:s.close()
    rec['attempts']=attempts;rec['fetched_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();F.write(path,rec);return rec


def main():
    F.guard()
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve acquisition')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(fetch,URLS):
            print('OFFICIAL',r['url'],r.get('http_status',r.get('error_type')),flush=True)
            for t in r.get('matching_lines',[]):print(t,flush=True)
    r=routes();print('ROUTE_RELATIONS',r.get('complete'),r.get('relations'),r['attempts'],flush=True)
    if r.get('complete'):
        for e in F.read(OUT/'route_relations.json')['elements']:print(e['id'],e.get('tags'),len(e.get('members',[])),flush=True)
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        source_script_sha256=F.sha(__file__),target_labels_read=False,target_polygons_read=False,new_school_facts=0))


if __name__=='__main__':main()
