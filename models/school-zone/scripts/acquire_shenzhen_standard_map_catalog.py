"""Read the public standard-map landing page and its declared application bundles."""
from pathlib import Path
from urllib.parse import urljoin,urlparse
import re
import requests
import shenzhen_full_inputs as F

OUT=F.REPO/'outputs/shenzhen_resolution_20261112/standard_map_catalog_investigation'
URL='https://guangdong.tianditu.gov.cn/shenzhen/bzmap'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve acquisition')
    OUT.mkdir(parents=True);receipts=[];session=requests.Session()
    def fetch(url,name):
        try:
            response=session.get(url,timeout=(8,25));body=response.content;(OUT/name).write_bytes(body)
            receipt=dict(url=url,status=response.status_code,bytes=len(body),file=name,sha256=F.sha(OUT/name),content_type=response.headers.get('Content-Type'))
            receipts.append(receipt);print(receipt,flush=True)
            return response.text if response.status_code==200 else ''
        except requests.RequestException as e:
            receipts.append(dict(url=url,error_type=type(e).__name__));print('FAILED',url,type(e).__name__,flush=True);return ''
    html=fetch(URL,'landing.html')
    scripts=re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',html)
    for i,src in enumerate(scripts[:8]):
        url=urljoin(URL,src)
        if urlparse(url).netloc!=urlparse(URL).netloc:continue
        fetch(url,f'script_{i}.js')
    F.write(OUT/'receipts.json',receipts)
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.iterdir() if p.is_file()},
        scope='Public landing and its explicitly linked scripts only; no private datasets or account access.',target_gold_read=False))


if __name__=='__main__':main()
