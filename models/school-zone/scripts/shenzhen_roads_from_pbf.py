"""Recover full local OSM inputs from a public Guangdong province extract."""
from pathlib import Path
import datetime, hashlib, json, os, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import shenzhen_full_inputs as F
sys.path.insert(0,str(F.REPO/'tmp/shenzhen_osmium_vendor'))
import osmium
import requests
from shapely.geometry import LineString, box


def main():
    F.guard();os.chdir(F.REPO)
    raw=F.OUT/'raw/osm';src=raw/'guangdong.osm.pbf'
    # libosmium on Windows needs an ASCII relative filename; same bytes, no copy.
    link=Path('tmp/shenzhen_guangdong.osm.pbf')
    if not link.exists():os.link(src,link)
    session=requests.Session();session.trust_env=False
    md5_url='https://download.openstreetmap.fr/extracts/asia/china/guangdong.osm.pbf.md5'
    r=session.get(md5_url,timeout=(8,30));r.raise_for_status()
    expected=r.text.split()[0];actual=hashlib.md5(src.read_bytes()).hexdigest()
    assert actual==expected,(actual,expected)
    F.write(raw/'guangdong_checksum.json',dict(url=md5_url,response=r.text,verified_md5=actual,sha256=F.sha(src)))
    tiles=F.plan()['road_tiles'];w=min(t['bbox_wsen'][0] for t in tiles);s=min(t['bbox_wsen'][1] for t in tiles)
    e=max(t['bbox_wsen'][2] for t in tiles);n=max(t['bbox_wsen'][3] for t in tiles)
    region=box(w,s,e,n)
    with osmium.io.Reader(str(link)) as rd:
        h=rd.header();timestamp=h.get('osmosis_replication_timestamp');generator=h.get('generator')
    class Extract(osmium.SimpleHandler):
        def __init__(self):super().__init__();self.elements=[];self.invalid=0;self.scanned=0
        def way(self,way):
            tags=dict(way.tags)
            if not any(t in tags for t in ['highway','railway','waterway']):return
            self.scanned+=1
            valid=[p for p in way.nodes if p.location.valid()]
            if len(valid)!=len(way.nodes):self.invalid+=1
            xy=[(p.lon,p.lat) for p in valid]
            if len(xy)<2:return
            if max(x[0] for x in xy)<w or min(x[0] for x in xy)>e or max(x[1] for x in xy)<s or min(x[1] for x in xy)>n:return
            if not LineString(xy).intersects(region):return
            assert len(valid)==len(way.nodes),f'Incomplete selected way {way.id}'
            self.elements.append(dict(type='way',id=way.id,version=way.version,tags=tags,
                                      nodes=[p.ref for p in valid],geometry=[dict(lon=x,lat=y) for x,y in xy]))
    handler=Extract();handler.apply_file(str(link),locations=True,idx='flex_mem')
    assert len(handler.elements)>1000,'Unexpectedly empty regional extract'
    ident='guangdong_complete_region';dest=raw/f'{ident}.json'
    F.write(dest,dict(version=0.6,generator=generator,osm3s=dict(timestamp_osm_base=timestamp),elements=handler.elements))
    receipt=dict(id=ident,complete=True,bbox_wsen=[w,s,e,n],ways=len(handler.elements),raw_sha256=F.sha(dest),
        source='https://download.openstreetmap.fr/extracts/asia/china/guangdong.osm.pbf',
        source_sha256=F.sha(src),source_md5=actual,osm_base=dict(timestamp_osm_base=timestamp),
        fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        extraction='Complete Guangdong OSM extract; all highway/railway/waterway ways intersecting the predeclared bbox, full way geometries retained',
        region_scope='Target window lies inland in Guangdong, containing Longhua and Longgang; province extract is not a catchment geometry',
        scanned_ways=handler.scanned,global_incomplete_ways=handler.invalid,selected_incomplete_ways=0,
        limitations='OSM itself is not a certified exhaustive road inventory; geometry snapshot differs from other cities')
    F.write(raw/f'{ident}.receipt.json',receipt)
    prior=F.OUT/'road_acquisition.json'
    if prior.exists():F.write(F.OUT/'road_acquisition_failed_api_attempts.json',F.read(prior))
    F.write(prior,[receipt]);F.log('REGIONAL_OSM_COMPLETE',len(handler.elements),timestamp)


if __name__=='__main__':main()
