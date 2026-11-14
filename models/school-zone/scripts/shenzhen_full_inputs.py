"""Acquire and build Shenzhen inputs without reading catchment polygons or Gold."""
from pathlib import Path
import argparse, collections, concurrent.futures, datetime, hashlib, importlib.util
import json, math, os, pickle, re, sys, threading, time, winreg
import numpy as np
import requests
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from zoning import shenzhen_sources as SZ
OUT = REPO/'data/深圳市/完整输入_20261111'
TR = Transformer.from_crs(4326, 32650, always_xy=True)
COORD = SZ.DATA/'构建工具/coordinates.py'
spec = importlib.util.spec_from_file_location('sz_coordinates', COORD)
CO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CO)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def log(*v): print(*v, flush=True)
def project(points):
    a = np.asarray(points, float)
    return np.array(TR.transform(a[:, 0], a[:, 1])).T
def guard():
    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0]).replace('\\', '/')
            if '/深圳市/' in p and any(v in p for v in ['/Gold/', '/评价数据/', 'polygons.json', '点落面']):
                raise PermissionError('Input pipeline cannot read target evaluation labels: '+p)
    sys.addaudithook(hook)


def unit_names(text):
    # Keep subcommittee identities: parent community is not interchangeable with a subunit.
    text = text.replace('居委会居委会', '居委会')
    text = re.sub(r'以及|南至|的非', '，', text)
    return sorted(set(re.findall(r'[\u4e00-\u9fff]{2,6}社区(?:[\u4e00-\u9fff]{0,5}居委会)?', text)))


def plan():
    path = OUT/'acquisition_plan.json'
    if path.exists(): return read(path)
    queries = []
    for d in SZ.DISTRICTS:
        p = SZ._load(d, '政策规则/规则包.json')
        names = sorted({n for s in p['school_scope_inventory'] for n in unit_names(s['base_scope_text'])})
        for n in names:
            queries.append(dict(district=d, name=n, kind='policy_unit'))
        queries.append(dict(district=d, name='社区居民委员会', kind='district_inventory'))
    queries.append(dict(district='龙华区', name='外国语学校高新校区', kind='missing_school'))
    points = [s['representative_point_wgs84'] for d in SZ.DISTRICTS for s in SZ._load(d, '小区数据/坐标组.json')]
    a = np.array(points); low = a.min(0)-.02; high = a.max(0)+.02
    xs, ys = np.linspace(low[0], high[0], 4), np.linspace(low[1], high[1], 3)
    tiles = [dict(id=f'tile_{i}_{j}', bbox_wsen=[xs[i], ys[j], xs[i+1], ys[j+1]]) for i in range(3) for j in range(2)]
    p = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(), districts=list(SZ.DISTRICTS),
             queries=queries, road_tiles=tiles, coordinate_chain='Baidu BD09 -> iterative WGS84 -> EPSG32650; OSM WGS84 -> EPSG32650',
             community_method='Changsha comm.infer: nearest of 8 within1500m, prefer no crossing then no major/secondary/rail/river; strict no road and <=400m; loose minor-only and <=800m',
             matching='Same district; exact community or subcommittee identity; office roles preferred; alternatives retained',
             school_missing='Query independent Baidu campus POI, do not copy merged official marker',
             training='native default full feature blocks, width32,40steps,10seeds; no target Gold reference sampling',
             evaluation='same frozen full target population and strict set scoring; no tuning on this run',
             source_hashes={str(SZ.DATA/d/'政策规则/规则包.json'):sha(SZ.DATA/d/'政策规则/规则包.json') for d in SZ.DISTRICTS},
             gold_read=False)
    write(path, p)
    return p


def credential():
    key = os.environ.get('BAIDU_MAP_AK')
    if not key:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as reg:
            key = winreg.QueryValueEx(reg, 'BAIDU_MAP_AK')[0]
    return key


LOCK = threading.Lock()
LAST = [0.]
def query_baidu(key, district, term, page=0):
    identity = hashlib.sha256(f'{district}|{term}|{page}'.encode()).hexdigest()[:24]
    dest = OUT/'raw/baidu'/f'{identity}.json'
    if dest.exists() and read(dest).get('response', {}).get('status') == 0:
        return read(dest)
    params = dict(query=district+term, region='深圳市', city_limit='true', scope=2,
                  page_size=20, page_num=page, ret_coordtype='bd09ll', output='json')
    with LOCK:
        time.sleep(max(0, .24-(time.monotonic()-LAST[0])))
        LAST[0] = time.monotonic()
    rec = dict(district=district, term=term, page=page, endpoint='https://api.map.baidu.com/place/v2/search',
               parameters_without_key=params, fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    s = requests.Session(); s.trust_env = False
    try:
        rr = s.get(rec['endpoint'], params=dict(params, ak=key), timeout=(8, 25))
        rec.update(http_status=rr.status_code, response=json.loads(rr.text.replace(key, '[REDACTED]')))
    except Exception as e:
        rec.update(error_type=type(e).__name__, response={})
    finally: s.close()
    write(dest, rec)
    return rec


def leaf(name):
    name = name.replace('居委会居委会', '居委会')
    if '社区' in name and name.split('社区', 1)[1].removesuffix('居委会'):
        return name.split('社区', 1)[1].removesuffix('居委会')
    return name.removesuffix('居委会').removesuffix('社区')


def candidates(records, district, unit, school=False):
    choices = {}
    n = leaf(unit)
    for rec in records:
        for i, p in enumerate(rec.get('response', {}).get('results', [])):
            name = p.get('name', '')
            if p.get('area') != district or not p.get('location'): continue
            if school:
                if not ('高新' in name and '外国语' in name) or any(v in name for v in ['停车', '公交', '门', '外卖架']): continue
                rank = 0
            else:
                match = re.search(re.escape(n)+r'(社区)?(?:居民委员会|居委会|党群服务中心|社区委员会|社区工作站|社区服务中心)$', name)
                if match:
                    prefix = name[:match.start()]
                    # 大富 != 南大富; 新围 != 水斗新围. Only administrative prefixes allowed.
                    if prefix and not (prefix.endswith(('街道', '办事处')) or prefix in ['深圳市', district, n+'社区']):
                        continue
                if not match:
                    if name not in [n+'社区', n]: continue
                    rank = 4
                else:
                    rank = 0 if ('居民委员会' in name or '居委会' in name) else 1 if '党群' in name else 2
            loc = p['location']; bd = [loc['lng'], loc['lat']]
            if not (113.7 < bd[0] < 114.7 and 22.4 < bd[1] < 23): continue
            c = dict(uid=p.get('uid'), name=name, address=p.get('address'), district=p.get('area'),
                     coordinates_bd09=bd, coordinates_wgs84=CO.bd_to_wgs(bd), role_priority=rank,
                     source_query=rec['term'], api_rank=i)
            choices[p.get('uid') or name] = c
    return sorted(choices.values(), key=lambda c:(c['role_priority'], c['api_rank'], c['uid'] or ''))


def fetch_baidu():
    p = plan(); key = credential()
    def one(q):
        d, name = q['district'], q['name']
        recs = [query_baidu(key, d, name)]
        if q['kind'] == 'district_inventory':
            total = recs[0].get('response', {}).get('total', 0)
            recs += [query_baidu(key, d, name, i) for i in range(1, min(20, math.ceil(total/20)))]
            return dict(q, pages=len(recs), total=total, records=recs)
        cs = candidates(recs, d, name, q['kind']=='missing_school')
        if not cs and q['kind']=='policy_unit':
            recs.append(query_baidu(key, d, leaf(name)+'社区居民委员会'))
            cs = candidates(recs, d, name)
        if not cs and q['kind']=='policy_unit':
            recs.append(query_baidu(key, d, leaf(name)+'党群服务中心'))
            cs = candidates(recs, d, name)
        return dict(q, selected=cs[0] if cs else None, candidates=cs,
                    statuses=[r.get('response', {}).get('status') for r in recs])
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for v in pool.map(one, p['queries']):
            rows.append(v)
            log('BAIDU', v['district'], v['name'], v.get('selected', {}).get('name') if v.get('selected') else v.get('pages', 'unresolved'))
            write(OUT/'baidu_progress.json', rows)
    write(OUT/'baidu_inventory.json', rows)


def fetch_roads():
    p=plan()
    endpoints=['https://overpass.kumi.systems/api/interpreter', 'https://overpass-api.de/api/interpreter']
    tasks=[]
    for tile in p['road_tiles']:
        rec=OUT/'raw/osm'/f"{tile['id']}.receipt.json"
        if rec.exists() and not read(rec).get('complete'):
            w,s,e,n=tile['bbox_wsen'];xx=[w,(w+e)/2,e];yy=[s,(s+n)/2,n]
            for i in range(2):
                for j in range(2):
                    tasks.append(dict(id=f"{tile['id']}_recovery_{i}_{j}",parent_tile=tile['id'],bbox_wsen=[xx[i],yy[j],xx[i+1],yy[j+1]]))
        else:tasks.append(tile)
    write(OUT/'road_recovery_plan.json',dict(tiles=tasks,reason='Split timed-out geographic requests; no label-dependent selection',original_plan_sha256=sha(OUT/'acquisition_plan.json')))
    def one(tile):
        dest=OUT/'raw/osm'/f"{tile['id']}.json"
        receipt=dest.with_suffix('.receipt.json')
        if dest.exists() and receipt.exists() and read(receipt).get('complete'): return read(receipt)
        w,s,e,n=tile['bbox_wsen']
        query=f'[out:json][timeout:40];(way[highway]({s},{w},{n},{e});way[railway]({s},{w},{n},{e});way[waterway]({s},{w},{n},{e}););out geom;'
        attempts=[]
        for endpoint in endpoints:
            sess=requests.Session();sess.trust_env=False
            sess.headers['User-Agent']='SchoolCatchmentResearch/1.0 (public road geometry for Shenzhen; cached bounded requests)'
            try:
                rr=sess.get(endpoint,params={'data':query},timeout=(8,55))
                if rr.status_code != 200:
                    attempts.append(dict(endpoint=endpoint,http_status=rr.status_code,response_excerpt=rr.text[:250]))
                    continue
                body=rr.json()
                ok=rr.status_code==200 and 'elements' in body and not body.get('remark')
                attempts.append(dict(endpoint=endpoint,http_status=rr.status_code,remark=body.get('remark')))
                if ok:
                    write(dest,body)
                    rec=dict(tile,complete=True,ways=len(body['elements']),raw_sha256=sha(dest),query=query,attempts=attempts,
                             fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),osm_base=body.get('osm3s',{}))
                    write(receipt,rec);log('OSM',tile['id'],rec['ways']);return rec
            except Exception as exc: attempts.append(dict(endpoint=endpoint,error_type=type(exc).__name__))
            finally:sess.close()
        rec=dict(tile,complete=False,attempts=attempts);write(receipt,rec);log('OSM_FAILED',tile['id']);return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(one,tasks))
    write(OUT/'road_acquisition.json',rows)
    if not all(r['complete'] for r in rows):raise RuntimeError('Road coverage incomplete; full model cannot run yet')


def audit_coordinates():
    result={}
    for d in SZ.DISTRICTS:
        sites=SZ._load(d,'小区数据/坐标组.json');schools=SZ.load_schools(d)
        deltas=[]
        for a in sites:
            deltas.append(np.max(np.abs(np.array(CO.bd_to_wgs(a['representative_point_bd09']))-a['representative_point_wgs84'])))
        for a in schools:
            if a['school_point_bd09']:deltas.append(np.max(np.abs(np.array(CO.bd_to_wgs(a['school_point_bd09']))-a['school_point_wgs84'])))
        assert max(deltas)<1e-10
        result[d]=dict(site_count=len(sites),school_markers=sum(bool(s['school_point_bd09']) for s in schools),
                       maximum_reconversion_error_degrees=max(deltas),point_order='longitude,latitude',metric_crs=32650)
    write(OUT/'coordinate_audit.json',result);log('COORDINATES',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['baidu','roads','coordinates'])
    a=ap.parse_args();guard();plan()
    {'baidu':fetch_baidu,'roads':fetch_roads,'coordinates':audit_coordinates}[a.phase]()
