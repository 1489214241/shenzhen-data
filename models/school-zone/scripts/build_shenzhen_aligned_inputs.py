"""Full native feature adapter. Independent point/road observations, no target Gold."""
import ast
import collections
import importlib.util
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shenzhen_full_inputs as F
import shenzhen_aligned_policy as AP
from zoning import data, rules, evidence as E, shenzhen_policy as SP

REPO, ROOT, OUT, SZ = F.REPO, F.ROOT, F.OUT, F.SZ
BASE_INPUT = OUT
OUT = REPO/'data/深圳市/规则对齐输入_20261111'
OPPOSITE = dict(zip('东南西北','西北东南'))


def module(name, p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


POLICY_PATH=ROOT/'深圳龙华_历史基线/方法/road_solution/policy.py'
PP=module('scope_syntax',POLICY_PATH)
NAME_PATH=ROOT/'科研推进/边界与社区证据参照模型_20261103/nametok.py'
NT=module('name_tokens',NAME_PATH)
SIDE_PATH=ROOT/'科研推进/四至补全_20261028/执行器副本/rules/src/spatial.py'
# Execute the exact Changsha predicate without importing its dataset-owning runtime.
tree=ast.parse(SIDE_PATH.read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='strict_line')
scope={'np':np}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SIDE_PATH),'exec'),scope)
strict_line=scope['strict_line']


def road_class(tags):
    if tags.get('railway') in ['rail','light_rail','subway','tram','narrow_gauge']:
        return 'rail'
    if tags.get('waterway') in ['river','canal','stream']:return 'river'
    h=tags.get('highway','').removesuffix('_link')
    if h in ['motorway','trunk','primary']:return 'major'
    if h=='secondary':return 'secondary'
    if h=='tertiary':return 'tertiary'
    if h in ['residential','unclassified','service','living_street','road']:return 'local'
    return None


def roads():
    receipts=F.read(OUT/'road_acquisition.json')
    assert all(r['complete'] for r in receipts), 'Incomplete road acquisition'
    ways={}
    for rec in receipts:
        p=OUT/'raw/osm'/f"{rec['id']}.json"
        assert F.sha(p)==rec['raw_sha256']
        for e in F.read(p)['elements']:
            if e['type']=='way':
                assert len(e.get('geometry',[]))==len(e.get('nodes',[])), e['id']
                ways[e['id']]=e
    lines=[];features=[];byname=collections.defaultdict(list)
    for ident,e in sorted(ways.items()):
        tags=e.get('tags',{});cls=road_class(tags)
        coords=[[v['lon'],v['lat']] for v in e.get('geometry',[])]
        if len(coords)<2:continue
        if cls:
            lines.append(dict(wid=ident,name=tags.get('name:zh') or tags.get('name') or '',cls=cls,xy=F.project(coords)))
        if tags.get('highway') or tags.get('railway') or tags.get('waterway'):
            f=dict(type='Feature',properties=dict(tags,osm_id=ident),geometry=dict(type='LineString',coordinates=coords))
            features.append(f)
            for k in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
                for name in str(tags.get(k,'')).split(';'):
                    if name:byname[name].append(f)
    p=REPO/'data/lines/深圳_完整输入_20261111.pkl'
    with p.open('wb') as f:pickle.dump(lines,f,protocol=5)
    F.write(OUT/'road_geometry.json',dict(type='FeatureCollection',features=features))
    F.write(OUT/'road_geometry_audit.json',dict(ways=len(ways),model_lines=len(lines),classes=dict(collections.Counter(l['cls'] for l in lines)),
            metric_crs=32650,line_file=str(p),line_sha256=F.sha(p),tiles=receipts))
    return E.lines('深圳_完整输入_20261111'),byname


def office_core(name):
    name=re.sub(r'^(?:广东省|深圳市|深圳|龙华区|龙岗区)','',name)
    name=re.sub(r'^.*?街道(?:办事处)?','',name)
    name=re.sub(r'(?:社区)?(?:居民委员会|居委会|党群服务中心|社区委员会|社区工作站|社区服务中心)$','',name)
    return name.removesuffix('社区')


def community_points(d, inventory):
    policy_rows=[r for r in inventory if r['district']==d and r['kind']=='policy_unit']
    points={}
    for r in policy_rows:
        c=r.get('selected')
        if c:
            uid=c['uid'];points.setdefault(uid,dict(c,policy_names=[]))['policy_names'].append(r['name'])
    # Distractor offices prevent nearest-only assignment to an incomplete policy-name list.
    for row in inventory:
        if row['district']!=d or row['kind']!='district_inventory':continue
        for rec in row['records']:
            for c in rec.get('response',{}).get('results',[]):
                if c.get('area')!=d or not c.get('location') or not c.get('uid'):continue
                if not re.search(r'居民委员会|居委会|党群服务中心|社区工作站',c.get('name','')):continue
                loc=c['location'];bd=[loc['lng'],loc['lat']]
                points.setdefault(c['uid'],dict(uid=c['uid'],name=c['name'],address=c.get('address'),district=d,
                    coordinates_bd09=bd,coordinates_wgs84=F.CO.bd_to_wgs(bd),policy_names=[],role_priority=0))
    # A committee and its party service centre may be duplicate offices of the same unit.
    # Keep matched policy office; map same-name alternatives to it, never by school labels.
    used={};out=[]
    for c in sorted(points.values(),key=lambda c:(not bool(c['policy_names']),c.get('role_priority',4),c['uid'])):
        core=office_core(c['name'])
        if core in used:
            prev=used[core]
            span=float(np.linalg.norm(F.project([prev['coordinates_wgs84'],c['coordinates_wgs84']])[0]-F.project([prev['coordinates_wgs84'],c['coordinates_wgs84']])[1]))
            prev.setdefault('alternative_offices',[]).append(dict(c,distance_from_selected_m=span))
            prev['policy_names']=sorted(set(prev['policy_names']+c['policy_names']))
        else:used[core]=c;out.append(c)
    F.write(OUT/d/'community_points.json',dict(points=out,policy_queries=len(policy_rows),
            matched_policy_queries=sum(bool(r['selected']) for r in policy_rows),
            unresolved_policy_names=[r['name'] for r in policy_rows if not r['selected']],
            note='Office/area points are uncertain proxies, not administrative boundary polygons; inventory API is not a certified full district register.'))
    return out


def infer_communities(coords, points, ls):
    pts=F.project([p['coordinates_wgs84'] for p in points]);D=np.linalg.norm(coords[:,None]-pts[None],axis=-1)
    results=[];hard={'major','secondary','rail','river'}
    for q in range(len(coords)):
        cand=[j for j in np.argsort(D[q]) if D[q,j]<=1500][:8]
        if not cand:results.append(dict(point=None,distance_m=None,gate=3,strict=False,loose=False));continue
        segments=shapely.linestrings(np.stack([np.repeat(coords[q:q+1],len(cand),axis=0),pts[cand]],axis=1))
        qi,li=ls.tree.query(segments,predicate='crosses');cross=[set() for _ in cand]
        for i,j in zip(qi,li):cross[i].add(ls.cls[j])
        free=[i for i,c in enumerate(cross) if not c];soft=[i for i,c in enumerate(cross) if not c&hard]
        j=(free or soft or [0])[0];gate=0 if free else 1 if soft else 2;pick=cand[j];dist=float(D[q,pick])
        results.append(dict(point=int(pick),distance_m=dist,gate=gate,strict=gate==0 and dist<=400,
                            loose=gate<=1 and dist<=800,policy_names=points[pick]['policy_names'],
                            community_poi_uid=points[pick]['uid'],community_poi_name=points[pick]['name']))
    return results


def parse_scope(text):
    text=text.replace('居委会居委会','居委会').strip()
    # Numbered independent scopes are alternatives, never one conjunction.
    parts=[v.strip(' 。;；') for v in re.split(r'(?:^|(?<=[。；;\n]))\s*\d+[.、．]',text) if v.strip(' 。;；')]
    if len(parts)>1:return PP.combine('or',[parse_scope(v) for v in parts])
    text=re.sub(r'^\d+[.、．]\s*','',text)
    # Directional hyphen-separated chains have the same conjunction semantics as 、.
    if '以' in text:text=re.sub(r'(?<=以[东西南北])[—－-]+','、',text)
    return PP.parse_policy(text)


def make_district(d,ls,byname,inventory):
    sites=SZ._load(d,'小区数据/坐标组.json');schools=SZ.load_schools(d);p=SP.load_policy(d)
    sid=[s['school_id'] for s in schools];n,S=len(sites),len(schools)
    coords_wgs=np.array([s['representative_point_wgs84'] for s in sites]);coords=F.project(coords_wgs)
    school_wgs=[]
    for s in schools:
        pt=s['school_point_wgs84']
        if pt is None:
            match=next(r for r in inventory if r['kind']=='missing_school')['selected']
            assert match and '高新' in s['school_name'] and '高新' in match['name']
            pt=match['coordinates_wgs84'];s=dict(s,school_point_wgs84=pt,full_input_coordinate_source='Baidu independent campus POI',baidu=match)
        school_wgs.append(pt)
        schools[sid.index(s['school_id'])]=s
    points=community_points(d,inventory);inf=infer_communities(coords,points,ls)
    cname=np.array([r.get('community_poi_uid','') for r in inf]);strict=np.array([r['strict'] for r in inf])
    loose=np.array([r['loose'] for r in inf]);cs=np.zeros((n,S,4),np.float32)
    st=np.full((n,S,4),-1,np.int8);negative=np.zeros((n,S),bool);hit=np.zeros((n,S),bool)
    audits=[];edgecache={};edgeaudit={};directional=[]
    policy_by={v['school_id']:v for v in p['school_scope_inventory']}
    decisions=SP.predict_district(d)
    named={q:[sid.index(s) for s in r['policy_supported_school_ids']]
           for q,r in enumerate(decisions) if r['policy_supported_school_ids']}

    def side(name,direction,alias=None):
        key=(name,direction,alias)
        if key not in edgecache:
            target=alias or name
            ff={f['properties']['osm_id']:f for f in byname.get(target,[])}
            if ff:
                rec=dict(status='ready',predicate='line_side',inward=direction,geometries=[f['geometry'] for f in ff.values()])
                vals=strict_line(rec,coords_wgs)
            else:vals=np.full(n,-1,np.int8)
            edgecache[key]=vals;edgeaudit[str(key)]=dict(policy_name=name,inward=direction,explicit_alias=alias,
                source_osm_ids=sorted(ff),status='ready' if ff else 'missing_geometry',
                values_count={str(k):int((vals==k).sum()) for k in [-1,0,1]})
        return edgecache[key]

    def member(unit,school,partial=False):
        unit=unit.replace('居委会居委会','居委会').strip()
        possible={r['name'] for r in inventory if r['district']==d and r['kind']=='policy_unit'}
        if unit not in possible:return np.full(n,-1,np.int8)
        matched=np.array([unit in r.get('policy_names',[]) for r in inf])
        if partial:cs[matched&loose,school,2]=1
        else:
            cs[matched&strict,school,0]=1
            cs[matched&loose&~strict,school,1]=1
        # An inferred office association is not a certified boundary: unmatched is unknown.
        return np.where(matched&strict,1,-1).astype(np.int8)

    admin_features=F.read(REPO/'data/深圳市/规则对齐独立资料_20261111/admin_geometry.json')['features']
    aligned=AP.Evaluator(sites,schools,[policy_by[s] for s in sid],p,byname,parse_scope,side,member,F.project,strict_line,policy_named=named,admin_features=admin_features).build()
    st=aligned['statuses'];negative=aligned['negative'];known=aligned['known'];kind=aligned['kind']
    audits=aligned['audits'];pending=aligned['pending']
    for q in range(n):
        if loose[q] and (cs[q,:,0:2]>0).any():cs[q,~(cs[q,:,0:2]>0).any(1),3]=1
    F.write(OUT/d/'aligned_observations.json',aligned['observations'])
    F.write(OUT/d/'name_matching_audit.json',aligned['named_audit'])
    F.write(OUT/d/'ring_geometry_audit.json',aligned['geometry_audit'])
    F.write(OUT/d/'admin_geometry_audit.json',aligned['admin_audit'])
    np.savez_compressed(OUT/d/'observation_arrays.npz',**aligned['observation_arrays'])
    # Native policy-name features, same tokenizer and four channels as Changsha/Hangzhou.
    name_match=np.zeros((n,S,4),np.float32);tokens=[NT.tokens(policy_by[s]['base_scope_text']) for s in sid]
    for q,site in enumerate(sites):
        if q in known:continue
        keys=[NT.key(x) for x in site['coordinate_group_building_names']];keys=[k for k in keys if len(k)>=3]
        best=np.zeros(S)
        for j in range(S):
            lengths=[len(t) for k in keys for t in tokens[j] if t in k or len(k)>=4 and k in t]
            if lengths:best[j]=max(lengths)
        hh=np.flatnonzero(best>0)
        if len(hh)==1:name_match[q,hh[0],0]=1
        elif len(hh)>1:name_match[q,hh,1]=1
        name_match[q,:,2]=np.minimum(best/10,1)
        if len(hh):name_match[q,best==0,3]=1
    dist=data.District(name=d,city='深圳',crs=32650,site_names=[s['building_name'] for s in sites],
        school_names=[s['school_name'] for s in schools],orig=None,coords=coords,groups=np.array([s['coordinate_group_id'] for s in sites]),
        school_xy=F.project(school_wgs),answers={},eval_rows=[],train_answers={},scope=None,not_reference=set(),not_training=set(),
        community_name=cname,community_strict=strict,community_school=cs,name_match=name_match,
        lines_key='深圳_完整输入_20261111',site_ids=[s['coordinate_group_id'] for s in sites],school_ids=sid,policy_facts={})
    rr=rules.RuleOutput(statuses=st,negative=negative,known=known,kind=kind)
    meta=dict(sites=sites,schools=schools,known={str(q):ss for q,ss in known.items()},known_kind={str(q):v for q,v in kind.items()},
              missing_school_marker_ids=[],projected_crs=32650,policy_reference_sites=len(known),
              singleton_training_sites=sum(len(v)==1 for v in known.values()),
              caution='Road/community rule outputs are model input estimates, not verified enrollment truth; no query Gold used.')
    F.write(OUT/d/'community_assignments.json',[dict(site_id=s['coordinate_group_id'],**v) for s,v in zip(sites,inf)])
    F.write(OUT/d/'policy_observations.json',audits);F.write(OUT/d/'road_identity_audit.json',edgeaudit)
    F.write(OUT/d/'unresolved_overlaps.json',pending)
    F.write(OUT/d/'model_metadata.json',meta)
    with (OUT/d/'district.pkl').open('wb') as f:pickle.dump(dist,f,protocol=5)
    with (OUT/d/'rules.pkl').open('wb') as f:pickle.dump(rr,f,protocol=5)
    np.savez_compressed(OUT/d/'model_arrays.npz',coords=coords,school_xy=dist.school_xy,community_name=cname,
                        community_strict=strict,community_school=cs,name_match=name_match,statuses=st,negative=negative)
    summary=dict(sites=n,schools=S,school_coordinates=S,community_points=len(points),strict_community_sites=int(strict.sum()),
                 loose_community_sites=int(loose.sum()),community_feature_sites=int((cs>0).any((1,2)).sum()),
                 road_sides=len(edgecache),road_sides_resolved=sum(v['status']=='ready' for v in edgeaudit.values()),
                 known_sites=len(known),reference_assignments=sum(map(len,known.values())),known_by_kind=dict(collections.Counter(kind.values())),
                 training_singleton_sites=meta['singleton_training_sites'],directional_known_entries=int((st>=0).sum()),
                 name_feature_sites=int((name_match>0).any((1,2)).sum()),unresolved_overlap_sites=len(pending))
    F.write(OUT/d/'input_summary.json',summary);F.log(d,summary)
    return summary


def main():
    F.guard()
    if (OUT/'manifest.json').exists():raise FileExistsError('Keep frozen aligned inputs')
    # Reuse exactly the acquired independent coordinates, community POIs and OSM ways.
    old_manifest=F.read(BASE_INPUT/'manifest.json')
    old_files={k.replace('\\','/'):v for k,v in old_manifest['files'].items()}
    needed=['road_geometry.json','baidu_inventory.json']
    for d in SZ.DISTRICTS:needed += [f'{d}/district.pkl']
    for rel in needed:
        assert F.sha(BASE_INPUT/rel)==old_files[rel],rel
    features=F.read(BASE_INPUT/'road_geometry.json')['features'];byname=collections.defaultdict(list)
    for f in features:
        for k in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
            for name in str(f['properties'].get(k,'')).split(';'):
                if name:byname[name].append(f)
    ls=E.lines('深圳_完整输入_20261111');inventory=F.read(BASE_INPUT/'baidu_inventory.json')
    summary={d:make_district(d,ls,byname,inventory) for d in SZ.DISTRICTS}
    F.write(OUT/'input_summary.json',summary)
    sources=[Path(__file__),Path(AP.__file__),Path(F.__file__),REPO/'scripts/shenzhen_admin_from_pbf.py',REPO/'zoning/rules.py',REPO/'zoning/shenzhen_policy.py',POLICY_PATH,NAME_PATH,SIDE_PATH,F.COORD]+AP.SOURCE_FILES
    for p in sources:
        dest=OUT/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(REPO/'data/深圳市/规则对齐独立资料_20261111/admin_geometry.json',OUT/'admin_geometry.json')
    F.write(OUT/'manifest.json',dict(schema='shenzhen_changsha_rule_alignment_v1',target_gold_read=False,target_polygons_read=False,
        files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'},
        derived_from=dict(manifest=str(BASE_INPUT/'manifest.json'),sha256=F.sha(BASE_INPUT/'manifest.json'),reused_files={rel:F.sha(BASE_INPUT/rel) for rel in needed}),summary=summary))


if __name__=='__main__':main()
