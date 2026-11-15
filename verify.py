"""Independent integrity checks of the delivered files, identities, and original data."""
from pathlib import Path
import hashlib
import json
import re
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

sources=read('来源与哈希.json')
for source in sources:
    assert sha(ROOT/source['package_path'])==source['sha256'],source['package_path']
for file in read('文件校验清单.json'):
    assert sha(ROOT/file['path'])==file['sha256'],file['path']
original_fdc=read(next(s['package_path'] for s in sources if s['original_workspace_path'].endswith('fdc-data.secondhand_buildings.深圳.json')))
path_total=0
for d,npaths,npolicies,nbuildings,ngroups in [('光明区',48,48,169,167),('坪山区',31,31,108,107)]:
    zones=read(f'{d}/polygons.json');schools=read(f'{d}/schools.json');policies=read(f'{d}/政策文本/年度原表.json')
    groups=read(f'{d}/小区数据/坐标组.json');buildings=read(f'{d}/小区数据/buildings.json')
    assert (len(zones),len(policies),len(buildings),len(groups))==(npaths,npolicies,nbuildings,ngroups)
    by_school={s['school_id']:s for s in schools};by_zone={z['zone_id']:z for z in zones}
    assert len(by_school)==len(schools) and len(by_zone)==len(zones)
    assert set(z['zone_id'] for s in schools for z in [by_zone[i] for i in s['zone_ids']])==set(by_zone)
    originals={s['ORG_ID']:s for s in read(zones[0]['source_archive'])['school_data']}
    archived=read(f'{d}/历史记录/未采用重复路径.json')
    assert {z['source_school_id'] for z in zones+archived}==set(originals)
    assert all(z['active_for_computation'] for z in zones)
    assert all(not z['active_for_computation'] for z in archived)
    for z in zones+archived:
        src=originals[z['source_school_id']]
        raw=src['MAPDATA']
        points=json.loads(re.sub(r'([{,])\s*(x|y)\s*:',r'\1"\2":',raw)) if isinstance(raw,str) else raw
        vertices=[[float(p['x']),float(p['y'])] for p in points]
        assert vertices==z['vertices'] and raw==z['source_mapdata']
        poly=Polygon(vertices)
        assert poly.convex_hull.area>0 and z['admitted']
        assert poly.is_valid==z['geometry_valid']
        assert z['map_scope_text']==src.get('XQ')
        school=by_school[z['school_id']]
        assert z['zone_id'] in school['zone_ids']+school['archived_zone_ids']
        policy=z['independent_texts'][0]
        assert policy['original_record'] in policies
        assert policy['original_scope_text']==policy['original_record']['scope_text']
        assert policy['scope_text'] in policy['original_scope_text']
        assert (ROOT/policy['document_local_path']).exists()
    points={}
    for b in buildings:
        original=original_fdc[b['source_row_index']]
        assert original==b['raw_record'] and original['area_name']==d
        assert original['location']['coordinates']==b['coordinates_bd09']==b['coordinates_bd09_original']
        assert b['building_id']==f"fdc:sz:{b['source_row_index']}"
        points.setdefault(tuple(b['coordinates_bd09']),[]).append(b['building_id'])
    assert len(points)==len(groups)
    assert {b['source_row_index'] for b in buildings}=={i for i,b in enumerate(original_fdc) if b.get('area_name')==d}
    for g in groups:
        assert points[tuple(g['representative_point_bd09'])]==g['coordinate_group_building_ids']
    if d=='光明区':
        missing=read(f'{d}/政策文本/官方文字有而地图未单列.json')
        assert len(schools)==48 and len(missing)==1
        school=by_school[missing[0]['school_id']]
        assert len(school['admission_ports'])==2 and not school['admission_ports'][0]['zone_ids']
        assert school['admission_ports'][0]['admission_port_id']==missing[0]['admission_port_id']
        two=next(z for z in zones if (z.get('admission_port_id') or '').endswith(':two'))
        assert two['school_id']==missing[0]['school_id']
        assert two['independent_texts'][0]['scope_text']==two['independent_texts'][0]['original_scope_text']
        assert two['school_polygon_accepted_by_user'] and not school['additional_port_polygon_required']
        assert not missing[0]['additional_polygon_required']
        assert missing[0]['scope_text'].startswith('深圳中学光明科学城学校一：')
    else:
        assert len(schools)==31
        assert all(len(s['zone_ids'])==1 for s in schools)
        a=next(z for z in zones if z['source_school_id']=='9c3d79a065d84861803bb201fbfb4742')
        assert len(archived)==1
        b=archived[0]
        assert b['source_school_id']=='3b78a6bd342d48898983203c5335d0b5' and a['school_id']==b['school_id']
        assert a['vertices']!=b['vertices']
        assert not read(f'{d}/官网实际边界覆盖层.json')['captured']
    path_total+=len(zones)
result=dict(passed=True,source_hashes=len(sources),map_paths=path_total,
    original_paths_and_text_exact=True,school_zone_referential_integrity=True,
    candidate_population_and_coordinates_exact=True,coordinate_group_partition_exact=True,
    school_one_branch_preserved=True,duplicate_path_archived_not_active=True)
(ROOT/'独立校验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
