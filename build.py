"""Build an offline source package. Original paths and source text are never repaired."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import math
import re
import shutil
import zipfile
from shapely.geometry import Polygon
from shapely.validation import explain_validity

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
AUDIT = Path('深圳市/光明坪山准入核查_20261115')
BASE = Path('深圳龙华_历史基线/深圳区')
IDENTITY = Path('深圳市/光明坪山身份复核_20261115')
SOURCES = {}
SELECTED_PINGSHAN = '9c3d79a065d84861803bb201fbfb4742'
ARCHIVED_PINGSHAN = '3b78a6bd342d48898983203c5335d0b5'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def archive(rel):
    rel = Path(rel)
    target = OUT / '来源存档' / rel
    source = ROOT / rel
    # Once archived, rebuild from the frozen package even outside the workspace.
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    SOURCES[rel.as_posix()] = dict(original_workspace_path=rel.as_posix(),
        package_path=target.relative_to(OUT).as_posix(), sha256=sha(target), bytes=target.stat().st_size)
    return target

def read(rel):
    return json.loads(archive(rel).read_text(encoding='utf-8-sig'))

def write(rel, value):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def text(rel, value):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value, encoding='utf-8')

def parse(raw):
    values = json.loads(re.sub(r'([{,])\s*(x|y)\s*:', r'\1"\2":', raw)) if isinstance(raw,str) else raw
    return [[float(p['x']),float(p['y'])] for p in values]

def local(rel):
    return archive(rel).relative_to(OUT).as_posix()

def point_id(point):
    return 'bd09:'+hashlib.sha256(json.dumps(point,separators=(',',':')).encode()).hexdigest()[:24]

CONFIG = [
    ('光明区','guangming',AUDIT/'current_sources/guangming_map_browser.json','https://zsxt.szgm.gov.cn:8443/visitgmgbxyxqdt','2026-11-15'),
    ('坪山区','pingshan',Path('科研推进/六区准入网络核验_20261023/深圳/pingshan_public_map_data.json'),'https://zsbm.szpsq.org.cn/visitpsgbxyxqdt','2026-10-23'),
]

fdc_rel = BASE/'fdc-data.secondhand_buildings.深圳.json'
fdc = read(fdc_rel)
coord_file = archive(Path('深圳市/四区数据集_20261103_v1/构建工具/coordinates.py'))
spec = importlib.util.spec_from_file_location('source_coordinates',coord_file)
coordinates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinates)
for name in ['核查结论.txt','browser_receipt.json','current_sources_receipt.json','初核汇总.json']:
    archive(AUDIT/name)
for name in ['guangming_policy_browser.json','pingshan_policy_browser.json','pingshan_map_status.html','pingshan_map_status.txt']:
    archive(AUDIT/'current_sources'/name)
archive(Path('科研推进/六区准入网络核验_20261023/深圳/pingshan_map_landing.html'))
for name in ['guangming_admission_ports.json','guangming_group_campuses.json',
             'pingshan_school_directory.json','pingshan_city_directory.json','坪山双路径比较.json','复核结论.txt']:
    archive(IDENTITY/name)

records, summaries, issues, schools_all, archived_records = [], [], [], [], []
for district, slug, map_rel, map_url, snapshot in CONFIG:
    raw = read(map_rel)
    policies = read(BASE/'政策文本'/f'{district}_2026_小学招生范围.json')
    audit = read(AUDIT/f'{district}_逐校初核.json')
    checked = {r['map_id']:r for r in audit['schools']}
    live_policy = AUDIT/'current_sources'/f'{slug}_policy_browser.json'
    policy_document = read(live_policy)
    compact = lambda s: re.sub(r'\s+','',s)
    assert all(compact(p['scope_text']) in compact(policy_document['text']) for p in policies)
    school_index, district_records, text_rows, missing = {}, [], [], []
    for source in raw['school_data']:
        check = checked[source['ORG_ID']]
        assert len(check['policy_matches']) == 1
        policy = policies[check['policy_matches'][0]['row_index']]
        vertices = parse(source['MAPDATA'])
        polygon = Polygon(vertices)
        # Noncollinear points define a face even when signed area cancels at a self-intersection.
        has_area = len(set(map(tuple,vertices))) >= 3 and polygon.convex_hull.area > 0
        assert has_area, source['MC']
        assert all(math.isfinite(v) for xy in vertices for v in xy)
        branch_two = check['match_status'] == 'explicit_named_subscope_two_only'
        school_id = f"sz:{slug}:school:{policy['record_id']}"
        zone_id = f"sz:{slug}:zone:{source['ORG_ID']}"
        selected_scope = policy['scope_text']
        school_name = policy['school_name_raw']
        notes = ['只要形成面即收录；自交不删除、不修复，原始路径及顶点顺序保留。']
        if branch_two:
            notes.append('按用户确认口径，本条现有polygon直接作为深圳中学光明科学城学校的polygon；一、二仅保留为官方报名条件，不拆学校，不要求另补端口一polygon。地图原名与原始路径保留。')
        if district == '坪山区':
            notes.append('地图采用2026-10-23存档；本轮入口维护，实际查询覆盖层未取得。')
        if check['match_status'] == 'official_search_address_corroborated_alias':
            notes.append('坪山第二小学别名依据同址与官方检索摘录；学校目录正文未读取。')
        if '深圳高级中学东校区' in school_name:
            notes.append('完整保留未来范围及过渡期条件；未将未来范围解释为现行范围。')
        independent = dict(scope_text=selected_scope, school_name_raw=policy['school_name_raw'],
            year=2026, source_url=policy['source_url'], source_position=f"{policy['source_section']} 第{policy['source_row']}行",
            identity_status=check['match_status'], applicability='2026正式文字；空间一致性未全面核验',
            policy_record_id=policy['record_id'], original_scope_text=policy['scope_text'], original_record=policy,
            document_local_path=local(live_policy), branch=None,
            map_source_admission_port='two' if branch_two else None)
        record = dict(district=district,school_id=school_id,zone_id=zone_id,source_school_id=source['ORG_ID'],
            school_name=school_name,map_school_name=source['MC'],vertices=vertices,source_mapdata=source['MAPDATA'],
            map_scope_text=source.get('XQ'),map_source_url=map_url,snapshot_time=snapshot,
            map_record_update_not_policy_year=source.get('GXSJ'),crs='BD-09_expected_BMap_contract',
            crs_note='沿用BMap原生BD-09坐标约定；原始导出无独立CRS声明，不作WGS84直接叠加。',
            source_archive=local(map_rel),geometry_valid=polygon.is_valid,geometry_issue=explain_validity(polygon),
            has_noncollinear_area=has_area,admitted=True,vertex_count=len(vertices),unique_vertex_count=len(set(map(tuple,vertices))),
            independent_texts=[independent],candidate_policy_texts=[],source_leads=[],alternate_query_boundary=None,
            query_override_capture_status='captured_empty' if district=='光明区' else 'not_captured',
            notes=notes,admission_port_id=school_id.replace(':school:',':port:')+':two' if branch_two else None,
            polygon_assignment_level='school',school_polygon_accepted_by_user=branch_two,
            polygon_vs_policy_status='未完成全面空间一致性核验',
            map_text_missing=not bool((source.get('XQ') or '').strip()),independent_text_missing=False)
        district_records.append(record)
        if school_id not in school_index:
            school_index[school_id] = dict(school_id=school_id, school_name=school_name,district=district,stage='小学',
                policy_record_ids=[policy['record_id']], zone_ids=[],source_school_ids=[],map_markers=[],
                geometry_status='has_area',identity_status=check['match_status'])
        school = school_index[school_id]
        school['zone_ids'].append(zone_id)
        school['source_school_ids'].append(source['ORG_ID'])
        marker = [float(source['POINTX']),float(source['POINTY'])] if source.get('POINTX') and source.get('POINTY') else None
        school['map_markers'].append(dict(source_school_id=source['ORG_ID'],coordinates_bd09=marker,
            coordinates_wgs84=coordinates.bd_to_wgs(marker) if marker else None,address=source.get('DZ'),
            point_role='官方地图标记点，不是学区polygon'))
        text_rows.append({k:record[k] for k in ['school_id','zone_id','district','school_name','map_school_name','map_scope_text','independent_texts','notes']})
    if district == '光明区':
        p = next(p for p in policies if p['school_name_raw']=='深圳中学光明科学城学校')
        sid = f"sz:{slug}:school:{p['record_id']}"
        portbase=sid.replace(':school:',':port:')
        school_index[sid]['admission_ports']=[
            dict(admission_port_id=portbase+':one',name='深圳中学光明科学城学校一',zone_ids=[],geometry_status='missing_not_inferred',
                 eligibility_text='监护人非凤凰英荟城、安居鸣鹿苑承租人不能申请。'),
            dict(admission_port_id=portbase+':two',name='深圳中学光明科学城学校二',zone_ids=school_index[sid]['zone_ids'],geometry_status='has_area',
                 eligibility_text='凤凰英荟城、安居鸣鹿苑承租人不能申请。')]
        school_index[sid]['identity_evidence']=local(IDENTITY/'guangming_admission_ports.json')
        school_index[sid]['identity_status']='same_school_two_admission_ports_not_two_schools'
        school_index[sid]['polygon_status']='existing_polygon_accepted_for_school_by_user'
        school_index[sid]['additional_port_polygon_required']=False
        missing.append(dict(school_id=sid,school_name=p['school_name_raw'],admission_port_id=portbase+':one',
            admission_port_name='深圳中学光明科学城学校一',policy_record_id=p['record_id'],
            scope_text=p['scope_text'].split('；',1)[0],original_scope_text=p['scope_text'],source_url=p['source_url'],
            reason='来源备注：官网未单列端口一地图。按用户确认，以该校现有polygon作为学校polygon，不再将此项作为缺面待办。',
            additional_polygon_required=False,status='source_note_not_blocker'))
    for school in school_index.values():
        if len(school['zone_ids']) > 1:
            issue=dict(district=district,school_id=school['school_id'],school_name=school['school_name'],
                zone_ids=school['zone_ids'],issue='same_school_multiple_original_paths',
                handling='用户授权二选一：采用2023年记录9c3d79a065d84861803bb201fbfb4742；2022年记录只存档，不参与计算。')
            school['identity_evidence']=local(IDENTITY/'pingshan_school_directory.json')
            school['identity_status']='same_school_same_site_supported_no_separate_campus_evidence'
            school['path_relation']='near_identical_overlapping_paths_probable_historical_duplicate'
            school['path_relation_certainty']='inference_not_official_database_version_explanation'
            issue['campus_review']='官网仅列和飞路11号；标记点约37米，边界交并比约99.21%，支持同校同址重复绘图，未证实两个校区。'
            write('坪山区/边界采用决定.json',dict(**issue,status='resolved_user_authorized_selection',
                selected_source_school_id=SELECTED_PINGSHAN,archived_source_school_id=ARCHIVED_PINGSHAN,
                reason='用户允许任选一条；选择录入较晚且地址电话完整者，不依据几何有效性排除。',
                source_metadata_date_not_policy_year=True))
            for record in district_records:
                if record['school_id']==school['school_id']:
                    record['notes'].append(issue['handling'])
                    record['notes'].append(issue['campus_review'])
    archived=[]
    for record in district_records:
        record['active_for_computation']=record['source_school_id']!=ARCHIVED_PINGSHAN
        if not record['active_for_computation']:
            record['archive_reason']='用户授权重复路径二选一；本条仅供溯源，不参与计算。'
            archived.append(record)
    district_records=[r for r in district_records if r['active_for_computation']]
    active_ids={r['zone_id'] for r in district_records}
    text_rows=[t for t in text_rows if t['zone_id'] in active_ids]
    for school in school_index.values():
        school['archived_zone_ids']=[z for z in school['zone_ids'] if z not in active_ids]
        school['zone_ids']=[z for z in school['zone_ids'] if z in active_ids]
        if school['archived_zone_ids']:
            school['archived_source_school_ids']=[ARCHIVED_PINGSHAN]
            school['source_school_ids']=[SELECTED_PINGSHAN]
            school['archived_map_markers']=[m for m in school['map_markers'] if m['source_school_id']==ARCHIVED_PINGSHAN]
            school['map_markers']=[m for m in school['map_markers'] if m['source_school_id']==SELECTED_PINGSHAN]
    write(f'{district}/历史记录/未采用重复路径.json',archived)
    archived_records.extend(archived)
    # Preserve source-port metadata without treating it as another school or a missing-polygon task.
    buildings,groups = [],{}
    for index, b in enumerate(fdc):
        if b.get('area_name') != district:
            continue
        xy=b.get('location',{}).get('coordinates')
        assert len(xy)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in xy)
        gid=point_id(xy);bid=f'fdc:sz:{index}';wgs=coordinates.bd_to_wgs(xy)
        row=dict(row_index=len(buildings),source_row_index=index,building_id=bid,building_name=b['building_name'],
            source_area_name=b['area_name'],source_block=b.get('bk_板块'),coordinates_bd09_original=xy,
            coordinates_bd09=xy,coordinates_wgs84=wgs,coordinate_group_id=gid,coordinate_status='available',
            coordinate_correction_source=None,source_path=local(fdc_rel),source_sha256=sha(archive(fdc_rel)),
            district_assignment_basis='FDC来源区标签，非行政边界核定',raw_record=b)
        buildings.append(row)
        if gid not in groups:
            groups[gid]=dict(coordinate_group_id=gid,representative_point_bd09=xy,representative_point_wgs84=wgs,
                building_id=bid,building_name=b['building_name'],coordinate_group_building_ids=[],coordinate_group_building_names=[])
        groups[gid]['coordinate_group_building_ids'].append(bid)
        groups[gid]['coordinate_group_building_names'].append(b['building_name'])
    assert len(buildings)==audit['summary']['fdc_records']
    assert len(groups)==audit['summary']['fdc_distinct_coordinate_groups']
    write(f'{district}/小区数据/buildings.json',buildings)
    write(f'{district}/小区数据/坐标组.json',list(groups.values()))
    write(f'{district}/schools.json',list(school_index.values()))
    write(f'{district}/texts.json',text_rows)
    write(f'{district}/政策文本/年度原表.json',policies)
    write(f'{district}/政策文本/官方文字有而地图未单列.json',missing)
    write(f'{district}/polygons.json',district_records)
    write(f'{district}/polygon与文字.json',district_records)
    write(f'{district}/学校路径对应.json',[dict(school_id=s['school_id'],school_name=s['school_name'],zone_ids=s['zone_ids']) for s in school_index.values()])
    write(f'{district}/原始地图分组.json',raw.get('groups'))
    write(f'{district}/官网实际边界覆盖层.json',dict(captured='actual_boundary_overrides' in raw,
        value=raw.get('actual_boundary_overrides'),interpretation='missing is unknown, not empty'))
    features=[]
    for record in district_records:
        ring=record['vertices'][:]
        if ring[-1]!=ring[0]:ring.append(ring[0])
        features.append(dict(type='Feature',id=record['zone_id'],properties={k:record[k] for k in
            ['school_id','zone_id','school_name','district','geometry_valid','geometry_issue','admitted','crs','source_school_id']},
            geometry=dict(type='Polygon',coordinates=[ring])))
    write(f'{district}/原始边界_BD09_非RFC7946.geojson',dict(type='FeatureCollection',
        coordinate_note='Native BMap BD-09 convention; NOT RFC7946 WGS84. No repair. Closing point appended only for GeoJSON.',features=features))
    write(f'{district}/地理数据/小区代表点_WGS84.geojson',dict(type='FeatureCollection',features=[dict(type='Feature',
        id=g['coordinate_group_id'],properties=dict(coordinate_group_id=g['coordinate_group_id'],building_names=g['coordinate_group_building_names']),
        geometry=dict(type='Point',coordinates=g['representative_point_wgs84'])) for g in groups.values()]))
    write(f'{district}/实验输入/输入清单.json',dict(buildings='../小区数据/buildings.json',schools='../schools.json',
        policy_source='../政策文本/年度原表.json',school_identity_paths_for_audit_only='../学校路径对应.json',
        polygons_are_evaluation_evidence_only=True,gold_generated=False,community_relations_ready=False,
        model_arrays_ready=False,source_package_ready=True))
    lines=[]
    for record in district_records:
        t=record['independent_texts'][0]
        lines += [record['school_name'],'地图ID：'+record['source_school_id'],
            f"顶点 {record['vertex_count']}；{record['geometry_issue']}；面已收录",'地图原文：',record['map_scope_text'] or '【未提供】',
            '2026独立官网文字（对应子范围）：',t['scope_text'],'年度原表完整文字：',t['original_scope_text'],
            '文字来源：'+t['source_url']+' '+t['source_position'],'地图来源：'+map_url,
            '备注：'+'；'.join(record['notes']),'\n'+'='*60+'\n']
    for item in missing:
        lines += ['【报名条件来源备注，不要求独立polygon】'+item['school_name'],item['scope_text'],item['source_url'],item['reason']]
    text(f'{district}/逐校文字.txt','\n'.join(lines)+'\n')
    summary=dict(district=district,schools=len(district_records),map_paths=len(district_records),
        source_map_paths=len(raw['school_data']),archived_duplicate_paths=len(archived),
        mapped_school_identities=sum(bool(s['zone_ids']) for s in school_index.values()),
        total_school_identities_including_policy_only=len(school_index),policy_entries=len(policies),policy_only_branches=len(missing),
        map_text_missing=sum(r['map_text_missing'] for r in district_records),with_independent_text=len(district_records),
        admitted_polygons=sum(r['admitted'] for r in district_records),self_intersections=sum(not r['geometry_valid'] for r in district_records),
        buildings=len(buildings),coordinate_groups=len(groups),map_snapshot=snapshot)
    summaries.append(summary)
    write(f'{district}/manifest.json',summary)
    text(f'{district}/使用说明.txt',f"{district}：{len(district_records)}条官方原始面；{len(buildings)}条小区，{len(groups)}个坐标组。\n"
         '学校身份见schools.json，边界记录见polygons.json，用school_id和zone_id关联；光明一/二另用admission_port_id表示报名端口。\n'
         'texts.json逐路径关联文字；年度原表完整保存在政策文本目录；原图分组不自动扩张学校范围。\n'
         '自交面完整收录；光明一/二不拆学校，现有polygon直接归学校，报名条件另存。\n'
         '本包已完成来源整理，没有生成Gold或模型输入数组。小区行政社区关系仍待补。\n')
    records.extend(district_records);schools_all.extend(school_index.values())

protocol=dict(schema='shenzhen_source_package_v1',version='20261115',districts=['光明区','坪山区'],
    polygon_admission='at_least_3_distinct_noncollinear_vertices',geometry_repair=False,
    invalid_polygon_filter=False,all_original_paths_retained=True,
    repeated_school_paths='user_authorized_single_selected_path_other_path_archived_only',
    selected_pingshan_source_school_id=SELECTED_PINGSHAN,archived_pingshan_source_school_id=ARCHIVED_PINGSHAN,
    official_policy_and_map_text_separate=True,policy_year=2026,policy_eligibility_expansion=False,
    gold_generated=False,training_run=False,coordinate_crs='BMap BD-09 convention; original coordinates unchanged',
    pingshan_query_override_status='not_captured',full_spatial_policy_consistency_verified=False)
protocol.update(identity_revision='20261115_admission_ports_and_duplicate_site_review',
    pingshan_gold_basis_user_confirmed=True,pingshan_gold_basis='2026-10-23_archived_original_MAPDATA',
    future_map_versions='archive_separately_do_not_silently_replace',
    guangming_one_two_relation='same_school_distinct_admission_ports_not_campuses',
    guangming_school_polygon_user_confirmed=True,
    guangming_school_polygon_source_id='dfc2a4c1204f4a0ca4f1b7c7b8dd936b',
    guangming_additional_port_one_polygon_required=False,
    pingshan_double_path_relation='same_school_same_site_supported_probable_historical_duplicate')
write('构建协议.json',protocol)
write('身份修订记录.json',dict(reason='官方问答确认光明一/二为同校报名端口',
    old_school_identity_count=80,new_school_identity_count=79,source_map_path_count=80,active_map_path_count=79,
    migrations=[dict(old_school_id=f'sz:guangming:school:sz_2026_primary_scope_0135:{port}',
        school_id='sz:guangming:school:sz_2026_primary_scope_0135',
        admission_port_id=f'sz:guangming:port:sz_2026_primary_scope_0135:{port}') for port in ['one','two']],
    pingshan_same_school_relation='同一school_id只采用2023年路径；2022年路径仅存档，不参与计算',
    review_report=local(IDENTITY/'复核结论.txt')))
write('两区_polygon与文字.json',records)
write('整理汇总.json',dict(districts=summaries,map_paths=len(records),source_map_paths=80,
    archived_duplicate_paths=len(archived_records),source_package_ready=True,gold_generated=False))
write('待核事项.json',issues+[
    dict(district='坪山区',issue='query_boundary_layer_not_captured',handling='收录全部原始绘图面；当前维护公告及历史版本记录随包保存。'),
    dict(district='两区',issue='community_input_relations_missing',handling='保留官网社区文字；不以评价polygon反推小区社区输入。')])

template=archive(Path('深圳市/polygon与文字整理_20261110/viewer_template.html')).read_text(encoding='utf-8')
template=template.replace('深圳四区','深圳光明、坪山').replace('四区_polygon与文字.json','两区_polygon与文字.json')
template=template.replace('360 条官方地图路径与逐校文字｜整理于 2026-11-10，地图快照为 11 月 3—4 日',
    '79 条采用边界与逐校文字｜整理于 2026-11-15；光明为本轮读取，坪山为 10 月 23 日快照')
template=re.sub(r'<a href="深圳光明、坪山_polygon与文字.zip" download>.*?</a>','',template)
template=template.replace('<a href="来源线索索引.json">历史 / 草案线索</a>','<a href="待核事项.json">待核事项</a>')
template=template.replace('全部四区','全部两区').replace('<option>福田区</option><option>南山区</option><option>龙华区</option><option>龙岗区</option>',
    '<option>光明区</option><option>坪山区</option>')
template=template.replace(' / 360 条',' / 79 条').replace('分区/${r.district}/','${r.district}/')
# Viewer selection must use zone_id: two Pingshan paths deliberately share a school_id.
template=template.replace('.school_id','.zone_id')
template=template.replace('地图文字 ${s.schools-s.map_text_missing} 条 · 独立文字已配对 ${s.with_independent_text} 校',
    '地图路径 ${s.map_paths} 条 · 对应学校 ${s.mapped_school_identities} 所 · 小区坐标组 ${s.coordinate_groups} 个')
template=re.sub(r'<footer>.*?</footer>',
    '<footer>自交面全部收录。深圳中学光明科学城学校直接采用现有polygon，一/二只保留为报名条件；坪山重复路径已选定2023年记录，另一条仅存档。未生成Gold；不以大学区名单扩张范围。GeoJSON为原生BD-09约定，非WGS84。</footer>',template)
payload=json.dumps(dict(records=records,summary=summaries),ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
text('index.html',template.replace('__DATA__',payload))

assert len(records)==79 and len({r['zone_id'] for r in records})==79
assert len(archived_records)==1 and archived_records[0]['source_school_id']==ARCHIVED_PINGSHAN
assert len({s['school_id'] for s in schools_all})==79  # 48 GM school/campus identities, 31 PS; ports are not schools
assert sum(not r['geometry_valid'] for r in records)==5
for district,slug,map_rel,url,snapshot in CONFIG:
    original={s['ORG_ID']:s for s in read(map_rel)['school_data']}
    subset=[r for r in records+archived_records if r['district']==district]
    assert set(original)=={r['source_school_id'] for r in subset}
    for r in subset:
        source=original[r['source_school_id']]
        assert r['vertices']==parse(source['MAPDATA']) and r['source_mapdata']==source['MAPDATA']
        assert r['map_scope_text']==source.get('XQ')
for source in SOURCES.values():
    assert sha(OUT/source['package_path'])==source['sha256']
write('来源与哈希.json',list(SOURCES.values()))
write('校验结果.json',dict(passed=True,map_paths=79,source_map_paths=80,archived_duplicate_paths=1,all_source_paths_and_map_text_exact_match=True,
    all_polygons_admitted=True,self_intersections_retained=5,policies=79,buildings=277,coordinate_groups=274,
    same_school_duplicate_paths_preserved=True,policy_only_branch_preserved=True,source_files=len(SOURCES),
    full_spatial_policy_consistency_verified=False,gold_generated=False))
text('使用说明.txt','''深圳光明、坪山数据整理包｜2026-11-15

打开index.html可离线检索、逐校查看原始边界与独立官网文字。
光明：48条地图面、48条年度文字、169条小区记录、167个坐标组。
坪山：采用31条地图面、31条年度文字、108条小区记录、107个坐标组；另1条重复路径仅存档。

只要polygon形成面就收录，不以自交作为排除标准；5条自交路径均原样保留。
光明schools.json收录48个学校/校区身份；学校一/二是同校两个报名端口。按用户确认直接采用现有polygon作为该学校polygon，不另补端口一polygon。
坪山31个学校身份各采用1条路径。东北师范大学深圳坪山实验学校采用2023年记录（122顶点），2022年记录（73顶点）仅存档。

各区目录：
  schools.json：学校身份及原始地图标记点，同校一/二报名条件另存；该学校现有polygon已采用。
  polygons.json、polygon与文字.json：每条原始边界、顶点、地图原文及独立政策原文。
  texts.json、逐校文字.txt：逐路径关联的文字；学校一另见政策文本/官方文字有而地图未单列.json。
  政策文本/年度原表.json：完整独立官网文字，合并单元格共享文字和条件均保留。
  小区数据/：原始记录、原始坐标、来源行号、稳定ID及坐标去重组。
  原始边界_BD09_非RFC7946.geojson：原生地图坐标，非WGS84；仅补闭合点，不修面。
  地理数据/小区代表点_WGS84.geojson：沿用既有BD09转换工具派生，原坐标不变。
  学校路径对应.json：school_id为学校身份、zone_id为路径身份，同校可有多个zone_id。
  原始地图分组.json：保留官方分组原记录，不据此传递扩校。
  历史记录/未采用重复路径.json：不参与计算的旧重复路径；原始来源数据亦完整保留。
  坪山区/边界采用决定.json：记录用户授权、选定与存档的ORG_ID。

来源存档/可直接查原始地图、政策全文、核查记录及维护公告；来源与哈希.json记录SHA256。
小区区名沿用FDC来源标签，未核定行政边界；坐标组不是最终可评分Gold数量。
光明地图本轮取得；坪山采用2026-10-23官方快照，当前维护，实际查询覆盖层尚未取得。
用户已确认坪山本版Gold以该快照原始绘图面为依据；今后新地图单独版本保存。本包尚未生成Gold。
坪山重复记录支持同校同址：官网列和飞路11号，标记相距约37米，边界交并比约99.21%，现有107坐标组均无命中差异。
重复绘图原因仍为推断；当前二选一依据用户授权，选择较晚且地址电话完整的记录，不声称官网已宣布替代关系。
深圳中学光明科学城学校关联完整官方一/二文字与现有polygon，一/二只表示报名条件；采用学校级地图口径，原始面不改绘。
端口一未单列地图仅作为来源备注，不是待办。别名证据层级和重复路径语义说明见待核事项及逐校备注。
资料整理与实验采用分开：本包未生成Gold、训练模型或改变现有实验数据；不以polygon构造预测输入。

重建：python -X utf8 build.py（Python及shapely）。从包内冻结来源重建，无需联网。
校验：python -X utf8 verify.py。页面验证：python -X utf8 verify_view.py（需本机浏览器环境）。
''')
# A package manifest covers all generated files and build tools, except itself and the ZIP.
files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.zip','.png'}
    and p.name not in {'文件校验清单.json','页面验证.json','独立校验结果.json'}]
write('文件校验清单.json',[dict(path=p.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)])
with zipfile.ZipFile(OUT/'光明坪山数据整理.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip':
            z.write(p,p.relative_to(OUT).as_posix())
print(json.dumps(summaries,ensure_ascii=False,indent=2))
