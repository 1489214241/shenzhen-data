"""重建龙华、龙岗长沙式标准数据；保留原始证据，不覆盖旧实验或训练数据。"""
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

REPO=Path(__file__).resolve().parents[1]
WORK=REPO.parent
SOURCE=REPO/'data/深圳市'
OUT=SOURCE/'标准数据_20261111'
OLD=WORK/'深圳市/四区数据集_20261103_v1'
DISTRICTS=['龙华区','龙岗区']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def txt(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    receipts=[]
    def archive(src,rel):
        target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        assert sha(src)==sha(target)
        receipts.append(dict(path=rel,sha256=sha(target),source_workspace_path=str(Path(src).relative_to(WORK))))
        return rel
    current=read(SOURCE/'四区_polygon与文字.json')
    rows=copy.deepcopy([r for r in current if r['district'] in DISTRICTS])
    assert len(rows)==202
    identity=read(REPO/'data/规则核查_20261111/龙华学校身份核查/五条身份核查.json')
    confirmed={r['school_id']:r for r in identity['items'] if r['relation'].startswith('same_')}
    for r in rows:
        if r['school_id'] in confirmed:
            assert len(r['candidate_policy_texts'])==1
            t=r['candidate_policy_texts'].pop()
            t['identity_status']='已核实同校/同小学部：'+confirmed[r['school_id']]['conclusion']
            t['identity_evidence_urls']=confirmed[r['school_id']]['identity_sources']
            r['independent_texts'].append(t);r['independent_text_missing']=False
            r['notes']=[n for n in r['notes'] if not n.startswith('候选文字仅供核对')]
    assert all(any(t['year']==2026 for t in r['independent_texts']) for r in rows)
    assert not any('民乐实验小学' in r['school_name'] for r in rows)
    protocol=dict(schema='shenzhen_point_gold_release_v2',version='20261111',active_districts=DISTRICTS,
        historical_districts=['福田区','南山区'],reference='长沙市/芙蓉区/标准数据_20261018/manifest.json',
        target='official_map_point_membership',gold_definition='代表点命中所选官方原始MAPDATA路径的全部学校/校区集合；不是最终录取或全部申请资格',
        coordinate_system='BD-09原生点与路径；WGS84派生点仅供地理工具',point_predicate='captured_official_BMapLib.GeoUtils.isPointInPolygon_offline_replay',
        all_hits_preserved=True,geometry_repair=False,invalid_polygon_filter=False,extra_policy_application_expanded=False,
        campus_split='观湖和高新共享原合并路径，各有独立学校ID；华仁沿用合名路径，华美保留已有独立路径',
        minle_excluded=True,unknown_is_negative=False,full_policy_scope_spatial_review_complete=False,
        known_base_scope_exception='华东师范大学附属深圳龙华学校教育集团原polygon混入电脑派位范围；原图标签保留，不能当作核准基础学区Gold',
        predictive_inputs_exclude_polygons_and_gold=True,model_training_package_ready=False)
    write(OUT/'构建协议.json',protocol)
    for name in ['fdc_shenzhen_original.json','既有用户坐标更正.json']:
        archive(OLD/'原始资料/小区目录'/name,'原始资料/小区目录/'+name)
    archive(OLD/'构建工具/coordinates.py','构建工具/coordinates.py')
    spec=importlib.util.spec_from_file_location('standard_sz_coordinates',OUT/'构建工具/coordinates.py')
    coordinates=importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinates)
    archive(REPO/'data/规则核查_20261111/龙华学校身份核查/五条身份核查.json','原始资料/校区核实/五条身份核查.json')
    archive(SOURCE/'校区拆分_20261111/拆分说明.json','原始资料/校区核实/拆分说明.json')
    archive(SOURCE/'使用排除/民乐实验小学_20261111.json','原始资料/校区核实/民乐排除.json')
    archive(REPO/'data/规则核查_20261111/基础划片规则.json','原始资料/政策核查/基础划片规则.json')
    archive(REPO/'data/规则核查_20261111/龙华大学区专项核查/核查证据.json','原始资料/政策核查/龙华大学区核查.json')
    hashes={a['sha256']:SOURCE/a['package_path'] for a in read(SOURCE/'来源与哈希.json')}
    documents={}
    for r in rows:
        for t in r['independent_texts']:
            orig=t.get('original_record',{});h=orig.get('source_file_sha256') or orig.get('policy_pdf_sha256')
            assert h in hashes,(r['school_name'],h)
            if h not in documents:
                src=hashes[h];documents[h]=archive(src,'原始资料/年度政策/'+src.name)
            t['document_local_path']=documents[h]
    district_rows={}
    for d in DISTRICTS:
        rs=[r for r in rows if r['district']==d];district_rows[d]=rs;base=OUT/d
        rawpath=SOURCE/rs[0]['source_archive'];raw=read(rawpath)
        rawrel=archive(rawpath,'原始资料/'+d+'/public_map_data.json')
        for receipt_name in ('receipt.json','核验汇总.json','逐校边界核验.json'):
            if (rawpath.parent/receipt_name).exists():archive(rawpath.parent/receipt_name,'原始资料/'+d+'/'+receipt_name)
        byid={r['ORG_ID']:r for r in raw['school_data']}
        rawrel_old=rs[0]['source_archive']
        function=raw.get('native_point_in_polygon_function')
        if not function:function=read(OLD/d/'评价数据/官网原算法逐点结果.json')['implementation_source']
        txt(OUT/'原始资料'/d/'isPointInPolygon.js',function)
        buildings=read(OLD/d/'小区数据/buildings.json');groups=read(OLD/d/'小区数据/坐标组.json')
        archive(OLD/d/'小区数据/buildings.json','原始资料/'+d+'/候选小区原表.json')
        archive(OLD/d/'小区数据/坐标组.json','原始资料/'+d+'/坐标组原表.json')
        write(base/'小区数据/buildings.json',buildings);write(base/'小区数据/坐标组.json',groups)
        schools=[];policies=[];zones=[]
        for r in rs:
            original=byid[r['source_school_id']]
            marker=[float(original['POINTX']),float(original['POINTY'])] if original.get('POINTX') and original.get('POINTY') else None
            campus=r.get('campus_derivation',{}).get('campus')
            school=dict(school_id=r['school_id'],school_name=r['school_name'],source_school_id=r['source_school_id'],map_school_name=r['map_school_name'],district=d,stage='小学',school_address=original.get('DZ'),
                school_point_bd09=marker if campus!='高新' else None,source_map_marker_bd09=marker,
                point_role='官方合并条目标记点不等于高新校址' if campus=='高新' else '官方地图标记点，非独立测绘校址',source_url=r['map_source_url'],source_archive=rawrel,campus_derivation=r.get('campus_derivation'))
            if campus=='高新':school['school_address']='深圳市龙华区观湖街道观盛四路19号（官方学校目录；未单独地理编码）'
            school['school_point_wgs84']=coordinates.bd_to_wgs(school['school_point_bd09']) if school['school_point_bd09'] else None
            schools.append(school)
            r['source_archive']=rawrel
            policies.append({k:copy.deepcopy(r[k]) for k in ['school_id','school_name','district','map_school_name','map_scope_text','map_source_url','snapshot_time','independent_texts','candidate_policy_texts','notes']})
            r['zone_id']=r['school_id'].replace(':school:',':zone:')
            zones.append({k:copy.deepcopy(r[k]) for k in ['zone_id','school_id','school_name','source_school_id','map_school_name','vertices','source_mapdata','geometry_valid','geometry_issue','crs','crs_note','map_source_url','source_archive','snapshot_time','vertex_count']})
        write(base/'学校数据/schools.json',schools);write(base/'schools.json',schools)
        write(base/'政策文本/texts.json',policies);write(base/'texts.json',policies)
        write(base/'评价数据/官方学区范围.json',zones);write(base/'polygons.json',zones)
        write(base/'polygon与文字.json',rs)
        text=[]
        for r in rs:
            text += [r['school_name'],'地图原名：'+r['map_school_name']]
            for t in r['independent_texts']:text += [t['scope_text'],t['source_url']+' '+t['source_position']]
            text += ['','='*65,'']
        txt(base/'政策文本/逐校文字.txt','\n'.join(text));txt(base/'逐校文字.txt','\n'.join(text))
        linked={t['policy_record_id'] for r in rs for t in r['independent_texts']}
        if d=='龙华区':
            unlinked=[t for t in read(SOURCE/'龙华年度文字_未直接配对地图记录.json') if t['record_id'] not in linked]
            assert len(unlinked)==1 and '北辰' in unlinked[0]['school_name_raw']
            write(base/'政策文本/官方文字有而地图未单列.json',unlinked)
        features=[]
        for z in zones:
            coords=copy.deepcopy(z['vertices'])
            if coords[0]!=coords[-1]:coords.append(coords[0])
            features.append(dict(type='Feature',id=z['school_id'],properties={k:z[k] for k in ['school_id','school_name','district'] if k in z},geometry=dict(type='Polygon',coordinates=[coords])))
        geo=dict(type='FeatureCollection',coordinate_note='BD-09原始路径，非RFC7946；自交不修复',features=features)
        write(base/'评价数据/原始边界_BD09_非RFC7946.geojson',geo);write(base/'原始边界_BD09_非RFC7946.geojson',geo)
        write(base/'地理数据/小区代表点_WGS84.geojson',dict(type='FeatureCollection',features=[dict(type='Feature',properties={k:b[k] for k in ['building_id','building_name','coordinate_group_id']},geometry=dict(type='Point',coordinates=b['coordinates_wgs84'])) for b in buildings if b['coordinates_wgs84']]))
        write(base/'地理数据/学校标记点_WGS84.geojson',dict(type='FeatureCollection',features=[dict(type='Feature',properties={k:s[k] for k in ['school_id','school_name','point_role']},geometry=dict(type='Point',coordinates=s['school_point_wgs84']) if s['school_point_wgs84'] else None) for s in schools]))
        txt(base/'地理数据/README.txt','WGS84为既有BD-09点的派生坐标；不用于点落面。学校原始标记点在学校数据中。此包不伪造尚未构建的OSM道路/社区特征。\n')
        write(base/'实验输入/输入清单.json',dict(buildings='../小区数据/buildings.json',schools='../学校数据/schools.json',policy_texts='../政策文本/texts.json',evaluation_data_forbidden=['../Gold','../评价数据'],school_map_marker_not_polygon=True,rule_observations_ready=False,model_arrays_ready=False))
    subprocess.run(['node',str(REPO/'scripts/compute_shenzhen_membership.cjs'),str(OUT)],check=True)
    summaries=[];validation=[]
    for d in DISTRICTS:
        base=OUT/d;rs=district_rows[d];zones=read(base/'评价数据/官方学区范围.json');groups=read(base/'小区数据/坐标组.json');buildings=read(base/'小区数据/buildings.json')
        result=read(base/'评价数据/官网原算法逐点结果.json');matrix=np.array(result['membership'],dtype=np.int8)
        assert result['query_group_ids']==[g['coordinate_group_id'] for g in groups]
        assert result['school_ids']==[r['school_id'] for r in rs]
        differences=[];boundaries={};checked=0
        for j,z in enumerate(zones):
            poly=Polygon(z['vertices'])
            assert poly.is_valid==z['geometry_valid']
            if poly.is_valid:
                for i,g in enumerate(groups):
                    if not g['representative_point_bd09']:continue
                    p=Point(g['representative_point_bd09']);covered=poly.covers(p);checked+=1
                    if covered!=(matrix[i,j]==1):differences.append(dict(group_id=g['coordinate_group_id'],school_id=z['school_id'],native=int(matrix[i,j]),shapely=covered))
                    if poly.boundary.distance(p)==0:boundaries.setdefault(i,[]).append(z['zone_id'])
        assert not differences,(d,differences[:3])
        native_replay_equal=None
        if d=='龙华区':
            old=read(OLD/d/'评价数据/官网原算法逐点结果.json');assert old['query_group_ids']==result['query_group_ids']
            idx={s:i for i,s in enumerate(old['school_source_ids'])}
            oldm=np.array(old['membership'])[:,[idx[z['source_school_id']] for z in zones]]
            native_replay_equal=bool(np.array_equal(oldm,matrix));assert native_replay_equal
        all_results=[];gold=[]
        for i,g in enumerate(groups):
            hits=[j for j in range(len(zones)) if matrix[i,j]==1];unknown=[zones[j]['school_id'] for j in range(len(zones)) if matrix[i,j]<0]
            ids=[zones[j]['school_id'] for j in hits];names=[zones[j]['school_name'] for j in hits]
            r=dict(g,gold_record_id='point:'+g['coordinate_group_id'],source_map_partition=d,label_kind='official_map_point_membership',
                label_status='unknown' if unknown else ('outside_selected_map' if not hits else ('multiple_schools' if len(hits)>1 else 'available')),
                predicate=protocol['point_predicate'],point_school_ids=ids,point_school_names=names,point_school_count=len(ids),point_zone_ids=[zones[j]['zone_id'] for j in hits],
                gold_school_id=ids[0] if len(ids)==1 else None,gold_school_name=names[0] if len(names)==1 else None,on_boundary_zone_ids=boundaries.get(i,[]),
                invalid_geometry_hit_zone_ids=[zones[j]['zone_id'] for j in hits if not zones[j]['geometry_valid']],unknown_school_ids=unknown,
                complete_within_selected_roster=not unknown,complete_within_all_official_schools=False,extra_policy_application_expanded=False,individual_enrollment_eligibility=False,
                base_scope_policy_review='未完成全量基础范围空间核验；本标签仅为所选原始地图命中')
            all_results.append(r)
            if hits and not unknown:gold.append(r)
        write(base/'评价数据/全部候选_点落面结果.json',all_results);write(base/'Gold/小区_学校Gold.json',gold)
        write(base/'评价数据/面外及未知.json',[r for r in all_results if not r['point_school_ids'] or r['unknown_school_ids']])
        good={r['coordinate_group_id'] for r in gold}
        published=[b for b in buildings if b['coordinate_group_id'] in good]
        write(base/'小区数据/发布小区_有地图命中.json',published)
        txt(base/'Gold/小区_学校Gold.txt','\n'.join(r['building_name']+'\t'+'；'.join(r['point_school_names']) for r in gold))
        write(base/'评价数据/当前Gold.json',dict(available=True,schema='shenzhen_current_point_gold_v2',path='../Gold/小区_学校Gold.json',label_kind='official_map_point_membership',full_enrollment_truth=False,base_scope_fully_verified=False))
        np.savez_compressed(base/'评价数据/点落面矩阵.npz',membership=matrix,coordinate_group_ids=np.array(result['query_group_ids']),school_ids=np.array(result['school_ids']))
        counts=dict(district=d,candidate_raw_records=len(buildings),candidate_coordinate_groups=len(groups),published_building_records=len(published),gold_coordinate_groups=len(gold),
            school_count=len(rs),unique_original_map_records=len({r['source_school_id'] for r in rs}),official_text_matched_schools=sum(bool(r['independent_texts']) for r in rs),
            unique_school_groups=sum(len(r['point_school_ids'])==1 for r in gold),multiple_school_groups=sum(len(r['point_school_ids'])>1 for r in gold),
            outside_groups=sum(not r['point_school_ids'] and not r['unknown_school_ids'] for r in all_results),unknown_groups=sum(bool(r['unknown_school_ids']) for r in all_results),
            geometry_valid=sum(r['geometry_valid'] for r in rs),hit_count_distribution=dict(Counter(str(r['point_school_count']) for r in all_results)))
        summaries.append(counts)
        check=dict(passed=True,shapely_checked_valid_polygon_pairs=checked,shapely_disagreements=differences,longhua_original_browser_replay_equal=native_replay_equal,all_school_ids_unique=len(set(result['school_ids']))==len(rs),source_records_preserved=True)
        validation.append(dict(district=d,**check));write(base/'数据清单/校验结果.json',check)
        write(base/'manifest.json',dict(schema='shenzhen_point_gold_release_v2',counts=counts,protocol='../构建协议.json',label_kind='official_map_point_membership',training_dataset_ready=False))
        txt(base/'README.txt',f'{d}：长沙式标准目录\n学校/校区{len(rs)}条，官方文字均已配对；候选小区{len(buildings)}条、坐标组{len(groups)}个；有原图命中Gold{len(gold)}组。\n小区数据保留全部候选，发布子集另列。Gold是原始官方地图点命中，不是最终入学资格。共享范围多校全部保留，不扩大学区申请资格。\n政策文本/texts.json、学校数据/schools.json、评价数据/官方学区范围.json通过school_id连接。\n实验输入清单不包含Gold或学区面；尚未生成规则观察、模型训练矩阵和训练/测试划分。\n')
    write(OUT/'数据集清单.json',dict(version='20261111',active_districts=DISTRICTS,school_count=202,districts=summaries,training_dataset_ready=False))
    write(OUT/'manifest.json',dict(schema='shenzhen_standard_dataset_v2',version='20261111',active_districts=DISTRICTS,historical_districts=['福田区','南山区'],school_count=202,districts=summaries,viewer='index.html',training_dataset_ready=False))
    write(OUT/'数据清单/校验结果.json',dict(passed=True,districts=validation))
    write(OUT/'来源与哈希.json',receipts)
    write(OUT/'两区_polygon与文字.json',rows)
    for filename in ['build_shenzhen_standard_dataset.py','compute_shenzhen_membership.cjs']:
        archive(REPO/'scripts'/filename,'构建工具/'+filename)
    write(OUT/'来源与哈希.json',receipts)
    build_viewer(rows,summaries)
    counts_text='\n'.join(f'{s["district"]}：{s["school_count"]}学校/校区；{s["candidate_raw_records"]}条小区；{s["gold_coordinate_groups"]}个有命中坐标组' for s in summaries)
    txt(OUT/'README.txt','深圳现用标准数据集：龙华区、龙岗区（2026-11-11）\n\n'+counts_text+'\n\n每区目录：小区数据、学校数据、政策文本、地理数据、Gold、评价数据、实验输入、数据清单。原始资料、来源与哈希和构建协议在包根目录。\n\n当前学校目录202条，均有2026独立官方范围文字。龙华民乐已排除；观湖、高新、华仁、华美各一条。北辰校区文字保留在“官方文字有而地图未单列.json”，未虚构面。\nGold与长沙一样是小区代表点落入官方地图范围的集合；原生BD-09计算，多命中全部保留，自交不修复。龙华华东师大范围含派位，已明确标记，不将此Gold冒充全量核准的基础学区。\n\n本包为可读取的标准数据及地图评价标签，不是已经跑完规则、特征和训练的模型包。高新缺独立校址坐标，保留未知；不复制观湖坐标冒充高新校址。\n福田、南山四区旧资料保留在上一级目录，不属于本现用版本。旧Gold、长沙/杭州数据、模型训练配置均未覆盖。\n复现：在代码库根目录运行 python -X utf8 scripts/build_shenzhen_standard_dataset.py。\n')
    activate()
    # 数据重建后同步编译语义规则，防止重新导入把规则层丢掉。
    from build_shenzhen_policy_rules import main as build_policy_rules
    build_policy_rules()
    print(json.dumps(dict(school_count=202,districts=summaries,output=str(OUT)),ensure_ascii=False))


def build_viewer(rows,summaries):
    template=SOURCE/'index_四区来源历史.html'
    if not template.exists():shutil.copyfile(SOURCE/'index.html',template)
    page=template.read_text(encoding='utf-8')
    stats=[dict(district=s['district'],schools=s['school_count'],map_text_missing=sum(r['map_text_missing'] for r in rows if r['district']==s['district']),with_independent_text=s['official_text_matched_schools']) for s in summaries]
    encoded=json.dumps(dict(records=rows,summary=stats),ensure_ascii=False).replace('</','<\\/')
    page=re.sub(r'(<script id="payload" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],page,flags=re.S)
    page=re.sub(r'<header>.*?</header>','<header><h1>深圳现用数据 · 龙华、龙岗</h1><p>202条学校／校区，均已配对2026官方文字。原始地图与文字逐校查看。</p><div class="toplinks"><a href="index.html">数据集目录</a><a href="两区_polygon与文字.json">完整图文JSON</a><a href="README.txt">使用说明</a></div></header>',page,flags=re.S)
    page=page.replace('深圳四区','深圳两区').replace('全部四区','全部两区').replace('<option>福田区</option>','').replace('<option>南山区</option>','').replace(' / 360 条',' / 202 条')
    page=re.sub(r'<footer>.*?</footer>','<footer>Gold为代表点命中原始地图的学校集合；完整基础学区适用性尚未逐校核准。BD-09原生路径，不作WGS84使用。</footer>',page,flags=re.S)
    txt(OUT/'polygon_view.html',page)
    body=''
    for s in summaries:
        d=s['district'];body+=f'<tr><td>{d}</td><td>{s["school_count"]}</td><td>{s["candidate_raw_records"]}</td><td>{s["candidate_coordinate_groups"]}</td><td>{s["gold_coordinate_groups"]}</td><td>{s["unique_school_groups"]}</td><td>{s["multiple_school_groups"]}</td><td>{s["outside_groups"]}</td></tr>'
    links=''
    for d in DISTRICTS:
        links+=f'<h2>{d}</h2><ul>'+''.join(f'<li><a href="{d}/{path}">{label}</a></li>' for label,path in [('学校目录','学校数据/schools.json'),('官方政策文字','政策文本/texts.json'),('全部小区','小区数据/buildings.json'),('代表点分组','小区数据/坐标组.json'),('发布小区（有地图命中）','小区数据/发布小区_有地图命中.json'),('官方polygon','评价数据/官方学区范围.json'),('点落面Gold','Gold/小区_学校Gold.json'),('逐小区结果（含面外/未知）','评价数据/全部候选_点落面结果.json'),('校验结果','数据清单/校验结果.json')])+'</ul>'
    txt(OUT/'index.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深圳现用标准数据集</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#203b48}a{color:#006b78}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #cbd6dd}th{background:#edf5f6}.table{overflow:auto}</style><h1>深圳现用标准数据集</h1><p>龙华区、龙岗区 · 2026-11-11 · 按长沙标准数据目录整理</p><p><a href="polygon_view.html">查看学校polygon与官方文字</a> · <a href="README.txt">使用说明</a> · <a href="数据集清单.json">数据集清单</a> · <a href="构建协议.json">构建口径</a> · <a href="来源与哈希.json">原始来源与哈希</a></p><div class="table"><table><tr><th>区</th><th>学校/校区</th><th>小区记录</th><th>坐标组</th><th>有命中Gold组</th><th>单校组</th><th>多校组</th><th>面外组</th></tr>'+body+'</table></div><p>多校命中完整保留。Gold表示代表点命中原始官方地图，不等于最终入学资格；龙华华东师大原图包含派位范围，基础划片适用性仍需单列处理。高新未取得独立校址坐标，保留未知。</p>'+links+'<p><a href="../index_四区来源历史.html">福田、南山等四区历史来源资料</a></p></html>')


def activate():
    pointer=dict(version='20261111',active_districts=DISTRICTS,historical_districts=['福田区','南山区'],dataset_path='标准数据_20261111',manifest='标准数据_20261111/manifest.json',viewer='标准数据_20261111/index.html',school_count=202,model_training_package_ready=False)
    write(SOURCE/'当前数据集.json',pointer)
    source_manifest=read(SOURCE/'manifest.json')
    source_manifest.update(role='历史四区来源整理清单；当前数据见当前数据集.json',is_current_dataset=False,current_dataset='当前数据集.json',active_districts=DISTRICTS)
    write(SOURCE/'manifest.json',source_manifest)
    readme=SOURCE/'README.txt'
    header='【当前使用入口】龙华、龙岗两区；202条学校/校区。数据在标准数据_20261111/，打开其index.html。当前指针为当前数据集.json。\n本级四区文件与福田、南山目录保留为历史来源；下方为历史接入说明。\n\n'
    previous=readme.read_text(encoding='utf-8-sig')
    if not previous.startswith('【当前使用入口】'):txt(readme,header+previous)
    txt(SOURCE/'index.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=标准数据_20261111/index.html"><title>深圳现用数据</title><p>现用：龙华、龙岗。<a href="标准数据_20261111/index.html">打开标准数据集</a> · <a href="index_四区来源历史.html">四区历史资料</a></p></html>')
    write(WORK/'深圳市/当前标准数据集.json',dict(version='20261111',active_districts=DISTRICTS,path_relative_to_this_file='../学区归属模型_代码库/data/深圳市/标准数据_20261111',school_count=202))
    write(WORK/'深圳市/当前polygon与文字.json',dict(version='20261111',path_relative_to_this_file='../学区归属模型_代码库/data/深圳市/标准数据_20261111',viewer_relative_to_this_file='../学区归属模型_代码库/data/深圳市/标准数据_20261111/polygon_view.html',manifest_relative_to_this_file='../学区归属模型_代码库/data/深圳市/标准数据_20261111/manifest.json',school_count=202,districts=DISTRICTS,zip_required=False,stage='长沙式标准数据及原图点落面标签；非已训练模型包'))


if __name__=='__main__':main()
