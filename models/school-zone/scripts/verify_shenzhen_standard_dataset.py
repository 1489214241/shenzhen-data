"""核验现用深圳数据的来源、索引、全部标签及输入/评价分离。"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'data/深圳市'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    current=read(ROOT/'当前数据集.json');base=ROOT/current['dataset_path']
    assert current['active_districts']==['龙华区','龙岗区']
    count=0
    for a in read(base/'来源与哈希.json'):
        assert hashlib.sha256((base/a['path']).read_bytes()).hexdigest()==a['sha256'],a['path']
    city_records=read(base/'两区_polygon与文字.json')
    assert len(city_records)==202
    for d,expected in [('龙华区',71),('龙岗区',131)]:
        folder=base/d;schools=read(folder/'学校数据/schools.json');policies=read(folder/'政策文本/texts.json');zones=read(folder/'评价数据/官方学区范围.json')
        ids=[s['school_id'] for s in schools]
        assert len(ids)==len(set(ids))==expected
        assert ids==[p['school_id'] for p in policies]==[z['school_id'] for z in zones]
        assert all(p['independent_texts'] and any(t['year']==2026 for t in p['independent_texts']) for p in policies)
        for p in policies:
            for t in p['independent_texts']:assert (base/t['document_local_path']).is_file()
        originals={s['ORG_ID']:s for s in read(base/'原始资料'/d/'public_map_data.json')['school_data']}
        for z in zones:
            raw=originals[z['source_school_id']]['MAPDATA']
            raw=json.loads(re.sub(r'([{,]\s*)(x|y)\s*:',r'\1"\2":',raw)) if isinstance(raw,str) else raw
            assert z['vertices']==[[float(v['x']),float(v['y'])] for v in raw],z['school_name']
        buildings=read(folder/'小区数据/buildings.json');groups=read(folder/'小区数据/坐标组.json')
        source_buildings=read(base/'原始资料'/d/'候选小区原表.json')
        assert buildings==source_buildings
        assert groups==read(base/'原始资料'/d/'坐标组原表.json')
        allrows=read(folder/'评价数据/全部候选_点落面结果.json');gold=read(folder/'Gold/小区_学校Gold.json')
        assert [r['coordinate_group_id'] for r in allrows]==[g['coordinate_group_id'] for g in groups]
        matrix=np.load(folder/'评价数据/点落面矩阵.npz',allow_pickle=False)
        assert list(matrix['school_ids'])==ids
        assert list(matrix['coordinate_group_ids'])==[g['coordinate_group_id'] for g in groups]
        for r,values in zip(allrows,matrix['membership']):
            expected_ids=[ids[j] for j,v in enumerate(values) if v==1]
            assert r['point_school_ids']==expected_ids
            assert len(expected_ids)==r['point_school_count']
            assert r['point_school_names']==[schools[ids.index(s)]['school_name'] for s in expected_ids]
            assert r['gold_school_id']==(expected_ids[0] if len(expected_ids)==1 else None)
            assert r['invalid_geometry_hit_zone_ids']==[zones[j]['zone_id'] for j,v in enumerate(values) if v==1 and not zones[j]['geometry_valid']]
            count+=1
        assert gold==[r for r in allrows if r['point_school_ids'] and not r['unknown_school_ids']]
        gold_ids={r['coordinate_group_id'] for r in gold}
        assert read(folder/'小区数据/发布小区_有地图命中.json')==[b for b in buildings if b['coordinate_group_id'] in gold_ids]
        assert (folder/read(folder/'manifest.json')['protocol']).is_file()
        assert (folder/'评价数据'/read(folder/'评价数据/当前Gold.json')['path']).is_file()
    lh=[r for r in city_records if r['district']=='龙华区']
    assert not any('民乐' in r['school_name'] for r in lh)
    for campus in ['观湖校区','高新校区','华仁校区','华美校区']:assert sum(campus in r['school_name'] for r in lh)==1
    a=next(r for r in lh if '观湖校区' in r['school_name']);b=next(r for r in lh if '高新校区' in r['school_name'])
    assert a['vertices']==b['vertices'] and a['school_id']!=b['school_id']
    rows=read(base/'龙华区/评价数据/全部候选_点落面结果.json')
    assert all((a['school_id'] in r['point_school_ids'])==(b['school_id'] in r['point_school_ids']) for r in rows)
    gaoxin=next(s for s in read(base/'龙华区/学校数据/schools.json') if '高新校区' in s['school_name'])
    assert gaoxin['school_point_bd09'] is None and gaoxin['school_point_wgs84'] is None
    for page in [base/'index.html',base/'polygon_view.html']:
        for link in re.findall(r'href="([^"]+)"',page.read_text(encoding='utf-8')):
            if not link.startswith(('http:','https:','#')):assert (page.parent/unquote(link)).exists(),link
    result=dict(passed=True,school_count=202,verified_coordinate_groups=count,all_map_vertices_exact=True,
        all_selected_school_policy_pairs_present=True,source_hashes_valid=True,coordinate_groups_unchanged=True,
        all_gold_sets_exactly_match_matrix=True,duplicate_campus_polygons_preserve_both_school_ids=True,
        gaoxin_school_location_not_fabricated=True,html_file_links_exist=True)
    (base/'数据清单/独立校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
