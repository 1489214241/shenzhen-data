"""将已核验的深圳polygon与文字整理结果放入本代码库data/深圳市。"""
from pathlib import Path
import hashlib
import json
import shutil

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO.parent / '深圳市/polygon与文字整理_20261110'
DEST = REPO / 'data/深圳市'
DISTRICTS = ['福田区', '南山区', '龙华区', '龙岗区']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    rows = json.loads((SOURCE/'四区_polygon与文字.json').read_text(encoding='utf-8'))
    assert len(rows) == 360 and len({r['school_id'] for r in rows}) == 360
    copies = []
    # 真正复制文件，运行时不依赖外层研究文件夹，也不读取ZIP。
    for p in SOURCE.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(SOURCE)
        if rel.parts[0] not in {'分区', '来源存档'} and p.name not in {
            '四区_polygon与文字.json', '来源与哈希.json', '来源线索索引.json',
            '龙华年度文字_未直接配对地图记录.json', '待核事项.json', '整理汇总.json', '校验结果.json', '页面验证.json'}:
            continue
        target = DEST / (Path(*rel.parts[1:]) if rel.parts[0] == '分区' else rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        assert sha(p) == sha(target)
        copies.append({'source':str(p), 'path':target.relative_to(DEST).as_posix(), 'sha256':sha(target)})
    page = (SOURCE/'index.html').read_text(encoding='utf-8').replace('分区/${r.district}/', '${r.district}/')
    page = page.replace('使用说明.txt', 'README.txt')
    (DEST/'index.html').write_text(page, encoding='utf-8')
    if (DEST/'index_四区来源历史.html').exists():
        (DEST/'index_四区来源历史.html').write_text(page, encoding='utf-8')
    districts = []
    for d in DISTRICTS:
        subset = [r for r in rows if r['district'] == d]
        schools = [{k:r[k] for k in ['school_id','source_school_id','school_name','map_school_name','district']} for r in subset]
        polygons = [{k:r[k] for k in ['school_id','source_school_id','school_name','district','vertices','source_mapdata',
            'vertex_count','unique_vertex_count','crs','crs_note','geometry_valid','geometry_issue',
            'map_source_url','snapshot_time','source_archive','alternate_query_boundary']} for r in subset]
        texts = [{k:r[k] for k in ['school_id','source_school_id','school_name','district','map_scope_text',
            'map_source_url','snapshot_time','map_record_update_not_policy_year','independent_texts',
            'candidate_policy_texts','source_leads','map_text_missing','independent_text_missing']} for r in subset]
        for name, content in [('schools.json',schools),('polygons.json',polygons),('texts.json',texts)]:
            write(DEST/d/name,content)
        assert [r['school_id'] for r in schools] == [r['school_id'] for r in texts] == [r['school_id'] for r in polygons]
        districts.append({'district':d,'school_count':len(subset),'schools':f'{d}/schools.json',
            'polygons':f'{d}/polygons.json','texts':f'{d}/texts.json','combined':f'{d}/polygon与文字.json'})
    for r in rows:
        assert (DEST/r['source_archive']).is_file()
    write(DEST/'manifest.json', {'schema':'shenzhen_polygon_text_sources_v1','version':'20261110',
        'city':'深圳市','school_count':360,'districts':districts,'viewer':'index.html',
        'data_stage':'原始polygon与文字已整理；未构建模型运行所需的完整特征和答案',
        'training_dataset_ready':False,'runtime_requires_external_workspace':False,
        'coordinate_note':'BD-09约定；龙岗预计BD-09；不当作WGS84',
        'shared_equal_and_self_intersecting_paths_retained':True,'extra_eligibility_expanded':False})
    write(DEST/'接入校验.json', {'passed':True,'records':360,'district_counts':{d:sum(r['district']==d for r in rows) for d in DISTRICTS},
        'copied_files_byte_identical':copies,'all_source_archive_links_exist':True,
        'polygons_and_texts_join_by':'school_id','zip_required':False})
    (DEST/'README.txt').write_text('''深圳四区 polygon 与文字：项目内实际数据

位置：学区归属模型_代码库/data/深圳市/
打开 index.html 逐校查看。所有数据和来源均在本目录内，不需要解压或依赖外部工作区。

每区（福田区、南山区、龙华区、龙岗区）有：
  schools.json        学校目录，稳定school_id
  polygons.json       官方原始顶点、坐标系、几何标记、另存查询层
  texts.json          地图原文、独立正式文字、待核候选及来源（不含边界坐标）
  polygon与文字.json  完整逐校合并记录
  原始边界_BD09_非RFC7946.geojson
  逐校文字.txt

按school_id连接边界和文字；共享、相同范围和自交均保留。文字空缺不伪造补齐。
来源存档/是原始依据；来源与哈希.json保留来源路径及校验值，运行时使用包内路径。
manifest.json是当前数据清单，接入校验.json记录复制一致性。

在代码库根目录读取：
  from zoning.shenzhen_sources import load_texts, load_polygons
  texts = load_texts('龙岗区')
  polygons = load_polygons('龙岗区')

本次接入的是原始边界与文字。已有模型的zoning.data.load()读取特征、参照和答案；
本资料尚未完成这些转换，因此不将深圳自动加入现有训练配置。
历史正式文字的发布年、当前适用待核状态以及草案/候选性质均保留。

从工作区整理结果重新同步：python -X utf8 scripts/import_shenzhen_sources.py
''',encoding='utf-8-sig')
    # Preserve project-local reviewed supplements across an import of the older package.
    if (DEST/'福田区/文字补查_20261111/网络记录').is_dir():
        from supplement_futian_texts import main as supplement_futian
        supplement_futian(export=False)
    from exclude_minle_source import main as exclude_minle
    exclude_minle(export=False)
    from split_longhua_campuses import main as split_campuses
    split_campuses()
    from build_shenzhen_standard_dataset import main as build_current
    build_current()
    print(f'已接入 {DEST}，现用龙华、龙岗202条学校/校区；福田、南山保留历史资料。')


if __name__ == '__main__':
    main()
