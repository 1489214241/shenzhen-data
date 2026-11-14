"""按用户决定排除民乐当前使用记录；原始官方存档和历史核查不变。"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/深圳市'
SID = 'sz:longhua:school:c666c50f5acf4c60a4b71a54aef6ac36'
HONGSHAN = 'sz:longhua:school:270930b511e946c38951ecf46a65476a'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def write(name, value):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(export=True):
    before = read('四区_polygon与文字.json')
    excluded = [r for r in before if r['school_id'] == SID]
    audit = '使用排除/民乐实验小学_20261111.json'
    if excluded:
        write(audit, {
            'date': '2026-11-11', 'school_id': SID,
            'decision': '用户明确要求从当前使用数据排除民乐实验小学，保留红山中学民治校区小学部',
            'reason': '存在2026新生招生计划由红山民治小学部承接的政策线索，红山已有独立polygon；本次为数据选用决定，不认定学校撤销或更名',
            'evidence_status': '官方征求意见稿第1页已读取承接条款；正式公告已读取，正式附件条款仍待直接复核',
            'draft_url': 'https://www.szlhq.gov.cn/lhjyj/attachment/1/1716/1716491/12804444.pdf',
            'notice_url': 'https://www.szlhq.gov.cn/bmxxgk/jyj/dtxx_124232/tzgg_124234/content/post_12826036.html',
            'original_record': excluded[0]})
    assert (ROOT/audit).exists()
    rows = [r for r in before if r['school_id'] != SID]
    assert len(rows) == 359 and len({r['school_id'] for r in rows}) == 359
    assert any(r['school_id'] == HONGSHAN for r in rows)
    write('四区_polygon与文字.json', rows)
    for name in ('schools.json', 'polygons.json', 'texts.json', 'polygon与文字.json'):
        write('龙华区/'+name, [r for r in read('龙华区/'+name) if r['school_id'] != SID])
    name = '龙华区/原始边界_BD09_非RFC7946.geojson'
    geo = read(name)
    geo['features'] = [r for r in geo['features'] if r['properties']['school_id'] != SID]
    write(name, geo)
    write('待核事项.json', [r for r in read('待核事项.json') if r['school_id'] != SID])
    p = ROOT/'龙华区/逐校文字.txt'
    blocks = p.read_text(encoding='utf-8-sig').split('='*65)
    p.write_text(('='*65).join(b for b in blocks if SID.split(':')[-1] not in b), encoding='utf-8-sig')

    counts = Counter(r['district'] for r in rows)
    summary = read('整理汇总.json')
    summary.update(school_count=len(rows), all_original_paths_retained=False,
                   original_paths_preserved_in_archive=True, selection_exclusions=audit)
    summary['scope'] = '当前使用的四区资料；民乐按用户决定排除，官方原始存档保留'
    for s in summary['districts']:
        rs = [r for r in rows if r['district'] == s['district']]
        s.update(schools=len(rs), map_text_missing=sum(r['map_text_missing'] for r in rs),
                 with_independent_text=sum(bool(r['independent_texts']) for r in rs),
                 with_candidate_text=sum(bool(r['candidate_policy_texts']) for r in rs),
                 valid_geometry=sum(r['geometry_valid'] for r in rs),
                 self_intersection=sum('Self-intersection' in r['geometry_issue'] for r in rs),
                 vertices_min=min(r['vertex_count'] for r in rs), vertices_max=max(r['vertex_count'] for r in rs))
    write('整理汇总.json', summary)
    manifest = read('manifest.json')
    manifest.update(school_count=len(rows), version='20261111_minle_excluded', selection_exclusions=audit)
    for d in manifest['districts']:
        d['school_count'] = counts[d['district']]
    write('manifest.json', manifest)

    p = ROOT/('index_四区来源历史.html' if (ROOT/'当前数据集.json').exists() else 'index.html')
    page = p.read_text(encoding='utf-8')
    pattern = r'(<script id="payload" type="application/json">)(.*?)(</script>)'
    payload = json.loads(re.search(pattern, page, re.S)[2])
    payload.update(records=rows, summary=summary['districts'])
    encoded = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    page = re.sub(pattern, lambda m: m[1]+encoded+m[3], page, flags=re.S)
    page = page.replace('360 条官方地图路径与逐校文字', '359 条当前使用的官方地图路径与逐校文字')
    page = page.replace(' / 360 条', ' / 359 条')
    if '使用排除/民乐实验小学_20261111.json' not in page:
        page = page.replace('</header>', '<p>2026-11-11：民乐实验小学按使用决定排除，龙华70条；<a style="color:#bce7e0" href="使用排除/民乐实验小学_20261111.json">排除说明与原记录</a>。</p></header>', 1)
    p.write_text(page, encoding='utf-8')

    for name in ('接入校验.json', '校验结果.json', '页面验证.json'):
        old = read(name)
        history = '使用排除/调整前_'+name
        if not (ROOT/history).exists():
            write(history, old)
    write('接入校验.json', dict(passed=True, records=len(rows), district_counts=dict(counts),
        original_import_audit='使用排除/调整前_接入校验.json', selection_exclusions=audit,
        polygons_and_texts_join_by='school_id', zip_required=False))
    check = read('校验结果.json')
    check.update(records=len(rows), geometry_valid=sum(r['geometry_valid'] for r in rows),
                 non_simple_paths_retained=sum(not r['geometry_valid'] for r in rows),
                 selection_exclusions=audit, original_validation='使用排除/调整前_校验结果.json')
    write('校验结果.json', check)
    write('页面验证.json', dict(validation_mode='静态数据一致性检查；本轮未重跑浏览器布局检查',
        payload_records=len(rows), payload_matches_current_json=True,
        previous_browser_validation='使用排除/调整前_页面验证.json'))
    p = ROOT/'README.txt'
    text = p.read_text(encoding='utf-8-sig')
    if audit not in text:
        text += '\n2026-11-11使用调整：排除民乐实验小学，龙华70条、四区359条；红山小学部保留。\n原始官方地图仍保留71条龙华记录；排除依据和原记录见 '+audit+'。\n重新导入会自动应用此项排除。\n'
    p.write_text(text, encoding='utf-8-sig')

    for name in ('schools.json', 'polygons.json', 'texts.json', 'polygon与文字.json'):
        rs = read('龙华区/'+name)
        assert len(rs) == 70 and SID not in {r['school_id'] for r in rs}
        assert HONGSHAN in {r['school_id'] for r in rs}
    assert len(geo['features']) == 70
    assert rows == [r for r in before if r['school_id'] != SID]
    source = ROOT/read(audit)['original_record']['source_archive']
    assert SID.split(':')[-1] in source.read_text(encoding='utf-8')
    write('使用排除/民乐排除校验.json', dict(passed=True, school_count=len(rows), district_counts=dict(counts),
        remaining_combined_records_unchanged=True, hongshan_retained=True,
        original_map_record_retained=True, original_map_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    if export:
        from export_shenzhen_source_index import main as export_index
        export_index()
    print(json.dumps({'excluded':SID, 'counts':dict(counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
