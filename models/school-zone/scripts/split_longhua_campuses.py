"""按2026年度表拆分校区实体，保留每条边界的官方原始记录来源。"""
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/深圳市'
AUDIT = '校区拆分_20261111'
GUANHU = 'sz:longhua:school:496FFFE16B935558E053BC09A8C0A5ED'
HUAREN = 'sz:longhua:school:c6a72391361d4221817db0289a292060'
HUAMEI = 'sz:longhua:school:92b2600af53243cd8652d9576d56df41'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def write(name, value):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    rows = read('四区_polygon与文字.json')
    before = copy.deepcopy(rows)
    old = {r['school_id']:r for r in rows}
    if GUANHU in old:
        assert HUAREN in old and HUAMEI in old
        write(AUDIT+'/拆分前记录.json', [old[k] for k in (GUANHU, HUAREN, HUAMEI)])
        changes = {}
        for parent, campuses in [(GUANHU, [('观湖','guanhu'), ('高新','gaoxin')]), (HUAREN,[('华仁','huaren')])]:
            source = old[parent]
            children = []
            for campus, suffix in campuses:
                r = copy.deepcopy(source)
                matches = [t for t in source['candidate_policy_texts'] if campus+'校区' in t['school_name_raw']]
                assert len(matches)==1
                t = copy.deepcopy(matches[0])
                t['identity_status'] = '已按官方年度表拆分校区对应；来源地图为合并校区记录'
                r.update(school_id=parent+':'+suffix,
                    school_name=re.sub(r'（小学部、初中部）', '（小学部）', t['school_name_raw'].replace('\n','')),
                    independent_texts=[t], candidate_policy_texts=[], independent_text_missing=False,
                    campus_derivation={'parent_school_id':parent,'campus':campus,'method':'从官方合并地图条目拆出年度表校区；原始顶点不变','official_map_record_is_combined':True})
                r['notes']=[n for n in r['notes'] if not n.startswith('候选文字仅供核对')]
                r['notes'].append('2026-11-11按用户决定分校区记录；学校ID独立，地图原名和source_school_id保留；相同polygon不合并学校。')
                children.append(r)
            changes[parent]=children
        rows = [child for r in rows for child in changes.get(r['school_id'],[r])]
        assert next(r for r in rows if r['school_id']==HUAMEI)==old[HUAMEI]
        assert all(r in rows for r in before if r['school_id'] not in changes)
    else:
        assert all(s in old for s in (GUANHU+':guanhu',GUANHU+':gaoxin',HUAREN+':huaren',HUAMEI))
    lh = [r for r in rows if r['district']=='龙华区']
    assert len(lh)==71 and len(rows)==360
    assert len({r['school_id'] for r in rows})==len(rows)
    assert len({r['school_name'] for r in lh})==71
    byid={r['school_id']:r for r in rows}
    assert byid[GUANHU+':guanhu']['vertices']==byid[GUANHU+':gaoxin']['vertices']
    assert all('民乐实验小学' not in r['school_name'] for r in lh)
    originals={r['school_id']:r for r in read(AUDIT+'/拆分前记录.json')}
    assert byid[HUAMEI]==originals[HUAMEI]
    for sid,parent in [(GUANHU+':guanhu',GUANHU),(GUANHU+':gaoxin',GUANHU),(HUAREN+':huaren',HUAREN)]:
        for key in ('vertices','source_mapdata','source_school_id','map_school_name','map_scope_text','source_archive'):
            assert byid[sid][key]==originals[parent][key]
    write('四区_polygon与文字.json',rows)
    write('龙华区/polygon与文字.json',lh)
    for filename in ('schools.json','polygons.json','texts.json'):
        keys=list(read('龙华区/'+filename)[0])
        write('龙华区/'+filename,[{k:r[k] for k in keys}|({'campus_derivation':r['campus_derivation']} if 'campus_derivation' in r else {}) for r in lh])
    geo=read('龙华区/原始边界_BD09_非RFC7946.geojson')
    properties=list(geo['features'][0]['properties'])
    geo['features']=[]
    for r in lh:
        coords=copy.deepcopy(r['vertices'])
        if coords[-1]!=coords[0]:coords.append(coords[0])
        geo['features'].append(dict(type='Feature',id=r['school_id'],properties={k:r[k] for k in properties},geometry=dict(type='Polygon',coordinates=[coords])))
    write('龙华区/原始边界_BD09_非RFC7946.geojson',geo)
    pending=[r for r in read('待核事项.json') if r['district']!='龙华区']
    for r in lh:
        pending.append(dict(school_id=r['school_id'],district=r['district'],school_name=r['school_name'],map_text_missing=r['map_text_missing'],
            independent_text_missing=r['independent_text_missing'],candidate_policy_names=[t['school_name_raw'] for t in r['candidate_policy_texts']],
            geometry_issue=r['geometry_issue'],alternate_query_boundary=bool(r['alternate_query_boundary']),polygon_vs_policy_status=r['polygon_vs_policy_status'],notes=r['notes']))
    pending_byid={r['school_id']:r for r in pending}
    write('待核事项.json',[pending_byid[r['school_id']] for r in rows])
    linked={t['policy_record_id'] for r in lh for t in r['independent_texts'] if t.get('policy_record_id')}
    unlinked=[r for r in read('龙华年度文字_未直接配对地图记录.json') if r['record_id'] not in linked]
    write('龙华年度文字_未直接配对地图记录.json',unlinked)
    lines=[]
    for r in lh:
        lines += [r['school_name'],'地图原名：'+r['map_school_name'],'学校ID：'+r['school_id'],'原始地图ID：'+r['source_school_id'],'地图原文：',r['map_scope_text'] or '【缺失】','地图来源：'+r['map_source_url']]
        for field,label in [('independent_texts','独立文字'),('candidate_policy_texts','候选文字')]:
            for t in r[field]:lines += [f'{label}（{t["year"]}）：',t['scope_text'],'来源：'+t['source_url']+' '+t['source_position'],t['identity_status']]
        lines += ['备注：'+'；'.join(r['notes']),'','='*65,'']
    (ROOT/'龙华区/逐校文字.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    summary=read('整理汇总.json')
    summary.update(school_count=len(rows),campus_split=AUDIT+'/拆分说明.json',longhua_annual_rows_not_directly_linked=len(unlinked))
    for s in summary['districts']:
        if s['district']=='龙华区':
            s.update(schools=len(lh),map_text_missing=sum(r['map_text_missing'] for r in lh),with_independent_text=sum(bool(r['independent_texts']) for r in lh),
                with_candidate_text=sum(bool(r['candidate_policy_texts']) for r in lh),valid_geometry=sum(r['geometry_valid'] for r in lh),
                self_intersection=sum('Self-intersection' in r['geometry_issue'] for r in lh),vertices_min=min(r['vertex_count'] for r in lh),vertices_max=max(r['vertex_count'] for r in lh))
    write('整理汇总.json',summary)
    manifest=read('manifest.json')
    manifest.update(school_count=len(rows),version='20261111_campuses_split',campus_split=AUDIT+'/拆分说明.json')
    for d in manifest['districts']:
        if d['district']=='龙华区':d['school_count']=len(lh)
    write('manifest.json',manifest)
    p=ROOT/('index_四区来源历史.html' if (ROOT/'当前数据集.json').exists() else 'index.html');page=p.read_text(encoding='utf-8')
    pattern=r'(<script id="payload" type="application/json">)(.*?)(</script>)'
    payload=json.loads(re.search(pattern,page,re.S)[2]);payload.update(records=rows,summary=summary['districts'])
    encoded=json.dumps(payload,ensure_ascii=False).replace('</','<\\/')
    page=re.sub(pattern,lambda m:m[1]+encoded+m[3],page,flags=re.S)
    page=page.replace('359 条当前使用的官方地图路径与逐校文字','360 条学校／校区记录与官方地图、文字').replace(' / 359 条',' / 360 条')
    page=page.replace('民乐实验小学按使用决定排除，龙华70条','民乐实验小学已排除；观湖、高新及华仁、华美分校区记录，龙华71条')
    p.write_text(page,encoding='utf-8')
    counts=dict(Counter(r['district'] for r in rows))
    write(AUDIT+'/拆分说明.json',dict(date='2026-11-11',user_decision='按校区分成独立学校记录，允许polygon相同',
        guanhu_gaoxin='共用原观湖、高新合并记录的55顶点polygon；各自对应年度表第8、10行',
        huaren='从华仁华美合名记录拆出华仁，沿用20顶点polygon，对应年度表第58行',
        huamei='已有独立华美记录继续使用15顶点polygon和第59行官方文字，不再新增一条华美；原合并条目不再作为学校',
        source_school_id_note='source_school_id指原始地图记录，可由两个独立school_id共用；不是新增官方地图记录',
        school_count=360,district_counts=counts,remaining_direct_pairing_candidates=['松和','桂花','新华小学部'],
        annual_rows_not_directly_linked=len(unlinked),original_archives_modified=False,geometry_policy_consistency_verified=False))
    for filename in ('接入校验.json','校验结果.json','页面验证.json'):
        check=read(filename)
        if not (ROOT/AUDIT/('调整前_'+filename)).exists():write(AUDIT+'/调整前_'+filename,check)
        if filename=='页面验证.json':check.update(payload_records=len(rows),payload_matches_current_json=True)
        else:
            check.update(records=len(rows),campus_split=AUDIT+'/拆分说明.json')
            if filename=='接入校验.json':check['district_counts']=counts
            else:check.update(geometry_valid=sum(r['geometry_valid'] for r in rows),non_simple_paths_retained=sum(not r['geometry_valid'] for r in rows),longhua_unlinked_annual_rows_preserved=len(unlinked))
        write(filename,check)
    p=ROOT/'README.txt';text=p.read_text(encoding='utf-8-sig')
    if AUDIT not in text:
        text+='\n当前校区拆分：观湖、高新独立学校ID，共用官方合并polygon；华仁使用原合名polygon，华美保留现有独立polygon，不重复建校。龙华71条、四区360条；民乐仍排除。详见 '+AUDIT+'/拆分说明.json。\n'
    p.write_text(text,encoding='utf-8-sig')
    from export_shenzhen_source_index import main as export_index
    export_index()
    print(json.dumps(dict(schools=len(rows),longhua=len(lh),distinct_school_ids=True,guanhu_gaoxin_same_polygon=True,huamei_original_retained=True),ensure_ascii=False))


if __name__=='__main__':main()
