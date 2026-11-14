"""从项目内既有资料生成四区全量来源目录，不联网、不改动文字和边界。"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1] / 'data/深圳市'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def main():
    rows = read('四区_polygon与文字.json')
    leads = read('来源线索索引.json')['sources']
    archives = read('来源与哈希.json')
    by_hash = {a['sha256']: a['package_path'] for a in archives}
    evidence = {}
    for a in archives:
        p = ROOT/a['package_path']
        assert p.is_file()
        if p.suffix in {'.json', '.md', '.txt'}:
            evidence[a['package_path']] = p.read_text(encoding='utf-8-sig')
    sources = {}

    def register(url, district, title, role, status, school_id=None, archive=None, year=None):
        if not url:
            return None
        key = 'source_' + hashlib.sha256(url.encode()).hexdigest()[:12]
        s = sources.setdefault(key, {'source_id':key,'url':url,'districts':[], 'title':title,
            'roles':[], 'status_notes':[], 'years':[], 'linked_school_ids':[], 'local_evidence':[]})
        if title and s['title'] == url:
            s['title'] = title
        for field,value in [('districts',district),('roles',role),('status_notes',status),('years',year),('linked_school_ids',school_id),('local_evidence',archive)]:
            if value is not None and value not in s[field]:
                s[field].append(value)
        return key

    # 先登记标题及文献状态，避免把草案或历史资料显示成当前正式全集。
    for lead in leads:
        register(lead['url'], lead['district']+'区', lead['title'], '独立文字或待核线索', lead['status']+'；'+lead['use'], year=lead['year'])
        if lead.get('notice'):
            register(lead['notice'],lead['district']+'区',lead['title']+'：发布页/目录','发布依据',lead['status'],year=lead['year'])

    annual_titles = {
        '龙华区':'深圳市龙华区2026年秋季公办学校招生范围、接受申请学位类型、咨询电话一览表',
        '龙岗区':'龙岗区2026年小学一年级学位申请指南（招生范围表）',
    }
    school_sources = []
    for r in rows:
        d = r['district']
        method = '2026-11-04用户从正常打开的官网页面导出运行时学校数据' if d == '龙岗区' else '2026-11-03正常浏览器采集的官方地图学校记录'
        mid = register(r['map_source_url'],d,d+'官方小学学区地图','polygon：学校记录MAPDATA',method,
                       r['school_id'],r['source_archive'])
        register(r['map_source_url'],d,d+'官方小学学区地图','地图文字：同一学校记录XQ','图文同平台，不能算两份独立证据',r['school_id'],r['source_archive'])
        item = {'district':d,'school_id':r['school_id'],'school_name':r['school_name'],
            'map_school_name':r['map_school_name'],'source_school_id':r['source_school_id'],
            'polygon_source':{'source_id':mid,'url':r['map_source_url'],'local_file':r['source_archive'],
                'record_key':'ORG_ID','record_value':r['source_school_id'],'field':'MAPDATA',
                'snapshot_time':r['snapshot_time'],'method':method,'crs':r['crs'],
                'alternate_query_boundary_present':bool(r['alternate_query_boundary'])},
            'map_text_source':{'source_id':mid,'url':r['map_source_url'],'local_file':r['source_archive'],
                'record_key':'ORG_ID','record_value':r['source_school_id'],'field':'XQ','missing':r['map_text_missing']},
            'independent_text_sources':[], 'candidate_text_sources':[], 'other_source_leads':[]}
        for field,dest,role in [('independent_texts','independent_text_sources','已直接配对独立文字'),
                                ('candidate_policy_texts','candidate_text_sources','草案或待核学校身份的文字候选')]:
            for t in r[field]:
                original = t.get('original_record',{})
                sha = original.get('source_file_sha256') or original.get('policy_pdf_sha256')
                archive = t.get('source_archive') or by_hash.get(sha)
                sid = register(t['source_url'],d,t.get('source_title') or annual_titles.get(d,t['source_url']),role,
                    t['identity_status']+'；'+t['applicability'],r['school_id'],archive,t['year'])
                item[dest].append({'source_id':sid,'url':t['source_url'],'year':t['year'],
                    'source_position':t['source_position'],'identity_status':t['identity_status'],
                    'applicability':t['applicability'],'original_document_local_file':archive,
                    'policy_record_id':t.get('policy_record_id')})
                if t.get('identity_evidence_url'):
                    register(t['identity_evidence_url'],d,'荔园校区更名的2019具名新闻报道','历史校名对应佐证',
                        '仅用于校区更名对应，不证明范围沿用',r['school_id'],t.get('identity_evidence_archive'),2019)
        for lead in r['source_leads']:
            sid = register(lead['url'],d,lead['title'],'学校相关线索（不作已核验对应）',lead['status'],r['school_id'],archive=lead.get('local_evidence'),year=lead['year'])
            item['other_source_leads'].append({'source_id':sid,'url':lead['url'],'title':lead['title'],'status':lead['status']})
        school_sources.append(item)

    # 保留年度表中尚未配对到地图记录的文字来源。
    unlinked_annual = read('龙华年度文字_未直接配对地图记录.json')
    for t in unlinked_annual:
        register(t['source_url'],'龙华区',annual_titles['龙华区'],'年度原表未直接配对条目',f'未直接配对的{len(unlinked_annual)}条年度记录另存，未丢弃',
                 archive=by_hash.get(t.get('source_file_sha256')),year=t['year'])

    # 承接核查、逐校核验文档中的官方支持链接也全部保留，区分其用途。
    for filename, body in evidence.items():
        if not filename.endswith('.md'):
            continue
        district = next((d for d in ['龙岗区','龙华区','福田区','南山区'] if d[:2] in filename),None)
        for title,url in re.findall(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',body):
            if not any(host in url for host in ['szft.gov.cn','szftedu.cn','szns.gov.cn','sz.gov.cn','szlhq.gov.cn','lg.gov.cn','lggov.cn']):
                continue
            register(url,district or '跨区',title,'核验/历史承接依据','作为核验或发布链依据，不自动视为逐校现行文字',archive=filename)

    for s in sources.values():
        for filename,body in evidence.items():
            if s['url'] in body and filename not in s['local_evidence']:
                s['local_evidence'].append(filename)
        s['linked_school_count'] = len(s['linked_school_ids'])
        s['local_evidence_types'] = sorted({Path(p).suffix for p in s['local_evidence']})
    for r in school_sources:
        for field in ['independent_text_sources','candidate_text_sources']:
            for t in r[field]:
                t['local_evidence'] = sources[t['source_id']]['local_evidence']

    source_list = sorted(sources.values(),key=lambda s:(s['districts'][0],s['title']))
    result = {'date':'2026-11-11','scope':f'当前四区{len(school_sources)}校实际polygon/文字来源，加已登记历史、草案及核验依据；福田包含2026-11-11网络补查，地图仍为原快照',
        'source_count':len(source_list),'school_count':len(school_sources),
        'local_evidence_note':'JSON/MD网页读取存档不等于原始PDF；已保存PDF/XLSX单独注明。',
        'sources':source_list}
    # 保留旧文件名作为兼容入口；新入口不再把条数写死在名称中。
    for filename,value in [('四区全部来源.json',result),('逐校来源.json',school_sources),('360校逐校来源.json',school_sources)]:
        (ROOT/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for district in ['福田区','南山区','龙华区','龙岗区']:
        value={'district':district,'local_paths_relative_to':'data/深圳市/',
            'sources':[s for s in source_list if district in s['districts']],
            'schools':[r for r in school_sources if r['district']==district]}
        (ROOT/district/'来源.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    E=lambda s:html.escape(str(s))
    A=lambda title,url:f'<a href="{E(url)}" target="_blank" rel="noopener">{E(title)}</a>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深圳四区全部来源</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei";max-width:1250px;margin:30px auto;padding:0 20px;color:#203b48}h1{font-size:26px}h2{font-size:21px;margin-top:35px}a{color:#006b78;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #d8e2e6;padding:10px;text-align:left;vertical-align:top}th{background:#edf5f6}article,details.school{border:1px solid #d8e2e6;padding:16px;margin:12px 0;border-radius:8px}summary{cursor:pointer}small,.note{color:#637b88}input{padding:12px;width:100%;box-sizing:border-box;font:inherit}code{overflow-wrap:anywhere}p{margin:8px 0}.path{font-size:12px}.source-title{font-weight:650;font-size:17px}</style><h1>深圳四区 · 文字与 polygon 全部来源</h1>',
        f'<p>{len(school_sources)} 校逐条可追溯 · {len(source_list)} 个去重来源链接 · 2026-11-11整理</p>',
        '<p class="note">地图使用已有快照；福田另含11月11日逐校网络补查。polygon取MAPDATA，地图原文取同条记录XQ；两者来自同一平台。独立正式文件、历史文件、草案、第三方转载和未取得全文的线索分别标明。</p>',
        '<p>'+A('逐校查看边界和文字','index.html')+' · '+A('完整来源JSON','四区全部来源.json')+' · '+A('逐校来源JSON','逐校来源.json')+'</p>',
        '<input id="q" placeholder="搜索学校、来源标题、区名或网址" aria-label="搜索全部来源"><h2>官方文档、地图入口与其他来源线索</h2>']
    for s in source_list:
        parts.append(f'<article class="item" id="{s["source_id"]}"><div class="source-title">{A(s["title"],s["url"])}</div><div>{E("、".join(s["districts"]))} · {E("；".join(s["roles"]))}</div><div>{A(s["url"],s["url"])}</div>')
        parts.extend('<p>'+E(note)+'</p>' for note in s['status_notes'])
        parts.append(f'<small>关联学校 {s["linked_school_count"]} 条（含候选/线索，不代表全部核验通过）</small>')
        parts.append('<details><summary>本地证据文件（网页读取存档与原始附件分开识别）</summary>')
        for f in s['local_evidence']:
            kind='原始附件' if Path(f).suffix in {'.pdf','.xlsx'} else '采集记录/整理与核验记录'
            parts.append('<div class="path">'+E(kind)+' · '+A(f,f)+'</div>')
        parts.append('</details></article>')
    parts.append(f'<h2>{len(school_sources)}校逐校来源</h2>')
    for r in school_sources:
        p=r['polygon_source'];m=r['map_text_source']
        parts.append(f'<details class="school item"><summary>{E(r["district"])} · {E(r["school_name"])} · {E(r["source_school_id"])}</summary>')
        parts.append('<p>polygon：'+A('官方地图',p['url'])+' · '+A('本地原始记录',p['local_file'])+'</p>')
        parts.append('<p>'+E(p['method'])+'；快照 '+E(p['snapshot_time'])+'；ORG_ID='+E(p['record_value'])+'；字段 MAPDATA。</p>')
        parts.append('<p>地图文字：同一原始记录的 XQ 字段；'+('原文缺失。' if m['missing'] else '原文已保存。')+'</p>')
        for field,label in [('independent_text_sources','已配对独立文字'),('candidate_text_sources','草案/待核候选'),('other_source_leads','历史、转载或待核线索')]:
            parts.append('<p><b>'+label+'</b>：'+('无已登记对应' if not r[field] else '')+'</p>')
            for t in r[field]:
                parts.append('<p>'+A(t.get('title',str(t.get('year',''))+' '+t.get('source_position','')),t['url'])+' · '+A('来源详情','#'+t['source_id'])+'<br>'+E(t.get('applicability',t.get('status','')))+'</p>')
        parts.append('</details>')
    parts.append('<script>document.getElementById("q").oninput=function(){const q=this.value.toLowerCase().trim();document.querySelectorAll(".item").forEach(e=>{e.hidden=q&&!e.textContent.toLowerCase().includes(q);});};</script></html>')
    (ROOT/'四区全部来源.html').write_text(''.join(parts),encoding='utf-8')
    page=ROOT/('index_四区来源历史.html' if (ROOT/'当前数据集.json').exists() else 'index.html')
    text=page.read_text(encoding='utf-8')
    if 'href="四区全部来源.html"' not in text:
        text=text.replace('<div class="toplinks">','<div class="toplinks"><a href="四区全部来源.html">四区全部来源（逐校）</a>',1)
        page.write_text(text,encoding='utf-8')
    manifest = read('manifest.json')
    assert len(school_sources)==manifest['school_count']
    assert Counter(r['district'] for r in school_sources)=={d['district']:d['school_count'] for d in manifest['districts']}
    assert all((ROOT/r['polygon_source']['local_file']).exists() for r in school_sources)
    for s in source_list:
        assert all((ROOT/f).exists() for f in s['local_evidence'])
    print(json.dumps({'schools':len(school_sources),'source_links':len(source_list),'all_local_evidence_paths_exist':True,
        'direct_independent_text_schools':sum(bool(r['independent_text_sources']) for r in school_sources)},ensure_ascii=False))


if __name__=='__main__':
    main()
