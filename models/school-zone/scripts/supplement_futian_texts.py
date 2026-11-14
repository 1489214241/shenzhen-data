"""Merge reviewed Futian web evidence. Offline, idempotent, no polygon edits.

Run after import_shenzhen_sources.py. Network records are immutable input evidence;
school indices below refer to the frozen 78-row snapshot, guarded by names.
"""
from pathlib import Path
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1] / 'data/深圳市'
REL = '福田区/文字补查_20261111'
OUT = ROOT / REL
RUN = 'futian_web_20261111'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def texts(obj):
    if isinstance(obj, dict):
        if obj.get('type') == 'text':
            yield obj['text']
        else:
            for v in obj.values():
                yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)


def main(export=True):
    rows = read(ROOT/'四区_polygon与文字.json')
    ft = [r for r in rows if r['district'] == '福田区']
    assert len(ft) == 78
    # Only replace this supplement's own entries, retaining all earlier work.
    for r in ft:
        for field in ('independent_texts', 'candidate_policy_texts', 'source_leads'):
            r[field] = [t for t in r[field] if t.get('supplement_run') != RUN]
    baseline = {r['school_id']:len(r['independent_texts']) for r in ft}
    protected = [p for d in ('福田区','南山区','龙华区','龙岗区') for p in (ROOT/d).glob('*')
                 if p.is_file() and (p.name in ('polygons.json','schools.json') or p.suffix == '.geojson'
                    or (d != '福田区' and p.name in ('texts.json','polygon与文字.json','逐校文字.txt')))]
    before = {p.relative_to(ROOT).as_posix():digest(p) for p in protected}
    original_map = [(r['school_id'],r['map_scope_text'],r['vertices'],r['source_mapdata']) for r in rows]
    all_added = []

    def add(i, name, year, url, title, scope, position, archive, note='', candidate=False, raw_name=None, identity=None):
        r = ft[i]
        assert name in r['school_name'], (i, name, r['school_name'])
        t = dict(scope_text=scope, school_name_raw=raw_name or r['school_name'], year=year,
                 source_url=url, source_title=title, source_position=position,
                 source_archive=f'{REL}/{archive}', supplement_run=RUN,
                 identity_status=identity or '原文学校名与地图记录直接对应',
                 applicability=('征求意见稿，不作正式现行范围' if candidate else
                     '历史学校正式发布范围；2026是否完整沿用及与polygon一致性尚待核对') + ('；'+note if note else ''),
                 publication_status='draft' if candidate else 'historical_official',
                 formal_gold_eligible=False)
        r['candidate_policy_texts' if candidate else 'independent_texts'].append(t)
        all_added.append(dict(school_id=r['school_id'],school_name=r['school_name'],kind='candidate' if candidate else 'independent',**t))
        # Update earlier, now-outdated lead descriptions, preserving the old note explicitly.
        for lead in r['source_leads']:
            if lead['url'] == url:
                if 'previous_status' not in lead:
                    lead['previous_status'] = lead['status']
                lead.update(status='2026-11-11已核读并配对历史文字；现行适用待核',use=t['applicability'])

    add(8,'岗厦',2022,'https://gxxx.szftedu.cn/xyzx_474/xxxw_475/202205/t20220531_139894.html',
        '文天祥小学（岗厦）2022年小一招生简章','深南路南、福华路以北、福田路以西、彩田路以东。',
        '二、招生地段，网页L36-L38','网络记录/read_0.json')
    add(33,'景莲',2021,'https://jlxx.szftedu.cn/zjjl/xxgk_8845/202107/t20210705_133934.html',
        '景莲小学关于2021年秋季接收插班生工作的公告',
        '雨田路以西，新洲路以东，北环路以南，莲花路以北的莲花北住宅区1——36栋，吉莲大厦1-4栋和宝莲大厦1-3栋。',
        '1.招生范围，网页L29-L31','网络记录/read_more.json',
        '楼栋限制完整保留；2024转学公告未列出范围，不能据此证明沿用')
    add(21,'景鹏',2018,'https://jpxx.szftedu.cn/xwgk_421/fwck/201801/t20180118_95736.html',
        '景鹏小学2018秋季一年级招生学位预警信息',
        '红荔西路以南，新洲路以西，深南大道以北，香蜜湖路以东，主要包括紫荆、香蜜、梅岭等社区。',
        '景鹏小学招生地段，网页L40-L42','网络记录/read_next.json',
        '2026报名指南另有景田南片共享招生安排，不能仅凭旧四至认定现行完整范围')
    add(37,'梅山',2022,'https://msxx.szftedu.cn/xxgk_579/xzfc_585/202207/t20220705_140502.html',
        '梅山小学2022年秋季转学插班工作公告',
        '边防二线以南，福田-南山区界以东，北环大道以北，梅丽路以西，西转接梅林路，接梅山街以南，接梅林一村围墙以北，接边防二线。\n小区名称：梅山苑一期、梅山苑二期、颂德花园；林海山庄、尚书苑、高达苑、鸿达苑、碧云天、锦林新居、裕林花园、通业花园、绿景虹湾、景源会所；劳教所干警生活基地、红岭中学宿舍、梅山中学宿舍、市外事办服务处宿舍、水务局宿舍、华港公寓、道路公司宿舍、建安集团宿舍；下梅林村、丰田村。',
        '三、招生范围表；网页打开超时，搜索工具返回该公告全文','网络记录/targeted_04.json',
        '标题/落款为2022，申报时间却写2021；日期冲突保留，不自行纠正')
    add(60,'福新',2025,'https://fxxx.szftedu.cn/xydt_6489/xxgg_6490/202507/P020250703353232451586.pdf',
        '福新小学2025年秋季转学插班公告',
        '深南路以南，滨河路以北，新洲路以东，金田路以西。（学区代码为60）。',
        'PDF第1页，一、招生范围；已目视核对原页','网页原件/38c23919ac1665dc.pdf',
        'PDF元数据标题残留2010，正文标题/落款是2025；按正文记录')
    add(12,'新沙',2023,'https://ftxs.szftedu.cn/xx/tzgg/202305/t20230529_158616.html',
        '新沙小学2023年小一招生公告中的范围图片',
        '15区\n方位：滨河路以南，新洲路转市场路再转新洲二街以西，福强路以北，沙嘴路口转新洲九街再转新洲三街以东。\n住宅名称：朗晴馨洲、嘉洲豪园、嘉洲富苑、绿景新洋房、锦洲花园、工商宿舍楼、新洲苑、绿景新苑、骏皇嘉园、星河雅居、湖北宝丰花园、康达尔公寓、锦滨苑、福昌苑、国税综合大楼、沙尾综合楼（沙尾机楼）、润福园、金福苑、新洲祠堂村、祥云天都世纪、南溪新苑、中城天邑、新洲南村、新洲中心村（1-55号）、嘉堡润金座C栋、新南苑、雄鹰大厦、香江东苑',
        '三、新沙小学招生范围；附图下半部15区表格，人工逐字转录',
        '原始附件/新沙小学_2023招生范围.png',
        '图中文字原样保留，嘉堡润等名称不擅自按地图改写；图片URL见已核读来源')
    ft[12]['independent_texts'][-1]['image_url'] = 'https://ftxs.szftedu.cn/xx/tzgg/202305/W020230529374184353898.png'

    # 2015 scope document and 2019 reported renaming are distinct evidence.
    oldurl='https://lyxx.szftedu.cn/hdpt/201704/t20170425_27381.html'
    oldbody='\n'.join(texts(read(OUT/'网络记录/read_next.json')))
    line=lambda n:re.search(rf'^L{n}: (.*)$',oldbody,re.M).group(1)
    for i,name,raw,scope,pos in [
        (0,'通新岭','荔园小学南校区',line(141),'南校区学区楼盘01，L138-L145'),
        (41,'百花','荔园小学北校区',line(152)+'\n'+line(155)+'\n原文另注：'+line(163),'北校区L151-L163'),
        (62,'玮鹏','荔园小学西校区',line(169),'西校区L167-L173')]:
        add(i,name,2015,oldurl,'荔园小学南、北、西校区招生地段范围',scope,pos,
            '网络记录/read_next.json','正文日期2015-03-24；附图未转为文字；北校区正文两处排除住宅分别写长东、长乐，冲突未消解',
            raw_name=raw,identity='2019-02-18南方都市报具名报道明确南→通新岭、北→百花、西→玮鹏；属于媒体佐证的历史校名映射，非2026范围确认')
        ft[i]['independent_texts'][-1]['identity_evidence_url']='https://www.sohu.com/a/295408520_161795'
        ft[i]['independent_texts'][-1]['identity_evidence_archive']=f'{REL}/网络记录/alias_identity.json'
        ft[i]['independent_texts'][-1]['scope_completeness']='已保存正文；附图未转录，不能称完整边界核验'

    draft='https://www.szftedu.cn/xxgk/tzgg/202505/t20250514_168847.html'
    for i,name,scope,pos in [
        (35,'梅丽','梅秀路及梅村路以东、梅林路以南、中康路以西、梅华路及北环大道以北合围区域。','小学部分L72'),
        (50,'梅华','梅林山及中康路以东、梅林山以南、彩田路以西、梅林路及北环大道以北合围区域。','小学部分L73')]:
        add(i,name,2025,draft,'福田区2025年新学校学位优化供给方案（征求意见稿）',scope,pos,
            '网络记录/current_and_draft.json','本轮未找到正式定稿原文；初中范围不混入小学',candidate=True)

    def lead(i,title,url,year,status,detail,archive,use='仅作追溯线索，不计入独立官方范围文字覆盖'):
        value=dict(district='福田',year=year,title=title,url=url,status=status,detail=detail,use=use,
                   local_evidence=f'{REL}/{archive}',supplement_run=RUN)
        ft[i]['source_leads'].append(value)

    # Whole historical table: associate links, not a second copy of map XQ.
    tableurl='https://m.shenchuang.com/show/1598576.shtml'
    tablelines={}
    for file in (OUT/'网络记录').glob('*.json'):
        for body in texts(read(file)):
            for part in body.split('--------------------------------------------------------------------------------'):
                if 'Source: open' in part and tableurl in part:
                    tablelines.update({int(n):s for n,s in re.findall(r'^L(\d+): (.*)$',part,re.M)})
    headings=[]
    for n,s in sorted(tablelines.items()):
        m=re.match(r'(\d+)\. (.*)',s)
        if m:headings.append((int(m[1]),m[2],n))
    assert [h[0] for h in headings] == list(range(1,60))
    mapping=[63,45,64,34,46,40,39,77,27,10,56,11,12,42,2,13,58,17,60,65,23,8,75,7,5,6,0,62,1,70,41,18,19,3,4,28,29,20,53,49,48,21,32,14,57,30,33,38,61,54,52,16,67,59,50,35,15,36,37]
    table_index=[]
    for (n,name,ln),i in zip(headings,mapping):
        detail=f'2021表第{n}项：{name}。同名或历史名候选关联；并未逐条完成更名与范围沿用核查。第三方注明深窗综合，未给逐校官方原文链接；不能当作独立官方文本。'
        lead(i,'2021福田公办小学范围汇总：'+name,tableurl,2021,'第三方历史汇总；59项，不是2026全区全集',detail,'网络记录/aliases_and_table_end.json')
        table_index.append(dict(item=n,school_name_raw=name,line=ln,candidate_school_id=ft[i]['school_id'],candidate_school_name=ft[i]['school_name'],url=tableurl,association_status='historical_name_lead_only'))
    write(OUT/'第三方59项索引.json',table_index)

    extra=[(0,'https://m.bendibao.com/show923798.html',0),(2,'https://m.bendibao.com/show923736.html',0),
           (62,'https://m.bendibao.com/show923799.html',0),(65,'https://sz.bendibao.com/qinzi/202161/865428.shtm',0),
           (48,'https://m.bendibao.com/show923790.html',1),(61,'https://m.bendibao.com/show923665.html',1),
           (64,'https://m.bendibao.com/show923730.html',1),(22,'https://m.bendibao.com/show924993.html',1),
           (72,'https://sz.bendibao.com/qinzi/2022512/889452.shtm',2),(26,'https://m.bendibao.com/show897342.html',2),
           (25,'https://sz.bendibao.com/qinzi/2023421/918550.shtm',2),(68,'https://m.bendibao.com/show893428.html',2)]
    for i,url,b in extra:
        lead(i,'本地宝范围资料：'+ft[i]['school_name'],url,'以原页为准','第三方历史资料；已打开原页',
             '部分文章注明福田教育网，但未链接逐校政策原文；可能转录地图XQ，不能新增官方独立证据。','网络记录/thirdparty_'+str(b)+'.json')

    current='https://mlzx.szftedu.cn/jyfw/tzgg_111/202606/t20260608_176689.html'
    lead(76,'2026报名指南：香梅片区共享招生规则',current,2026,'正式招生关系；不是本校独立四至',
         '侨香南校地段可将侨香南校或香蜜作为第一志愿；福田区外国语小学地段先以原校为第一志愿，可选侨香南校或香蜜为提前志愿。香蜜原始XQ仍为空，不用申请资格拼造本校范围。',
         '网络记录/targeted_2.json','保留条件式招生关系；不修改polygon或填补XQ')
    for i,title,url,year,archive,detail in [
        (33,'景莲小学2024转学公告','https://jlxx.szftedu.cn/xyzx_8847/xxgg_8849/202407/t20240705_164851.html',2024,'网络记录/read_more.json','全文没有列出具体范围，不能证明旧地段沿用'),
        (35,'梅丽小学2025转学公告','https://ftml.szftedu.cn/xyzx_532/gg/tzgg_537/202507/t20250707_169663.html',2025,'网络记录/targeted_04.json','全文没有列出具体范围，不能代替2025草案的正式定稿'),
        (39,'上沙学校招生入口','https://sszx.szftedu.cn/xxzs/xxzs/202603/t20260324_172675.html',2026,'网络记录/read_deep.json','仅列招生咨询电话，没有范围文字'),
        (42,'石厦学校转载2022报名指南','https://sxxx.szftedu.cn/zszp/202205/t20220530_139870.html',2022,'网络记录/read_deep.json','指向地段代码图，没有单独列出本校范围')]:
        lead(i,title,url,year,'已读；不含可补录的本校范围',detail,archive)

    for meta_file in (OUT/'网页原件').glob('*.metadata.json'):
        meta=read(meta_file)
        if 'mp.weixin.qq.com' not in meta['url']:continue
        meta['content_status']='captcha_page_not_article' if 'captcha' in meta.get('final_url','') else 'unverified'
        write(meta_file,meta)
        i=58 if '_qSyJd' in meta['url'] else 61 if 'g_fGM' in meta['url'] else 28
        lead(i,'学校官网链接的2026招生/转学公告',meta['url'],2026,'官网确认发布链接；微信验证码页，正文未取得',
             '已从学校官网追溯至公众号；HTTP200返回验证码页面，不算取得公告全文。',meta_file.relative_to(OUT).as_posix())

    for r in ft:r['independent_text_missing']=not bool(r['independent_texts'])
    # Exact field preservation: no map scope or geometry substitutions.
    assert original_map == [(r['school_id'],r['map_scope_text'],r['vertices'],r['source_mapdata']) for r in rows]
    write(ROOT/'四区_polygon与文字.json',rows)
    write(ROOT/'福田区/polygon与文字.json',ft)
    oldtexts=read(ROOT/'福田区/texts.json')
    for t,r in zip(oldtexts,ft):
        assert t['school_id']==r['school_id']
        for k in ('independent_texts','candidate_policy_texts','source_leads','independent_text_missing'):t[k]=r[k]
    write(ROOT/'福田区/texts.json',oldtexts)
    summary=read(ROOT/'整理汇总.json')
    summary.update(date='2026-11-11',scope='四区原始资料；福田追加逐校网络检索及历史学校范围文字')
    for s in summary['districts']:
        if s['district']=='福田区':
            s['with_independent_text']=sum(bool(r['independent_texts']) for r in ft)
            s['with_candidate_text']=sum(bool(r['candidate_policy_texts']) for r in ft)
    write(ROOT/'整理汇总.json',summary)
    pending=read(ROOT/'待核事项.json')
    byid={r['school_id']:r for r in ft}
    for p in pending:
        if p['school_id'] in byid:
            r=byid[p['school_id']];p['independent_text_missing']=r['independent_text_missing']
            p['candidate_policy_names']=[t['school_name_raw'] for t in r['candidate_policy_texts']]
    write(ROOT/'待核事项.json',pending)

    lines=[]
    for r in ft:
        lines += [r['school_name'],'ID：'+r['school_id'],'地图原文：',r['map_scope_text'] or '【原始XQ为空】','地图来源：'+r['map_source_url']]
        for field,label in [('independent_texts','独立发布文字（历史适用另核）'),('candidate_policy_texts','草案/待核文字')]:
            lines.append(label+'：')
            if not r[field]:lines.append('无已配对记录')
            for t in r[field]:lines += [str(t['year'])+' '+t['school_name_raw'],t['scope_text'],t['source_url'],t['source_position'],t['identity_status'],t['applicability']]
        lines.append('其他来源/追溯线索：')
        for t in r['source_leads']:lines += [t['title']+' '+t['url'],t['status']+'；'+t['use']]
        lines += ['','='*65,'']
    (ROOT/'福田区/逐校文字.txt').write_text('\n'.join(lines),encoding='utf-8-sig')

    pagefile=ROOT/('index_四区来源历史.html' if (ROOT/'当前数据集.json').exists() else 'index.html')
    page=pagefile.read_text('utf-8')
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',page,re.S)[1])
    payload['records']=rows;payload['summary']=summary['districts']
    encoded=json.dumps(payload,ensure_ascii=False).replace('</','<\\/')
    page=re.sub(r'(<script id="payload" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],page,flags=re.S)
    page=page.replace('名称 / 校区对应尚待核实','草案 / 适用或校区对应待核').replace('待确认身份 · ','待核文字 · ').replace('有待核身份的文字候选','有草案或待核文字').replace('更多历史或草案来源线索','更多历史、转载或待核来源线索')
    if '福田补查：78校逐校清单' not in page:
        page=page.replace('</header>',f'<p><a style="color:#bce7e0" href="{REL}/index.html">福田补查：78校逐校清单</a> · 文字补查于2026-11-11</p></header>',1)
    page=re.sub(r'<a href="深圳四区_polygon与文字.zip" download>.*?</a>','',page)
    pagefile.write_text(page,encoding='utf-8')

    audit=[]
    for i,r in enumerate(ft):
        file=OUT/'网络记录'/f'search_{i//4*4:02d}.json'
        search=read(file);q=search['queries'][i%4]
        name=q['q'] if isinstance(q,dict) else q
        if not isinstance(q,dict):
            if i<12:name='深圳 福田 '+name+' 招生 地段 范围'
            elif i<28:name='深圳 '+name+' 招生范围 招生地段 -预算 -决算 -章程'
            elif i<44:name='福田 '+name+' 招生地段 范围 site:szftedu.cn'
            elif i<60:name='深圳 '+name+' "招生" "范围"'
            else:name='深圳 福田 '+name+' 学区 招生地段'
        audit.append(dict(school_id=r['school_id'],school_name=r['school_name'],map_text_present=bool(r['map_scope_text']),
            first_pass_query=name,first_pass_archive=file.relative_to(ROOT).as_posix(),
            independently_published_scope_found=bool(r['independent_texts']),
            independent_texts=r['independent_texts'],candidate_texts=r['candidate_policy_texts'],source_leads=r['source_leads'],
            current_scope_fully_verified=False,remaining='仍需核对发布年份之后调整、当前学校身份和polygon空间一致性；缺少独立文字不代表网上不存在'))
    write(OUT/'78校逐校核查.json',audit)
    # Store final versions, including image/identity evidence added after add().
    reviewed=[dict(school_id=r['school_id'],school_name=r['school_name'],kind=field,**t) for r in ft
              for field in ('independent_texts','candidate_policy_texts') for t in r[field] if t.get('supplement_run')==RUN]
    write(OUT/'已核读来源.json',reviewed)
    counts=dict(total=78,first_pass_searched=78,map_text_present=77,
        independent_before=sum(bool(n) for n in baseline.values()),
        independent_after=sum(bool(r['independent_texts']) for r in ft),
        newly_paired_schools=sum(bool(r['independent_texts']) and not baseline[r['school_id']] for r in ft),
        still_without_independent=sum(not r['independent_texts'] for r in ft),
        third_party_table_rows=59,third_party_linked_schools=sum(any('第三方' in l['status'] for l in r['source_leads']) for r in ft),
        draft_schools=2,exhaustive_web_coverage_claimed=False)
    write(OUT/'补查汇总.json',counts)
    report='''福田小学学区文字补查（2026-11-11）

检索范围：当前78条学校记录逐校首轮检索；继续查政府、教育局、学校官网、学校公众号、历年公告、深圳之窗、本地宝、媒体更名报道等。完整查询与读取返回在网络记录/。不是声称已经穷尽互联网。
范围与计数：独立发布的范围文字由12校增加到21校，本次补录9校。21校中9校关联2026正式方案，12校是历史学校/政策文件；不等于21校现行范围完整核验通过。57校仍缺已配对独立官方范围文字，地图XQ有77校；香蜜XQ原始为空。
补录：文天祥（岗厦）、景莲、景鹏、梅山、福新、新沙、荔园通新岭、百花、玮鹏。前4份原来已有线索，此次直接整理配对；福新由搜索片段提升为完整PDF；新沙原图含四至与住宅表；荔园3校以2019具名媒体报道核对旧名，2015正文附图尚未转录。
草案：梅丽、梅华2025四至另存candidate_policy_texts，未找到正式定稿原文，不能当正式范围。初中范围未混入小学。
转载：深圳之窗2021表只有59项；只关联来源线索，不冒充2026全区原表。另核读12份本地宝页面，不把转录XQ当成独立官方证据。
访问问题：华新、明德碧海、明德香蜜2026公众号链接由学校官网确认，但正文被微信验证码页拦住；HTTP200不等于成功。保留链接与实际响应，不计文字覆盖。
重要细节：景莲楼栋限定保留；梅山公告2021/2022日期混用保留；福新PDF元数据误写2010而正文2025；新沙新洲中心村限定1-55号；荔园北校区长东/长乐排除项冲突保留；香蜜共享申请资格不拼成本校四至。

直接使用：本区../texts.json、../polygon与文字.json、../逐校文字.txt，及全市四区_polygon与文字.json和index.html均已同步。78校逐校核查.json记录每校原文、来源、查询和缺口。网页原件/含真实下载响应，metadata说明URL和HTTP状态；验证码响应不属于公告原文。网络记录/为工具读取记录，不是原始PDF。
复现：在代码库运行 python -X utf8 scripts/supplement_futian_texts.py；重新运行import_shenzhen_sources.py也自动应用本补充，避免被旧包覆盖。
本次不改原始地图文字、polygon、学校ID、其他三区数据或Gold。
'''
    (OUT/'补查说明.txt').write_text(report,encoding='utf-8-sig')
    E=lambda x:html.escape(str(x))
    A=lambda title,url:f'<a href="{E(url)}" target="_blank" rel="noopener">{E(title)}</a>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>福田78校文字补查</title><style>body{font:15px/1.8 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#203b48}a{color:#006b78;overflow-wrap:anywhere}details{padding:15px;margin:12px 0;border:1px solid #ccd9df;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;background:#f2f6f7;padding:12px}input{padding:12px;width:95%;font:inherit}.note{color:#687b86}</style>',
           '<h1>福田78校 · 范围文字与来源补查</h1>',f'<p>独立发布范围文字：{counts["independent_before"]} → {counts["independent_after"]} 校；仍缺 {counts["still_without_independent"]} 校。包含历史文字，不能直接视为2026现行范围。</p>',
           '<p>'+A('补查说明','补查说明.txt')+' · '+A('逐校JSON','78校逐校核查.json')+' · '+A('可用文字数据','../texts.json')+' · '+A('地图与文字总览','../../index.html')+'</p>',
           '<p class="note">59项历史转载只作追溯线索；学校原文、草案、缺口分别列出。原始polygon和XQ均保留。公众号验证码、日期冲突和楼栋限制均有记录。</p>',
           '<input id="q" placeholder="搜索学校、道路、住宅或来源">']
    for r in audit:
        parts.append('<details class="school"><summary>'+E(r['school_name'])+' · '+('有独立发布范围（年份见下）' if r['independently_published_scope_found'] else '独立官方范围仍缺')+'</summary>')
        parts.append('<p>首次检索：'+E(r['first_pass_query'])+'</p>')
        for field,label in [('independent_texts','独立发布范围'),('candidate_texts','草案/待核文字')]:
            for t in r[field]:
                parts.append('<p><b>'+E(label+' · '+str(t['year']))+'</b> '+A(t.get('source_title',t['school_name_raw']),t['source_url'])+'</p><pre>'+E(t['scope_text'])+'</pre><p>'+E(t['identity_status']+'；'+t['applicability'])+'</p>')
                if t.get('source_archive'):parts.append(A('本地原件/读取记录','../../'+t['source_archive']))
        for l in r['source_leads']:parts.append('<p>'+A(l['title'],l['url'])+'<br>'+E(l['status']+'；'+l.get('detail',''))+'</p>')
        parts.append('</details>')
    parts.append('<script>document.getElementById("q").oninput=function(){const q=this.value.toLowerCase().trim();document.querySelectorAll(".school").forEach(e=>{e.hidden=q&&!e.textContent.toLowerCase().includes(q);});};</script></html>')
    (OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')

    # Register input evidence, not mutable audit outputs, in the city source index.
    archives=[a for a in read(ROOT/'来源与哈希.json') if not a['package_path'].startswith(REL+'/')]
    for folder in ('网络记录','网页原件','原始附件'):
        for p in sorted((OUT/folder).glob('*')):
            if p.is_file():archives.append(dict(original_path=str(p),package_path=p.relative_to(ROOT).as_posix(),sha256=digest(p)))
    write(ROOT/'来源与哈希.json',archives)
    leads=read(ROOT/'来源线索索引.json')
    for l in leads['sources']:
        if l['url'].endswith('P020250703353232451586.pdf'):
            l.update(status='2026-11-11取得原始PDF并目视核读第1页',detail='完整四至及括号学区代码60均已核读，PDF原件保存在项目内',use='历史正式范围已直接配对；2026延续另核')
    write(ROOT/'来源线索索引.json',leads)
    manifest=read(ROOT/'manifest.json')
    manifest['text_supplements']=[dict(id=RUN,date='2026-11-11',audit=f'{REL}/78校逐校核查.json',summary=f'{REL}/补查汇总.json')]
    write(ROOT/'manifest.json',manifest)
    check=read(ROOT/'校验结果.json');check['sources']=len(archives);check['text_supplement_audit']=f'{REL}/校验.json';write(ROOT/'校验结果.json',check)
    assert before == {p.relative_to(ROOT).as_posix():digest(p) for p in protected}
    write(OUT/'校验.json',dict(passed=True,protected_files_sha256=before,map_text_and_polygon_exactly_preserved=True,
        other_district_data_unchanged=True,school_count=78,counts=counts,source_files_exist=all((ROOT/a['package_path']).exists() for a in archives)))
    if export:
        from export_shenzhen_source_index import main as export_index
        export_index()
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':main()
