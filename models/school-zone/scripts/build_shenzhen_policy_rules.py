"""从已配对的2026官方文字编译基础划片规则；不读polygon或Gold。"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zoning import shenzhen_policy

ROOT = REPO / 'data/深圳市'
VERSION = '20261111_base_scope_rules_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def norm(t):
    # 只规范等价分隔符，不把“甲、乙”和名为“甲乙”的地域合并。
    return re.sub(r'\s+', '', t).replace('，', '、').replace(',', '、').rstrip('。')


def compile_policy(district, texts, input_hashes):
    # 完整官方文本始终保留；只对下面逐条核实的章节标题和办学地址注记做结构化拆分。
    school_ids = [s['school_id'] for s in texts]
    lookup = {s['school_id'].split(':school:')[1]: s for s in texts}
    relations, inventory = [], []

    def sid(short):
        return lookup[short]['school_id']

    def evidence(ids):
        return [dict(school_id=s['school_id'], school_name=s['school_name'],
                     **{k:t[k] for k in ['scope_text', 'source_url', 'source_position', 'year', 'document_local_path']})
                for s in texts if s['school_id'] in ids for t in s['independent_texts'] if t['year'] == 2026]

    def add(rid, kind, ids, reason, triggers=(), names=(), **extra):
        ids = sorted(set(ids))
        assert set(ids) <= set(school_ids)
        relations.append(dict(rule_id=rid, kind=kind, school_ids=ids, trigger_school_ids=sorted(set(triggers)),
                              exact_site_names=list(names), reason=reason, evidence=evidence(ids), **extra))

    bases = {}
    for s in texts:
        t = next(t for t in s['independent_texts'] if t['year'] == 2026)
        raw = t['scope_text']; base = raw; extras = []
        status = 'individual_scope_requires_site_applicability'
        if '多校联动学区：' in base:
            base, tail = base.split('多校联动学区：', 1)
            base = base.replace('单校划片学区：', '').strip()
            extras.append(dict(kind='linked_application', quote='多校联动学区：' + tail))
        if '世纪春城小区属于大学区范围。' in base:
            base = base.replace('世纪春城小区属于大学区范围。', '')
            extras.append(dict(kind='university_application', quote='世纪春城小区属于大学区范围。'))
        if '单片学区：' in base:
            base = base.removeprefix('单片学区：')
        # 这些注记仅涉及办学地址/初中，保留在原文但不当作小学基础范围差异。
        for annotation in ['（借址新围学校，项目名）', '（暂在初中部办学）', '（小一借址原观澜党校）',
                           '（初中部2026 年仍面向全民治街道进行招生，2027 年起按照划定范围招收新生。）']:
            base = base.replace(annotation, '')
        if '其中，' in base and '优享学区' in base:
            base, tail = base.split('其中，', 1)
            extras.append(dict(kind='priority_only', quote='其中，' + tail))
        if '面向' in base:
            status = 'broad_scope_pending'
        if '深户' in base:
            status = 'population_condition_pending'
        if '一体化办学' in base:
            status = 'integrated_school_pending'
        bases[s['school_id']] = base.strip()
        inventory.append(dict(school_id=s['school_id'], school_name=s['school_name'],
                              base_scope_text=base.strip(), original_scope_text=raw,
                              extra_clauses=extras, status=status, evidence=evidence([s['school_id']])))

    # 同文的整个基础范围可以双向预测；不把整街道招生、户籍条件、统筹开班自动当共享。
    by_scope = defaultdict(list)
    for s in inventory:
        if s['status'] == 'individual_scope_requires_site_applicability':
            by_scope[norm(s['base_scope_text'])].append(s['school_id'])
    for key, ids in sorted(by_scope.items()):
        if len(ids) > 1:
            h = hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest()[:10]
            add('EQ-' + h, 'shared', ids, '官方年度表完整基础范围一致；只表达范围集合，不表示自由择校或录取保证', triggers=ids)

    if district == '龙华区':
        university = [sid(x) for x in ['496FFFE16B755558E053BC09A8C0A5ED', '42392d26506d455da3eab13483999166',
                       '496FFFE16B945558E053BC09A8C0A5ED', '496FFFE16B955558E053BC09A8C0A5ED', 'a863a2a0d9434c7c8370f9af437bae6e']]
        add('LH-UNIVERSITY', 'university', university, '大学区保留原地段单校，不按五校成员扩展；教科院附实仅世纪春城参加',
            official_scheme='https://www.szlhq.gov.cn/attachment/1/1739/1739845/12826915.docx',
            participation_limits={sid('a863a2a0d9434c7c8370f9af437bae6e'): '仅世纪春城小区；不同分期身份须核实'})
        east = sid('94aed576e2854236bfd117b281031afa')
        linked = sid('dd655370e05640259a918bdfb9146830')
        for rid, target, why in [('LH-EAST-BASE', east, '原图含电脑派位范围；只有单片四至成立才确认基础归属'),
                                  ('LH-LINK-BASE', linked, '区分单校划片与多校联动；只执行单校划片范围')]:
            add(rid, 'base_scope', [target], why, complete_base_scope=True, scope_text=bases[target])
        add('LH-LOTTERY', 'eligibility', [east], '电脑随机派位资格不扩大基础归属；不得整校删除，也不得保留为已核准的基础面',
            official_scheme='https://www.szlhq.gov.cn/attachment/1/1720/1720517/12826915.docx')
        add('LH-LINK', 'eligibility', [linked], '多校联动条款单列，不能当作基础范围')
    else:
        # LG01 经原文“优享”拆开后已由同基础范围分组；优享是加分层，不互斥单校化。
        # 单向包含以明确原文为依据，不从polygon相交或“共享”词语传播。
        local = [
            ('LG02', ['496FFFE16D2B5558E053BC09A8C0A5ED','496FFFE16CEC5558E053BC09A8C0A5ED','4de8828a16684f23a0f86a534cfa2549'],
             ['496FFFE16D2B5558E053BC09A8C0A5ED'], ['万科麓城'], '只在麓城小学自身范围内对应三校；不向三联其他片区扩展'),
            ('LG03', ['570454dc28bc4fec90c5149c2a12ea37','496FFFE16CE95558E053BC09A8C0A5ED'],
             ['570454dc28bc4fec90c5149c2a12ea37'], [], '只共享悦澜山原文道路围合范围，不把兰著全部地段加给悦澜山'),
            ('LG04', ['44d24aa4278e4c4fb9a2ff16e8c2761f','cecf915b0a7d4b038eb60bdd49a068de','a241a4bce82940c79906fbca2d362f50'],
             ['44d24aa4278e4c4fb9a2ff16e8c2761f'], ['珠江旭景佳园','富豪花园','十二橡树庄园','和成世纪','德润荣君府','大族云峰','荷棠里'],
             '只共享和美自身四至与明确住宅，不把马安堂扩给和美'),
            ('LG05', ['58c59cc2ba13405a9372d2096e8db34c','496FFFE16CF65558E053BC09A8C0A5ED'], [],
             ['和谐家园','银领公馆','特发香阅四季','阅景花园','尚峰花园','健康花城'], '南湾实验第一片共享；不传递到第二片或石芽岭其他范围；不自动合并健康花城与岭宏健康家园'),
            ('LG06', ['58c59cc2ba13405a9372d2096e8db34c','496FFFE16CFC5558E053BC09A8C0A5ED','496FFFE16C9D5558E053BC09A8C0A5ED'], [],
             ['和通花园','中兆花园','泰雅园','龙泉别墅'], '南湾实验第二片共享；桂芳园1—6期另需准确分期，不做名称前缀合并'),
            ('LG07', ['496FFFE16CDE5558E053BC09A8C0A5ED','496FFFE16D0D5558E053BC09A8C0A5ED'], [],
             ['远洋新干线'], '只对明确点名楼盘共享；一期二期和商业广场不凭前缀自动合并'),
            ('LG08', ['ee3005e048ca42bfaa228e2927283f96','496FFFE16CEF5558E053BC09A8C0A5ED','496FFFE16D1F5558E053BC09A8C0A5ED'],
             ['496FFFE16CEF5558E053BC09A8C0A5ED','496FFFE16D1F5558E053BC09A8C0A5ED'], [], '文理包含吉祥/文景范围；反向不扩到文理新增片区'),
            ('LG09-A', ['02f11df45da0485a81366d7754c1c849','b4f7994665414818a2b73f1cd1e552b9'],
             ['b4f7994665414818a2b73f1cd1e552b9'], [], '深外龙岗包含平湖二实范围；不把深外所有范围倒灌给平湖二实'),
            ('LG09-B', ['02f11df45da0485a81366d7754c1c849','496FFFE16D1A5558E053BC09A8C0A5ED'],
             ['496FFFE16D1A5558E053BC09A8C0A5ED'], ['顺明园'], '深外龙岗包含辅城坳范围；借址不等于和二实互换全部范围'),
        ]
        for rid, members, triggers, names, why in local:
            ids = [sid(x) for x in members]
            all_text = '\n'.join(e['scope_text'] for e in evidence(ids))
            assert all(name in all_text for name in names), rid
            add(rid, 'shared', ids, why, [sid(x) for x in triggers], names, scope_completeness='local_only')

    for s in inventory:
        if s['status'].endswith('_pending'):
            add('PENDING-' + s['school_id'], 'pending', [s['school_id']], s['status'] + '；须核实适用条件，不自动多校化或单校化')
        # 只接纳完全无道路、条件或括注的明确名单。复杂文字留给道路/行政观察，不用字符串命中硬判。
        base = s['base_scope_text']
        if s['status'] == 'individual_scope_requires_site_applicability' and not re.search(r'[以除不()（）\[\]：:；;]|社区|街道|范围|区域|招生|其中|优享|共享|学区|居民|居委|统筹|借址', base):
            names = [t.strip().strip('。') for t in re.split('[、，,。\n]', base) if t.strip().strip('。')]
            if names:
                add('NAMED-' + s['school_id'], 'shared', [s['school_id']], '年度原表明确无条件住宅名单；仅完整名称匹配', names=names)

    # 精确名称同时获得的基础条款可求集合；并非通过学校关系传递得到。
    return dict(schema='shenzhen_policy_rules_v1', version=VERSION, district=district, year=2026, stage='小学',
                target='base_catchment_prediction', school_ids=school_ids, input_hashes=input_hashes,
                relations=relations, school_scope_inventory=inventory,
                contract=dict(university_expansion=False, shared_requires_scope=True, transitive_closure=False,
                              fuzzy_site_matching=False, predictions_are_training_facts=False,
                              reads_gold=False, reads_school_polygons=False, unresolved_is_negative=False))


def main():
    current = read(ROOT / '当前数据集.json'); base = ROOT / current['dataset_path']
    summary = []
    for d in current['active_districts']:
        folder = base / d
        inputs = {p: shenzhen_policy.source_hash(folder / p) for p in ['texts.json','schools.json','小区数据/坐标组.json']}
        package = compile_policy(d, read(folder / 'texts.json'), inputs)
        for s in read(folder / 'texts.json'):
            for t in s['independent_texts']:
                relative = '../' + t['document_local_path']
                package['input_hashes'][relative] = shenzhen_policy.source_hash(folder/relative)
        package['build_code_sha256'] = {str(p.relative_to(REPO)):shenzhen_policy.source_hash(p) for p in
                                        [Path(__file__), REPO/'zoning/shenzhen_policy.py']}
        if d == '龙华区':
            documents = REPO.parent/'深圳市/四区数据集_20261103_v1/数据清单/三校以上政策核验_20261103/官方资料'
            for rid, name in [('LH-UNIVERSITY','longhua_2026_notice_1739845_12826915'),
                              ('LH-LOTTERY','longhua_2026_notice_1720517_12826915')]:
                relation = next(r for r in package['relations'] if r['rule_id'] == rid)
                for suffix in ['.txt', '.docx']:
                    original = documents/(name+suffix)
                    target = folder/'政策规则/官方附件'/(name+suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, target)
                    relative = target.relative_to(folder).as_posix()
                    package['input_hashes'][relative] = shenzhen_policy.source_hash(target)
                    if suffix == '.txt':
                        source = original.read_text(encoding='utf-8-sig')
                        relation['scheme_quotes'] = [line for line in source.splitlines() if any(word in line for word in
                            ['单校划片范围', '小一选取', '电脑随机派位学区：', '单片学区：', '所属原学区范围学校志愿'])]
                        relation['scheme_text_path'] = relative
        write(folder / '政策规则/规则包.json', package)
        sites = read(folder / '小区数据/坐标组.json')
        decisions = [shenzhen_policy.apply(package, s) for s in sites]
        write(folder / '政策规则/逐小区规则结果.json', decisions)
        write(folder / '政策规则/逐校基础范围及资格条款.json', package['school_scope_inventory'])
        summary.append(dict(district=d, schools=len(package['school_ids']), relations=len(package['relations']),
                            relation_types=dict(Counter(r['kind'] for r in package['relations'])),
                            sites=len(sites), policy_supported_sites=sum(bool(r['policy_supported_school_ids']) for r in decisions),
                            policy_multischool_sites=sum(len(r['policy_supported_school_ids']) > 1 for r in decisions),
                            model_predictions_run=False))
        listing = read(folder / '实验输入/输入清单.json')
        listing['base_scope_policy_rules'] = '../政策规则/规则包.json'
        listing['policy_rule_decisions'] = '../政策规则/逐小区规则结果.json'
        listing['policy_rule_engine_ready'] = True
        listing['spatial_clause_observations_complete'] = False
        write(folder / '实验输入/输入清单.json', listing)
    write(base / '基础划片规则执行汇总.json', dict(version=VERSION, districts=summary, raw_gold_changed=False,
          model_training_rerun=False, pending='缺少道路/行政范围或分期身份的条款保持unknown；模型推断的共享集合不作为核验训练标签'))
    current['base_scope_rules'] = dict(version=VERSION, enabled=True, summary='标准数据_20261111/基础划片规则执行汇总.json')
    write(ROOT / '当前数据集.json', current)
    note = ('基础划片规则已启用（2026-11-11）\n'
            '大学区保留原地段；共享只在已确认适用范围输出多校；优享区保留加分属性；派位/联动仅作资格。\n'
            '规则包：各区/政策规则/规则包.json，逐小区实际执行：各区/政策规则/逐小区规则结果.json。\n'
            '输入只读年度官方文字及小区身份，不读polygon/Gold。模型预测触发的共享集合不作为核验训练标签。\n'
            '同一学校只共享局部时，禁止反向或传递闭包扩展；大学区原地段未确认不强行选校。\n'
            '运行：python -X utf8 scripts/run_shenzhen_policy.py --output outputs/新目录\n'
            '加冻结模型：--predictions 文件.json；格式 {区名:{坐标组ID:[学校ID]}}，须覆盖两区所有候选。\n'
            '每次结果写入新目录，先冻结预测再评价。原图标签保留；基础语义未核准的情况单列，不伪造基础准确率。\n'
            '独立事实：--facts 文件.json；每条必须绑定site_id、rule_id、year=2026、stage=小学、status及来源evidence。\n'
            '政策精确点名直接执行；道路/社区/分期缺少独立对应时保持unknown。未完成深圳特征适配与训练重跑。\n')
    (base/'基础划片规则使用说明.txt').write_text(note, encoding='utf-8')
    page = base/'index.html'
    html = page.read_text(encoding='utf-8')
    if '基础划片规则使用说明.txt' not in html:
        html = html.replace('</h1>', '</h1><p><a href="基础划片规则使用说明.txt">大学区、共享学区规则（已执行）</a> · <a href="基础划片规则执行汇总.json">执行汇总</a> · <a href="龙华区/政策规则/规则包.json">龙华规则与原文</a> · <a href="龙岗区/政策规则/规则包.json">龙岗规则与原文</a></p>', 1)
        page.write_text(html, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
