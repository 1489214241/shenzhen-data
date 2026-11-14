"""Readable results and original policy quotes; no model/input edits."""
import json
import experiment_shared_school_prior as E


def collect_quotes(value):
    found=[]
    if isinstance(value,dict):
        if value.get('scope_text') and value.get('source_url'):
            found.append(dict(text=value['scope_text'],url=value['source_url'],
                              position=value.get('source_position'),school=value.get('school_name')))
        if value.get('quote') and isinstance(value.get('source'),dict):
            found.append(dict(text=value['quote'],url=value['source'].get('url'),
                              position=value.get('paragraph_review_id'),school=None))
        for child in value.values():found.extend(collect_quotes(child))
    elif isinstance(value,list):
        for child in value:found.extend(collect_quotes(child))
    unique={json.dumps(x,ensure_ascii=False,sort_keys=True):x for x in found}
    return list(unique.values())


def main():
    c=E.verify();E.check('prediction_freeze.json')
    r=E.read(E.OUT/'results.json');audit=E.read(E.OUT/'output_audit.json');assert audit['passed']
    names={};priors={};ledgers={};policies={}
    for d in c['targets']:
        meta=E.read(E.INPUT/d/'metadata.json')
        names.update({s['school_id']:s['school_name'] for s in meta['schools']})
        priors.update({(d,x['site_id']):x for x in E.read(E.OUT/'priors'/f'{d}.json')})
        ledgers[d]=E.read(E.INPUT/d/'evidence_ledger.json')
        policies[d]=E.read(E.INPUT/d/'reviewed_policy.json')
    baseline={(x['d'],x['site_id']):x for x in E.read(E.OUT/'scored_rows/baseline_current.json')}
    group=E.read(E.OUT/'scored_rows/text_group_exact.json')
    labels={'baseline_native':'原始 Modern＋政策规则','baseline_current':'最近 Modern＋补校模型＋政策规则',
        'text_minimum':'文字多校组作为至少保留项（预定主比较）',
        'text_group_exact':'文字多校组作为全部输出（完整性假设）',
        'text_count_admission':'只给文字组校数，补校模型排序选取',
        'text_count_native':'只给文字组校数，原生 Modern 排序选取',
        'estimated_minimum':'加上模型估计适用组，至少保留',
        'estimated_group_exact':'加上模型估计适用组，作为全部输出',
        'estimated_count_admission':'加上估计校数，补校模型排序选取',
        'estimated_count_native':'加上估计校数，原生 Modern 排序选取'}
    lines=['共享学区提前输入实验（2026-11-14）','',
        '结论：共享学区可按多校关系输入，不限两校。只给数量不足以解决选校，且已知关系不一定穷尽全部学校。',
        '本轮实际做的是提前构造多校条件，约束冻结 Modern 的最终集合选择；未重训网络，也不是大语言模型提示词实验。',
        '旧流程已在后处理加入这些关系：改成“至少保留这些学校”时，1761条计分输出逐条完全相同。',
        '把有文字依据的成员组假设为完整输出，912→915（修正6、误伤3）。提升0.1704个百分点，空间95%区间[-0.1726,0.5764]，未确认稳定提升。',
        '不据此修改正式模型默认值。本轮是看过此前结果后的开发实验，不是独立验证。','',
        '固定比较条件：1761个地图命中坐标组；41个未命中组仅保留输出；242个语义待核组仍计分。',
        '学校集合必须完全一致才算对；保留FP/FN和学校级precision/recall；原Gold文件哈希不变。',
        '三次种子是此前补校头初始化；原生Modern三种子均值固定。本轮不新增训练，不选阈值、不按深圳成绩挑参数。','',
        '实验臂 | 全对/1761 | 准确率 | 单校全对/803 | 多校全对/958 | 龙华/627 | 龙岗/1134 | 修正/误伤（对最近结果）']
    for arm,m in r['metrics'].items():
        comp=r['comparisons'].get(arm,{}).get('all',{})
        lines.append(f"{labels[arm]} | {m['all']['correct']} | {m['all']['exact']:.2f}% | {m['single']['correct']} | {m['multi']['correct']} | {m['龙华区']['correct']} | {m['龙岗区']['correct']} | {comp.get('corrected','—')}/{comp.get('harmed','—')}")
    lines.extend(['','证据覆盖与完整性：',
        '龙华26条、龙岗79条（合计105）可提前提供至少两校的文字及独立归属证据；不是全体958条多校Gold均提前知道校数。',
        '另有397条仅能由原生模型首选学校触发共享范围推测；单列估计实验，不称作官方确认的小区多校事实。',
        '105条中，与固定地图答案完全同组67条；有23条只是正确学校集合的一部分；15条至少存在一个文字/归属输入与地图标签的冲突。',
        '输入校数恰与地图校数相等75条，但其中仍有学校身份不一致；校数相同不保证选校正确。',
        '这些是对冻结输入的事后审计，未将地图关系回填输入。冲突原因须后续逐条审查，不能据此断言官网错了。','',
        '仅诊断：如果事后从评分答案告诉模型真实校数（测试信息，不可用于实际效果宣传）：',
        f"冻结原生排序取真实前k：{r['oracle_count_diagnostic_only']['native']['all']['correct']}/1761；冻结补校排序取真实前k：{r['oracle_count_diagnostic_only']['admission']['all']['correct']}/1761。",
        '这两项未经过完整旧政策后处理，只诊断固定排序；不能与完整系统直接作同输入比较，也不是重训后的理论上限。',
        '即便校数完全正确，当前冻结排序仍选不齐很多学校。因此仍需改善成员识别，而不是只强制输出数量。','',
        '关键案例（下文“地图”只指固定评分标签，不等于已核实的家庭入学资格）：'])
    changed=[]
    for row in group:
        key=row['d'],row['site_id'];old=baseline[key]
        if row['prediction']==old['prediction']:continue
        prior=priors[key]
        direct=[x for x in ledgers[row['d']] if x['site_id']==row['site_id'] and x['positive'] and x['routes']['hard_constraint']]
        ruleids={x['rule_id'] for x in prior['proofs']}
        rules=[x for x in policies[row['d']]['relations'] if x['rule_id'] in ruleids]
        quotes=collect_quotes([direct,rules])
        change=dict(district=row['d'],site_id=row['site_id'],name=row['name'],
            before=[names[s] for s in old['prediction']],after=[names[s] for s in row['prediction']],
            map_scoring_set=[names[s] for s in row['truth']],before_correct=old['correct'],after_correct=row['correct'],
            semantic_pending=row['semantic_pending'],source_quotes=quotes,input_proofs=direct,shared_rules=rules)
        changed.append(change)
        if row['correct']==old['correct']:continue
        lines.extend(['',f"{row['d']}｜{row['name']}｜{'修正' if row['correct'] else '误伤'}｜语义待核={row['semantic_pending']}",
            '原输出：'+'；'.join(change['before']),'本次输出：'+'；'.join(change['after']),
            '地图评分集合：'+'；'.join(change['map_scoring_set'])])
        if not row['correct']:
            lines.append('机制：强制把已知学校组当作全部答案，删除了原输出中仍被地图评分要求的学校。')
        for q in quotes:
            lines.extend([f"原文 [{q['school'] or '小区行政归属'}，{q['position']}]：{q['text']}",f"来源：{q['url']}"])
    lines.extend(['','工程核验：5项针对性测试通过；输出复算27322条通过；现用数据指针、输入、代码及预测均已锁定哈希。',
        '准备阶段曾误用历史目录核对源文件，已在预测生成及评分前修复；原代码与修复记录保留，不影响以上结果。',
        '复现：依次执行 scripts/experiment_shared_school_prior.py 的 prepare、infer、score、audit（输出目录已存在时拒绝覆盖）。',
        f'实验目录：{E.OUT}',f'预测：{E.OUT / "predictions"}',f'原文与逐条变化：{E.DOC / "逐条变化及政策原文_仅诊断.json"}',
        '下一步应区分“已知至少对应这些学校”与“已经确认这是完整集合”。完整集合才用于排除额外学校；局部已知关系仍需模型补齐。'])
    E.DOC.mkdir(parents=True,exist_ok=True)
    (E.DOC/'实验结果与政策原文.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    E.write(E.DOC/'逐条变化及政策原文_仅诊断.json',dict(changed_predictions=changed,diagnostic_only=True,never_training_input=True))
    E.write(E.DOC/'实验结果.json',r)
    E.write(E.DOC/'实际使用入口.json',dict(script=str(E.REPO/'scripts/experiment_shared_school_prior.py'),
        prior_logic=str(E.REPO/'zoning/shared_school_prior.py'),output=str(E.OUT),
        protocol=str(E.OUT/'protocol.json'),audit=str(E.OUT/'output_audit.json'),adopted=False))
    E.write(E.DOC/'交付校验.json',{str(p):E.sha(p) for p in E.DOC.iterdir() if p.is_file() and p.name!='交付校验.json'})
    print('REPORT WRITTEN',E.DOC,flush=True)


if __name__=='__main__':main()
