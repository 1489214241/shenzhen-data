"""深圳基础划片规则。预测端只消费政策及独立适用事实，禁止读 polygon/Gold。

大学区不做成员扩展；共享只沿有原文依据的范围方向应用，一次执行，不做传递闭包。
模型命中某校触发的共享集合仍是预测，不冒充核验事实或训练标签。
"""
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path


def normalize_name(value):
    # 保留期数、楼栋、括号内容；不做前缀匹配或模糊别名合并。
    return re.sub(r'\s+', '', value).replace('（', '(').replace('）', ')')


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(district):
    from . import shenzhen_sources
    folder = shenzhen_sources.DATA / district
    policy = shenzhen_sources._load(district, '政策规则/规则包.json')
    for relative, digest in policy['input_hashes'].items():
        if source_hash(folder / relative) != digest:
            raise ValueError(f'政策输入已变化，请重新构建深圳规则：{relative}')
    return policy


def apply(policy, site, predicted_school_ids=(), facts=()):
    """facts 可提供人工/道路/行政范围核实结果；必须绑定当前小区和规则、保留证据。

    每条事实：site_id, rule_id, status(satisfied/not_satisfied/unknown),
    evidence_kind, evidence, year, stage；大学区另需 base_school_id。
    预测只用于单向共享扩展，不用于认定原地段或制造确定性排除。
    """
    sid = site['coordinate_group_id']
    names = site.get('coordinate_group_building_names', [site.get('building_name', '')])
    names = {normalize_name(n) for n in names}
    school_ids = set(policy['school_ids'])
    original = set(predicted_school_ids)
    if not original <= school_ids:
        raise ValueError('预测包含本区学校目录之外的学校ID')
    rule_ids = {r['rule_id'] for r in policy['relations']}
    fact_by = {}
    for f in facts:
        if f['site_id'] != sid or f['rule_id'] not in rule_ids:
            raise ValueError('适用事实与小区/规则不对应')
        if f.get('year') != policy['year'] or f.get('stage') != '小学':
            raise ValueError('适用事实年份/学段不对应')
        if f.get('evidence_kind') not in {'official_named_site', 'verified_road_scope', 'verified_admin_scope', 'manual_source_review'} or not f.get('evidence'):
            raise ValueError('适用事实必须有独立政策/身份/地理证据；禁止用Gold或学区面作为输入')
        if f['status'] not in {'satisfied', 'not_satisfied', 'unknown'}:
            raise ValueError('不认识的适用状态')
        if f['rule_id'] in fact_by:
            raise ValueError('同一规则存在重复适用事实')
        fact_by[f['rule_id']] = f

    applied, pending, candidates, exclusions = [], [], [], set()
    grounded, inferred = set(), set()
    university_choices = []
    for r in policy['relations']:
        members = set(r['school_ids'])
        f = fact_by.get(r['rule_id'])
        named = bool(names & {normalize_name(n) for n in r.get('exact_site_names', [])})
        status = f['status'] if f else ('satisfied' if named else 'unknown')
        # 多个不同楼盘共用坐标，不把一个楼盘的身份事实传播给另一个。
        if named and len(names) > 1 and not f:
            status = 'unknown'
        if status == 'not_satisfied':
            continue
        if r['kind'] == 'university':
            if status == 'satisfied' and f and f.get('base_school_id') in members:
                university_choices.append((members, f['base_school_id'], r['rule_id']))
                grounded.add(f['base_school_id'])
                applied.append(dict(rule_id=r['rule_id'], basis='verified_original_base', school_ids=[f['base_school_id']]))
            elif len(original & members) > 1 or status == 'satisfied':
                pending.append(dict(rule_id=r['rule_id'], reason='原地段未唯一确认；不按大学区成员表任选一校'))
            continue
        if r['kind'] in {'eligibility', 'pending'}:
            if original & members or status == 'satisfied':
                pending.append(dict(rule_id=r['rule_id'], reason=r['reason']))
            continue
        if r['kind'] == 'base_scope':
            # 在已核实的单片之外，才能排除该校；unknown 绝不当 outside。
            if status == 'satisfied':
                grounded |= members
                applied.append(dict(rule_id=r['rule_id'], basis='verified_scope', school_ids=sorted(members)))
            continue
        if r['kind'] != 'shared':
            raise ValueError('未知政策规则类型')
        if status == 'satisfied':
            grounded |= members
            applied.append(dict(rule_id=r['rule_id'], basis='exact_policy_name' if named and not f else 'verified_scope', school_ids=sorted(members)))
        elif not f and original & set(r.get('trigger_school_ids', [])):
            # 仅用原始预测触发。新增学校不再次触发别的共享关系。
            inferred |= members
            applied.append(dict(rule_id=r['rule_id'], basis='model_scope_implication', school_ids=sorted(members)))
        elif original & members or any(n and any(n in normalize_name(t) or normalize_name(t) in n for t in r.get('exact_site_names', [])) for n in names):
            candidates.append(dict(rule_id=r['rule_id'], school_ids=sorted(members), status='unknown', reason='缺少本片适用事实；不做整校扩展或前缀别名匹配'))

    # 独立确认的基础范围同样遵守直接包含关系；仅以这一轮的原始事实触发一次。
    # 例如确认行知原范围，也确认同范围深中龙华；不能继续沿别的局部共享传递。
    directly_grounded = set(grounded)
    for r in policy['relations']:
        f = fact_by.get(r['rule_id'])
        if (r['kind'] == 'shared' and directly_grounded & set(r.get('trigger_school_ids', []))
                and (not f or f['status'] == 'satisfied')):
            grounded.update(r['school_ids'])
            applied.append(dict(rule_id=r['rule_id'], basis='verified_scope_implication', school_ids=sorted(r['school_ids'])))
    # 不满足某个唯一单片条款，可以排除该学校；不能从共享局部不满足推成整校排除。
    for r in policy['relations']:
        f = fact_by.get(r['rule_id'])
        if r['kind'] == 'base_scope' and r.get('complete_base_scope') and f and f['status'] == 'not_satisfied':
            exclusions.update(r['school_ids'])
    result = (grounded if grounded else original | inferred) - exclusions
    for members, base, rid in university_choices:
        # 有独立共享事实的学校保留；大学区本身不贡献其他成员。
        result -= (members - {base}) - grounded
        result.add(base)
    conflict = bool(grounded & exclusions)
    for members, base, rid in university_choices:
        if any(other != base and members == mm for mm, other, _ in university_choices):
            conflict = True
    if conflict:
        pending.append(dict(rule_id='conflict', reason='独立适用事实相互冲突，停止自动输出'))
        result = set()
    status = 'policy_supported' if grounded and not conflict else ('model_with_policy' if result else 'unknown')
    return dict(coordinate_group_id=sid, building_name=site.get('building_name'),
                raw_model_school_ids=sorted(original), predicted_school_ids=sorted(result),
                policy_supported_school_ids=sorted(grounded - exclusions) if not conflict else [],
                status=status, applied_rules=applied, pending=pending, candidates=candidates,
                excluded_school_ids=sorted(exclusions), policy_version=policy['version'],
                target='base_catchment_prediction', enrollment_eligibility=False)


def predict_district(district, predictions=None, facts=None):
    """稳定坐标组ID连接。无模型结果时，只输出有独立政策适用依据的集合，其余弃权。"""
    from . import shenzhen_sources
    policy = load_policy(district)
    sites = shenzhen_sources._load(district, '小区数据/坐标组.json')
    predictions, facts = predictions or {}, facts or {}
    known_ids = {s['coordinate_group_id'] for s in sites}
    if (set(predictions) | set(facts)) - known_ids:
        raise ValueError('预测/事实包含未知坐标组，可能使用了旧版索引')
    return [apply(policy, s, predictions.get(s['coordinate_group_id'], []), facts.get(s['coordinate_group_id'], [])) for s in sites]


def annotate_output(dist, rows):
    """正式 predict.answers 的深圳输出入口；按独立ID对齐，拒绝旧数组位置静默映射。"""
    if dist.city != '深圳':
        return rows
    policy = load_policy(dist.name)
    ids = getattr(dist, 'site_ids', None)
    school_ids = getattr(dist, 'school_ids', None)
    if ids is None or school_ids is None:
        raise ValueError('深圳模型适配必须提供 site_ids（坐标组ID）及 school_ids；不能按旧学校下标套规则')
    from . import shenzhen_sources
    sites = {s['coordinate_group_id']: s for s in shenzhen_sources._load(dist.name, '小区数据/坐标组.json')}
    byname = dict(zip(dist.school_names, school_ids))
    if len(byname) != len(school_ids) or set(school_ids) != set(policy['school_ids']):
        raise ValueError('模型学校目录与现用规则版本不一致')
    name_byid = dict(zip(school_ids, dist.school_names))
    out = deepcopy(rows)
    for row in out:
        site = sites[ids[row['q']]]
        if normalize_name(row['name']) not in {normalize_name(n) for n in site['coordinate_group_building_names']}:
            raise ValueError('模型小区名称与坐标组身份不一致')
        decision = apply(policy, site, [byname[n] for n in row['schools']], getattr(dist, 'policy_facts', {}).get(ids[row['q']], []))
        row['schools'] = [name_byid[s] for s in decision['predicted_school_ids']]
        row['policy_decision'] = decision
        if decision['applied_rules']:
            row['source'] += '＋基础划片规则'
    return out


def rule_output(dist):
    """正式规则入口：仅独立政策支持可作为已知参照；模型推断的多校不进入训练。"""
    import numpy as np
    from .rules import RuleOutput
    ids = getattr(dist, 'site_ids', None)
    school_ids = getattr(dist, 'school_ids', None)
    if ids is None or school_ids is None:
        raise ValueError('深圳模型数据必须提供稳定site_ids与school_ids后才可执行规则')
    policy = load_policy(dist.name)
    if set(school_ids) != set(policy['school_ids']):
        raise ValueError('深圳模型学校目录与政策版本不一致')
    decisions = {r['coordinate_group_id']: r for r in predict_district(dist.name, facts=getattr(dist, 'policy_facts', None))}
    known, kind = {}, {}
    neg = np.zeros((dist.n, dist.school_n), dtype=bool)
    byid = {s:i for i,s in enumerate(school_ids)}
    for q, site_id in enumerate(ids):
        result = decisions[site_id]
        if normalize_name(dist.site_names[q]) != normalize_name(result['building_name']):
            raise ValueError('深圳模型小区身份与政策索引不同')
        for s in result['excluded_school_ids']:
            neg[q, byid[s]] = True
        if result['policy_supported_school_ids']:
            known[q] = [byid[s] for s in result['policy_supported_school_ids']]
            kind[q] = 'policy_base'
    return RuleOutput(statuses=np.full((dist.n, dist.school_n, 4), -1, dtype=np.int8), negative=neg, known=known, kind=kind)
