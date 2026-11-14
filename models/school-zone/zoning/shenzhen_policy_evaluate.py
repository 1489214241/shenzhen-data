"""深圳规则评价：只在预测完成后读取原图标签，标签不由政策输出反写。

保留全量原图评分与语义待核清单；未核准的基础标签不输出“基础学区准确率”。
"""
def audit_gold(policy, gold):
    inventory = {s['school_id']: s for s in policy['school_scope_inventory']}
    university = [set(r['school_ids']) for r in policy['relations'] if r['kind'] == 'university']
    mixed = {s for r in policy['relations'] if r['kind'] == 'base_scope' for s in r['school_ids']}
    result = []
    for g in gold:
        labels = set(g['point_school_ids'])
        reasons = []
        if labels & mixed:
            reasons.append('原图可能含派位/联动范围，缺独立基础边界审定')
        if any(len(labels & u) > 1 for u in university):
            reasons.append('大学区原图多命中，不能凭成员身份任选一个原地段')
        for s in sorted(labels):
            if inventory[s]['status'].endswith('_pending'):
                reasons.append(inventory[s]['status'] + ':' + s)
        result.append(dict(coordinate_group_id=g['coordinate_group_id'], building_name=g['building_name'],
                           original_map_school_ids=sorted(labels), unresolved_reasons=reasons,
                           base_school_ids=None, base_label_status='pending' if reasons else 'map_only_not_independently_certified',
                           original_map_label_preserved=True))
    return result


def score(predictions, gold, audit):
    pred = {p['coordinate_group_id']: p for p in predictions}
    if len(pred) != len(predictions):
        raise ValueError('重复预测坐标组')
    pending = {r['coordinate_group_id'] for r in audit if r['unresolved_reasons']}
    rows = []
    for g in gold:
        key = g['coordinate_group_id']
        if key not in pred:
            raise ValueError('缺少评价小区预测；不得静默缩减分母')
        p = pred[key]; truth = set(g['point_school_ids']); before = set(p['raw_model_school_ids']); after = set(p['predicted_school_ids'])
        rows.append(dict(coordinate_group_id=key, before_correct=before == truth, after_correct=after == truth,
                         missing_school_ids=sorted(truth-after), wrong_school_ids=sorted(after-truth),
                         wrong_added_school_ids=sorted((after-before)-truth),
                         abstained=not after, semantic_pending=key in pending))
    n = len(rows)
    return dict(target='official_map_point_membership', n=n,
                correct=sum(r['after_correct'] for r in rows), exact_set_accuracy=sum(r['after_correct'] for r in rows)/n if n else None,
                before_correct=sum(r['before_correct'] for r in rows),
                corrected=sum(not r['before_correct'] and r['after_correct'] for r in rows),
                harmed=sum(r['before_correct'] and not r['after_correct'] for r in rows),
                wrongly_expanded_sites=sum(bool(r['wrong_added_school_ids']) for r in rows),
                abstained=sum(r['abstained'] for r in rows), semantic_pending=len(pending),
                base_scope_accuracy=None, base_scope_accuracy_reason='基础标签尚未全量独立核准；不能把原图分数当作基础学区分数',
                rows=rows)


def score_final_output(dist, rows):
    """主流程最终输出计分；不拿argmax旧分数替代规则后的完整集合。"""
    from . import shenzhen_sources, shenzhen_policy
    if dist.city != '深圳':
        raise ValueError('此评价入口只适用于深圳现用数据')
    decisions = [row['policy_decision'] for row in rows]
    sites = shenzhen_sources._load(dist.name, '小区数据/坐标组.json')
    if {r['coordinate_group_id'] for r in decisions} != {r['coordinate_group_id'] for r in sites}:
        raise ValueError('主流程深圳输出未覆盖完整候选集，不能静默缩减评价分母')
    policy = shenzhen_policy.load_policy(dist.name)
    gold = shenzhen_sources.load_point_gold(dist.name)
    return score(decisions, gold, audit_gold(policy, gold))
