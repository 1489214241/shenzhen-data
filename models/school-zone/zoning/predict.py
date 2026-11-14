"""第六步：规则集合/模型最高概率；长沙核验通过的小区再应用政策集合。"""
import csv
import json
import numpy as np
from . import changsha_sets, shenzhen_policy


def answers(dist, rules, P):
    rows = []
    for q in range(dist.n):
        if q in rules.known:
            rows.append(dict(q=q, name=dist.site_names[q], source='核验参照' if rules.kind[q] == 'verified' else '规则判出（' + {'named':'名单', 'policy_base':'基础划片政策'}.get(rules.kind[q], '四至') + '）',
                             schools=[dist.school_names[s] for s in rules.known[q]], confidence=''))
        else:
            s = int(P[q].argmax()); rows.append(dict(q=q, name=dist.site_names[q], source='模型', schools=[dist.school_names[s]], confidence=f'{P[q, s]:.3f}'))
    rows = changsha_sets.annotate_output(dist, rules, rows)
    return shenzhen_policy.annotate_output(dist, rows)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        has_sets = any('set_basis' in r for r in rows)
        has_policy = any('policy_decision' in r for r in rows)
        w = csv.writer(f); w.writerow(['小区编号', '小区', '来源', '学校', '把握'] + (['多校依据', '任选资格', '适用范围', '政策年份'] if has_sets else []) + (['基础划片规则审计'] if has_policy else []))
        for r in rows:
            w.writerow([r['q'], r['name'], r['source'], '、'.join(r['schools']), r['confidence']] +
                       ([r.get('set_basis', ''), r.get('choice_status', ''), r.get('policy_applicability', ''), r.get('policy_year', '')] if has_sets else []) +
                       ([json.dumps(r.get('policy_decision', {}), ensure_ascii=False)] if has_policy else []))
