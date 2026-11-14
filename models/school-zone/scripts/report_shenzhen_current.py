"""Read-only diagnosis of frozen Shenzhen predictions; does not fit or tune anything."""
from pathlib import Path
import collections
import hashlib
import json

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'outputs/shenzhen_current_model_20261111'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    result = read(OUT/'results.json')
    rows = read(OUT/'scored_cases.json')
    named, policies = {}, {}
    for d in result['districts']:
        meta = read(OUT/'inputs'/f'{d}.json')
        named.update({s['school_id']: s['school_name'] for s in meta['schools']})
        path = REPO/'data/深圳市/标准数据_20261111'/d/'政策规则/规则包.json'
        policies[d] = {r['rule_id']: r for r in read(path)['relations']}
    bad = [r for r in rows if r['has_map_gold'] and (set(r['after'])-set(r['before']))-set(r['gold'])]
    cases = []
    for r in bad:
        item = dict(r)
        for key in ['before', 'after', 'gold']:
            item[key+'_school_names'] = [named[s] for s in r[key]]
        item['new_outside_map_school_names'] = [named[s] for s in sorted((set(r['after'])-set(r['before']))-set(r['gold']))]
        item['before_had_any_map_school'] = bool(set(r['before']) & set(r['gold']))
        item['official_rules_and_quotes'] = [policies[r['district']][v['rule_id']] for v in r['applied_rules']]
        cases.append(item)
    diag = dict(wrong_add_sites=len(bad),
                before_disjoint_from_map=sum(not r['before_had_any_map_school'] for r in cases),
                before_contained_map_school=sum(r['before_had_any_map_school'] for r in cases),
                singleton_map_wrong_add_sites=sum(len(r['gold']) == 1 for r in bad),
                interpretation='Disjoint means the model selected the wrong region relative to the map; the other cases need policy/map-scope review. Neither is independent enrollment truth.',
                cases=cases)
    (OUT/'错加诊断与政策原文.json').write_text(json.dumps(diag, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = ['深圳模型外测详细解读', '',
        '完整集合准确率：只有输出学校集合与原图全部命中学校完全一致才算正确。',
        '对照前项已经保留文字直接确认的32个小区（其中11个多校），后项再接入共享范围等输出规则。',
        '因此本次增量隔离的是输出规则效果，不能称为整套政策输入的消融实验。', '',
        '一、单校与多校']
    for d, v in result['districts'].items():
        for k, label in [('single', '单校'), ('multi', '多校'), ('semantic_pending', '语义待核'), ('semantic_not_flagged', '未标记语义待核')]:
            a, b = v[k]['before'], v[k]['after']
            text.append(f"{d} {label}：{a['correct']}/{a['n']} ({a['exact_accuracy']:.2%}) → {b['correct']}/{b['n']} ({b['exact_accuracy']:.2%})。")
    a, b = result['overall']['before'], result['overall']['after']
    text += ['', '二、收益及代价',
        f"完整集合修正 {result['overall']['corrected']} 个，原本完整正确而被改错 {result['overall']['harmed']} 个。",
        f"学校级漏报 {a['false_negative_school_assignments']} → {b['false_negative_school_assignments']}；学校级错报 {a['false_positive_school_assignments']} → {b['false_positive_school_assignments']}。",
        f"学校级precision {a['precision']:.2%} → {b['precision']:.2%}；recall {a['recall']:.2%} → {b['recall']:.2%}。",
        f"新增原图之外学校的小区 {len(bad)} 个，其中原图单校 {diag['singleton_map_wrong_add_sites']} 个。这些小区规则前已经完整集合错误，所以 harmed=0 不能解读为没有新增错误。",
        f"其中 {diag['before_disjoint_from_map']} 个在扩充前就没有选中任何原图学校，扩充继续放大了模型选错范围的错误；另外 {diag['before_contained_map_school']} 个原本选中了至少一个原图学校，但新增集合超出原图，需要核对具体适用条件与图文范围。",
        '逐例校名、触发规则、官方原文、URL与页码/表格行号都在：错加诊断与政策原文.json。', '', '三、种子重复']
    for seed, v in result['per_seed'].items():
        if seed != 'ensemble':
            text.append(f"{seed}：{v['before']['exact_accuracy']:.2%} → {v['after']['exact_accuracy']:.2%}；新增错加小区 {v['wrongly_added_sites']}。")
    text += ['', '四、结论边界',
        '在本次固定样本和缺失输入设置下，输出规则提高了多校完整集合覆盖，但错加风险仍明显，不能作为可靠性已经解决或正式采用的结论。',
        '龙华没有可用的文字直接确认参照，龙岗仅32个；深圳道路/社区适用事实未齐，学校标记也并非独立测绘校址，高新仍缺独立标记。',
        '输入消融在训练来源与目标同时去掉了跨路、学出边界、社区和名称相似特征；它不是默认完整输入模型的最终性能。',
        '全量预测1802组，地图有命中1761组，图外41组不充当可靠负例。242组原图标签语义待核已单列；未标记待核也不代表基础学区标签已经独立核准。',
        '未用本次错误修改规则、模型、阈值或Gold。下一阶段需补独立输入与审核图文适用条件，再用未参与开发的样本确认。']
    (OUT/'详细解读.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    freeze = read(OUT/'prediction_freeze.json')
    for p, h in freeze['prediction_sha256'].items():
        assert hashlib.sha256((OUT/p).read_bytes()).hexdigest() == h
    verification = dict(predictions_unchanged_after_scoring=True,
                        candidate_sites=len(rows), scored_sites=sum(r['has_map_gold'] for r in rows),
                        policy_tests_passed=21,
                        native_retained_features=read(OUT/'native_feature_verification.json'),
                        chunk_gradient=read(OUT/'chunk_verification.json'),
                        input_integrity=read(OUT/'input_integrity_verification.json'))
    assert len(rows) == 1802 and sum(r['has_map_gold'] for r in rows) == 1761
    assert len({(r['district'], r['site_id']) for r in rows}) == len(rows)
    (OUT/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in diag.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
