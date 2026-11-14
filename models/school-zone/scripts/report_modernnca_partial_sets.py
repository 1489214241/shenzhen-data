"""Post-score partial-set report. Never imported by fitting or selection."""
import json
from pathlib import Path

import numpy as np
import experiment_modernnca_partial_sets as P


def main():
    P.DOC = P.E.REPO.parent / '科研推进/ModernNCA部分集合学习_20261113'
    P.install()
    E = P.E
    r = E.read(P.OUT / 'results.json')
    audit = E.read(P.OUT / 'audit.json')
    assert audit['passed']
    assert 'partial_label_compatibility_independently_checked' in audit
    old_result = E.read(P.BASE / 'results.json')
    assert old_result['gold_sha256'] == r['gold_sha256']
    rows = {a: E.read(P.OUT / 'scored_rows' / f'{a}.json') for a in E.ARMS}
    rows['source_only_set_readout'] = E.read(P.BASE / 'scored_rows/learned_set_readout.json')
    names = {
        'categorical_policy': '原Modern加现有政策',
        'untrained_set_readout': '未训练集合头',
        'source_only_set_readout': '上一轮仅完整源标签训练',
        'learned_set_readout': '本轮加入深圳部分正关系训练',
    }
    groups = {
        '全部': lambda x: True,
        '龙华区': lambda x: x['d'] == '龙华区',
        '龙岗区': lambda x: x['d'] == '龙岗区',
        '单校': lambda x: x['size'] == 1,
        '多校': lambda x: x['size'] > 1,
        '两校': lambda x: x['size'] == 2,
        '三校及以上': lambda x: x['size'] >= 3,
        '语义待核': lambda x: x['semantic_pending'],
        '其余': lambda x: not x['semantic_pending'],
    }
    metrics = {a: {g: E.M.metric([x for x in rr if f(x)]) for g, f in groups.items()} for a, rr in rows.items()}
    # Align by identities, then verify both population and labels before pairing.
    baseline = {(x['d'], x['site_id']): x for x in rows['source_only_set_readout']}
    current = rows['learned_set_readout']
    assert len(baseline) == len(current) == 1761
    assert {(x['d'], x['site_id']) for x in current} == set(baseline)
    for x in current:
        b = baseline[x['d'], x['site_id']]
        assert x['truth'] == b['truth'] and x['block'] == b['block']
    comparison = {}
    for g, f in groups.items():
        rr = [x for x in current if f(x)]
        delta = [int(x['correct']) - int(baseline[x['d'], x['site_id']]['correct']) for x in rr]
        comparison[g] = dict(delta_pp=100 * float(np.mean(delta)), corrected=delta.count(1),
                             harmed=delta.count(-1), ci95=E.M.evaluate.bootstrap(rr, delta, 20261104, 3000))
    coverage = {}
    for g, f in groups.items():
        rr = [x for x in current if f(x)]
        assert all(not x['correct'] or x['candidate_present'] for x in rr)
        coverage[g] = dict(n=len(rr), correct=sum(x['correct'] for x in rr),
                           correct_set_absent=sum(not x['candidate_present'] for x in rr),
                           correct_set_present_but_not_selected=sum(x['candidate_present'] and not x['correct'] for x in rr))
    result = dict(original_result=r, all_arm_metrics=metrics, versus_source_only=comparison,
                  post_score_error_partition=coverage, adopted=False, independent_validation=False,
                  diagnostic_only_not_training=True)
    doc = P.DOC
    E.write(doc / '四组对比与错误分层.json', result)
    E.write(doc / '方案.json', E.read(P.OUT / 'protocol.json'))
    E.write(doc / '输入核查.json', E.read(P.OUT / 'input_audit.json'))
    E.write(doc / '审计.json', audit)
    lines = [
        'ModernNCA部分集合学习实验（2026-11-13）', '',
        '本轮补上上一轮接法遗漏：深圳已有340条部分正关系现在直接参与新集合头训练，'
        '其中龙岗53条各有三所已知正学校。未知学校保持未知，不制造完整深圳训练答案。',
        'ModernNCA-LN编码器和匹配权重继续冻结；本轮训练学校表示变换及集合评分头，不是全模型端到端重训。', '',
        '一、同一1761条地图Gold的完整集合准确率',
        '方案 | 全部 | 龙华区 | 龙岗区 | 单校 | 多校 | 两校 | 三校及以上',
    ]
    for arm, name in names.items():
        lines.append(name + ' | ' + ' | '.join(
            f"{metrics[arm][g]['correct']}/{metrics[arm][g]['n']} ({metrics[arm][g]['exact']:.2f}%)"
            for g in ['全部', '龙华区', '龙岗区', '单校', '多校', '两校', '三校及以上']))
    lines += ['', '语义待核样本不删除：']
    for arm, name in names.items():
        m = metrics[arm]['语义待核']
        lines.append(f"{name}：{m['correct']}/{m['n']} ({m['exact']:.2f}%)")
    for arm in ['source_only_set_readout', 'untrained_set_readout', 'categorical_policy']:
        z = comparison['全部'] if arm == 'source_only_set_readout' else r['comparisons'][f'learned_set_readout minus {arm}']['全部']
        lines += ['', f"相对{names[arm]}：{z['delta_pp']:+.3f}个百分点；纠正{z['corrected']}、改错{z['harmed']}；"
                  f"95%空间块bootstrap区间{z['ci95']}。"]
    lines += ['', '三种子结果（全部正确数）：']
    for seed in map(str, E.M.SEEDS):
        lines.append(seed + '：' + '；'.join(
            f"{names[a]}={r['seeds'][a][seed]['全部']['correct']}" for a in E.ARMS)
            + f"；上一轮仅源标签={old_result['seeds']['learned_set_readout'][seed]['全部']['correct']}")
    lines += ['', '二、实际训练内容与选择边界',
        '原有4884条完整源训练标签（4790单校、94两校）保留；新增到集合头损失的是原本已有的深圳部分标签，'
        '龙华37条、龙岗303条。没有新增事实、参照或完整标签。',
        '部分集合损失为全部候选logsumexp减去兼容候选logsumexp。兼容候选包含所有已知正学校，'
        '不得包含明确负学校；未知不作负例。当前深圳部分标签没有明确负例。',
        '训练假设来自冻结政策集合、单校和已有正关系下界。下界仅作为假设，不宣称完整。'
        '有限候选中部分行只有一个兼容集合，仍存在假设空间限制。',
        '训练沿用原项目平衡权重并联合归一化。源采样随机流保持不变，目标采样使用独立随机流。',
        '按源验证选择4组配置和80/160/320/640步检查点，没有依据本轮深圳评分调参数。',
        '选中检查点：' + json.dumps(r['source_selection'], ensure_ascii=False),
        '最终三种子源验证：' + json.dumps(r['source_ensemble'], ensure_ascii=False),
        '检查点单校保护条件通过，不等于最终集成通过；最终集成保护条件为'
        + str(r['source_ensemble']['ensemble_single_gate_passed']) + '。选择步数落在640步预算上限。', '',
        '三、预测候选限制与后验错误统计',
        '预测使用与上一轮完全相同的逐小区候选；没有按目标答案扩候选。大学区不自动扩成员，'
        '共享/共同范围保留适用条件；地图重合不当作入学资格。',
    ]
    for g in ['全部', '单校', '多校', '两校', '三校及以上']:
        c = coverage[g]
        lines.append(f"{g}：正确{c['correct']}；正确集合不在候选中{c['correct_set_absent']}；"
                     f"正确集合在候选中但未选中{c['correct_set_present_but_not_selected']}。")
    lines += ['', '以上错误分层只在预测封存后使用Gold统计，禁止回填成训练候选或标签。',
        '固定候选库对417条样本缺失完整正确集合，其中多校404条。本轮训练无法突破此结构上限。'
        '这不是断言官方没有相应资料。', '',
        '四、核查与结论',
        f"重放{audit['target_probability_arrays_bitwise_replayed']}组目标概率、"
        f"核对{audit['output_rows_independently_recounted']}条输出、"
        f"重放{audit['teacher_latent_rows_bitwise_replayed']}条教师缓存；340条部分标签兼容性逐条核查通过。",
        '全部1802条输出保留；41条地图外不评分；242条语义待核样本仍计入1761条并单报。'
        '原Gold、文件哈希和模型选择已核查。',
        '前一轮目标结果已用于开发判断，这一轮不是独立验证。没有证明Modern无法处理多校。'
        '默认模型不变，adopted=false。',
        '本轮总体结果' + ('高于' if metrics['learned_set_readout']['全部']['correct'] > metrics['categorical_policy']['全部']['correct'] else '未超过')
        + '原Modern政策对照；源集成单校保护条件' + ('通过' if r['source_ensemble']['ensemble_single_gate_passed'] else '未通过') + '，不采用为默认模型。',
        '', '完整产物目录：' + str(P.OUT),
    ]
    (doc / '结果与限制.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    changed = []
    control = {(x['d'], x['site_id']): x for x in rows['categorical_policy']}
    for x in current:
        a = control[x['d'], x['site_id']]
        b = baseline[x['d'], x['site_id']]
        if x['prediction'] != a['prediction'] or x['prediction'] != b['prediction']:
            changed.append(dict(x, original_modern_prediction=a['prediction'], original_modern_correct=a['correct'],
                                source_only_prediction=b['prediction'], source_only_correct=b['correct'],
                                diagnostic_only_not_training=True))
    E.write(doc / '逐条变化_禁止作为训练输入.json', changed)
    E.write(doc / '交付清单.json', dict(files={str(p): E.sha(p) for p in doc.iterdir()
                                           if p.is_file() and p.name != '交付清单.json'},
                                      report_script={str(Path(__file__)): E.sha(Path(__file__))}, adopted=False))
    print('REPORT WRITTEN', doc, flush=True)
    print(json.dumps(dict(metrics=metrics['learned_set_readout'], versus_source_only=comparison['全部'],
                          error_partition=coverage), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
