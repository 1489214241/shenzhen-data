"""Deliver the implemented evidence routes and the sealed experiment outcome."""
import hashlib
import json
from pathlib import Path

import continue_modernnca_evidence_v2 as C


def main():
    C.install(); E = C.E
    result = E.read(E.OUT / 'results.json')
    ia = E.read(E.OUT / 'input_route_audit.json')
    oa = E.read(E.OUT / 'output_audit.json')
    assert ia['passed'] and oa['passed']
    E.verify(); E.check('prediction_freeze.json'); E.check('weights_freeze.json')
    doc = E.DOC; doc.mkdir(parents=True, exist_ok=True)
    names = dict(native_modern_policy='原Modern政策对照', control_legacy='旧输入＋旧约束（同架构训练对照）',
                 control_verified='旧输入＋仅文字证明约束', routed_legacy='新输入＋旧约束',
                 routed_verified='新输入＋仅文字证明约束（预定主候选）', control_raw='旧输入裸模型', routed_raw='新输入裸模型')
    primary = result['comparisons']['routed_verified minus control_legacy']['全部']
    diagnosis = dict(diagnostic_only_not_training=True, adopted=False, arms={})
    for arm in E.ARMS:
        selected = result['selection'][arm]
        histories = [E.read(E.OUT / 'selection' / arm / f'trial{j}/trial.json')['history']
                     for j in range(len(E.read(E.OUT / 'protocol.json')['grid']))]
        trained = [r for history in histories for r in history if r['step'] > 0]
        unchanged = {}
        zero_columns = {}
        for seed in E.M.SEEDS:
            state = E.torch.load(E.OUT / 'weights' / arm / f'{seed}.pt', map_location=E.M.DEVICE, weights_only=True)
            original = E.teacher({}, seed).state_dict()
            unchanged[str(seed)] = all(E.torch.equal(v, original[k]) for k, v in state.items())
            w = state['encoder.encoder.weight']
            zero_columns[str(seed)] = bool((w[:, 41:44] == 0).all() and (w[:, 71:74] == 0).all())
        diagnosis['arms'][arm] = dict(selected_step=selected['best']['step'],
            trained_checkpoints=len(trained), trained_checkpoints_passing=sum(r['source_gate_passed'] for r in trained),
            maximum_trained_single_correct=max(r['single_correct'] for r in trained),
            source_singleton_floor=selected['best']['singleton_floor'],
            weights_equal_warm_start=unchanged, whole_scope_column_weights_zero=zero_columns,
            calibrated_threshold=result['source_calibration'][arm]['threshold'])
    lines = ['深圳全链路修复v2：实施、实验与限制（2026-11-13）', '',
        f"实际结论：预定主候选{result['metrics']['routed_verified']['全部']['correct']}/1761，"
        f"同条件旧输入旧约束{result['metrics']['control_legacy']['全部']['correct']}/1761；"
        f"变化{primary['delta_pp']:+.3f}个百分点，纠正{primary['corrected']}条、损害{primary['harmed']}条。不采用。",
        '本轮没有解决多校预测。输入与约束修复已落地，但两组最终均选中0步：所有训练后检查点均未通过来源单校保护。'
        '保存权重逐张量等于扩列后的原教师，新范围特征列权重仍为0；不能把它们已接线说成模型已经学会使用。',
        '定位出的主要断点：来源校准阈值为1，两组裸模型在深圳均只输出单校，多校正确数为0。'
        '旧约束对照能补对322条多校；同一模型改为仅文字证明约束后只剩53条，损失269条。'
        '新输入保留旧约束仍从897降至882；新输入再改约束后从882降至613。'
        '这是本次受控对照的分解，不能证明所有范围特征或参照都无效。',
        '下一步技术方向：保留单校排序能力，单独学习第二所及后续学校的条件适用性；'
        '旧推断只作为可反驳的候选或特征，不能未经核实改成真值，也不能在替代模型尚未学会时直接删除其预测作用。'
        '新的集合预测必须先在来源验证同时通过单校保护和多校非退化检查；未通过时保留原方案。'
        '这是后续建议，本轮没有完成该额外模型，也没有用本轮深圳错误回填训练。', '',
        '本轮不只修改输出头：已建立统一输入快照、逐关系证据路由、基础条款编译、完整范围特征通道、'
        '来源与深圳部分标签训练、可反驳估计的输出约束，并运行完整Modern编码器微调对照。'
        '实验是否采用与代码是否完成分开报告。原数据、原Gold、历史实验和默认模型保持。', '',
        '一、已落地内容',
        '1. 统一可追踪输入：391条有文字证明的正关系、7条明确负关系进入深圳部分监督；269条估计关系'
        '只作特征／参照线索，不再作为训练真值或硬输出条件。所有关系保留来源、年份、学段、极性和通道。'
        '这沿用现有已审核资料，没有声称本轮新增获取391条独立事实。',
        '2. 全部已接纳文字正关系均进入参照：补齐花语岸、华龙新村对应阳光小学的两条无条件名单关系；'
        '户籍限定的其他分支保持未知。参照648条学校关系，覆盖108/202校（原491条、80/202校）。'
        '参照库仍含估计，不把648条全部称为核验事实。',
        '3. 大学区按条款处理：保留龙华5所大学区成员校各自原地段；不从成员名单扩校。'
        '户籍混合条款只保留独立无条件OR分支，AND、排除、居民小组资格和未解析条件不得丢弃。'
        '龙华70校、龙岗118校有可支持的基础分支；其余范围仍保留原文和待核状态。',
        '4. 完整范围判断接入网络输入：三城统一增加整条范围的未知／范围外估计／范围内估计三列，'
        '原41/27列扩为44/30列；参照路径按参照学校对齐，学校路径按学校ID对齐。'
        '完整AST、条款级观察和来源随快照保存。此版输入的是完整公式执行结果，尚不是神经网络直接读取原文或整棵AST。',
        '5. 不再只能从固定完整集合目录里选：逐学校输出成员假设，再执行已落实的文字正／负约束。'
        '未知不当负例、估计不强制扩校、无学校图传递闭包；明确排除后为空则保留拒判。'
        '这去掉旧有限模板目录的表达限制，不保证学校组合预测正确。',
        '6. 全网络参与微调：保留ModernNCA-LN的证据编码、查询、距离和学校聚合；'
        '对旧教师输入层做零权重扩列，使旧信息的初始投影保持。源69条部分标签也参与；'
        '使用观测关系平均BCE，完整单校源标签对多报学校施加惩罚，不再依赖只约束单校内部排序的KL。', '',
        '二、实验边界',
        '两输入组使用相同Modern架构、起点、学习率候选、正则、来源划分和随机种子；'
        '源单／多校权重沿用旧平衡设置。输入版本是组合干预，不能把总变化全部归因于某一个新字段。',
        '先运行0/20/40/80步检查；训练候选未通过单校保护、但来源曲线尚在恢复，因此在没有深圳推断或评分前，'
        '另建冻结实验延长一次至160/320步，同时补齐两条参照接入。两次实验均保留，延长并非独立确认。',
        '检查点及阈值只用原1246条来源验证选择；要求单校准确率比原教师下降不超过0.5个百分点，'
        '在通过者中按完整集合正确数选。包含0步起点；若最终选0步，不得宣称新监督训练有效。',
        '每个模型分别运行旧约束、新文字约束和裸输出；预定主比较是routed_verified减control_legacy。'
        '不看深圳结果挑选另外的组合或混合分区。', '',
        '三、同一1761条固定地图Gold，完整学校集合相等',
        '方案 | 全部 | 龙华627 | 龙岗1134 | 单校803 | 多校958 | 三校及以上364',
    ]
    for arm in ['native_modern_policy', 'control_legacy', 'control_verified', 'routed_legacy', 'routed_verified', 'control_raw', 'routed_raw']:
        metric = result['metrics'][arm]
        lines.append(names[arm] + ' | ' + ' | '.join(f"{metric[g]['correct']}/{metric[g]['n']} ({metric[g]['exact']:.2f}%)"
            for g in ['全部', '龙华区', '龙岗区', '单校', '多校', '三校及以上']))
    for key, values in result['comparisons'].items():
        z = values['全部']
        lines.append(f"{key}：{z['delta_pp']:+.3f}个百分点；纠正{z['corrected']}、损害{z['harmed']}；95%空间块区间{z['ci95']}。")
    lines += ['', '四、模型是否真正选择了训练后权重']
    for arm in E.ARMS:
        selected = result['selection'][arm]
        lines.append(f"{arm}：选中{selected['best']['step']}步，配置{selected['hp']}；源选择指标"
                     + json.dumps(selected['best'], ensure_ascii=False))
        lines.append('最终三种子来源验证：' + json.dumps(result['source_calibration'][arm], ensure_ascii=False))
        lines.append('三种子深圳主输出正确数：' + str({s: result['seeds'][f'{arm}_verified'][s]['全部']['correct'] for s in map(str, E.M.SEEDS)}))
    lines += ['', '五、采用状态与尚未解决的问题',
        'adopted=false；independent_validation=false。源码修复和输入审计通过不等于准确率提升。'
        '本轮目标区、来源验证均已有开发历史；正向点估计也不能代替独立验证。',
        '来源完整多校训练仍只有双校；社区点仍是位置代理；道路、别名、资格和地图年度语义未全部解决。'
        '来源监督、参照稀疏程度与深圳不同。参照库的估计来源已记录，但本版没有给每种来源训练独立可信度模型。',
        '本版神经输入使用完整范围三值执行结果，尚未实现原文／AST的可学习结构编码；'
        '可反驳估计和更大输出空间也可能失去原规则的正确补充，已通过交叉解码对照单列。'
        '不能把任何结果解释为所有范围建模或Modern本身无效。',
        '全部1802条输出保留；41条地图外不计分；242条语义待核仍计入并单报。没有根据目标Gold补参照、'
        '制造完整深圳标签、调整候选或修改答案。', '',
        '六、核查与复现',
        '8项针对性测试通过：大学区原地段、混合资格OR、AND／排除、估计与硬约束隔离、'
        '范围状态和参照学校对齐、明确负例、扩列投影保持、未知标签零梯度。',
        '所有来源标签／划分保持；198382个目标小区—学校范围状态和参照掩码重放；'
        '强文字关系已进入参照的遗漏数为0。',
        f"目标概率重放{oa['target_probability_arrays_bitwise_replayed']}组；裸输出及文字约束输出核对"
        f"{oa['raw_and_verified_output_rows_checked']}条；全部集成分数、来源选模和阈值、Gold哈希再次核查。",
        '新输入目录：' + str(E.INPUT),
        '完整实验目录：' + str(E.OUT),
        '执行入口：scripts/continue_modernnca_evidence_v2.py；阶段train/infer/score/audit_inputs/audit_outputs。'
        '已有封存输出不应随意重跑覆盖；prepare会拒绝覆盖。',
    ]
    lines += ['', f"本轮主比较实际结论：{primary['delta_pp']:+.3f}个百分点，95%区间{primary['ci95']}。"]
    (doc / '实施结果与限制.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    E.write(doc / '输入接入核查.json', ia)
    E.write(doc / '输出与评分核查.json', oa)
    E.write(doc / '选模失败定位_禁止作为训练输入.json', diagnosis)
    E.write(doc / '实际使用入口.json', dict(input_directory=str(E.INPUT), experiment_directory=str(E.OUT),
        main_candidate='routed_verified', protocol=str(E.OUT / 'protocol.json'),
        adopted=False, production_default_changed=False))
    before = {(r['d'], r['site_id']): r for r in E.read(E.OUT / 'scored_rows/control_legacy.json')}
    changes = []
    for r in E.read(E.OUT / 'scored_rows/routed_verified.json'):
        b = before[r['d'], r['site_id']]
        if r['prediction'] != b['prediction']:
            changes.append(dict(r, before_prediction=b['prediction'], before_correct=b['correct'], diagnostic_only_not_training=True))
    E.write(doc / '逐条变化_禁止作为训练输入.json', changes)
    E.write(doc / '交付校验.json', dict(files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in doc.iterdir() if p.is_file() and p.name != '交付校验.json'},
        reporter_sha256=E.sha(Path(__file__)), adopted=False))
    print('V2 REPORT WRITTEN', doc, flush=True)


if __name__ == '__main__':
    main()
