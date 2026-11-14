"""Post-score report only. Never imported by either inference pipeline."""
from collections import Counter
import json
import experiment_modernnca_scope_cells as X


def pct(a,b):return f'{100*a/b:.2f}%'


def main():
    first=X.read(X.E.OUT/'results.json');second=X.read(X.OUT/'results.json')
    a1=X.read(X.E.OUT/'audit.json');a2=X.read(X.OUT/'audit.json')
    assert a1['passed'] and a2['passed']
    doc=X.DOC;doc.mkdir(parents=True,exist_ok=True)
    metrics=second['metrics'];comp=second['comparisons']['scope_cell_sets minus categorical_policy']['全部']
    cov=second['candidate_coverage'];c0=first['candidate_coverage']['set_same_evidence']
    lines=['ModernNCA 条件学校集合实验（2026-11-13）','',
        '结论：以下是固定 ModernNCA-LN 的集合决策原型开发结果，不是已经解决多校问题，也不是新训练的集合编码器。默认模型与原始 Gold 均未改动。',
        '实验顺序：先封存基于整段范围/共享条款的候选和预测，再评分；发现子范围表达缺口后，封存第二版通用条款子范围构造并评分。第二版是在看过第一版深圳结果之后提出的，不能称为独立验证。','',
        '一、同模型、同权重、同参照、同输入正关系、同1761条评分记录的比较',
        '方案 | 全部 | 龙华627 | 龙岗1134 | 单校803 | 多校958']
    names={'categorical_policy':'Modern原单校输出+现有政策','set_same_evidence':'第一版整段范围候选集合','scope_cell_sets':'第二版条款子范围候选集合'}
    for arm,name in names.items():
        m=metrics[arm]
        lines.append(name+' | '+' | '.join(f"{m[k]['correct']}/{m[k]['n']} ({m[k]['exact']:.2f}%)" for k in ['全部','龙华区','龙岗区','单校','多校']))
    lines += ['',f"第二版相对原Modern对照：{comp['delta_pp']:+.3f}个百分点；纠正{comp['corrected']}条，改错{comp['harmed']}条；分区分层1km块bootstrap3000次，95%区间{comp['ci95']}。",
        '三种子正确数变化：'+str({s:second['seeds']['scope_cell_sets'][s]['全部']['correct']-second['seeds']['categorical_policy'][s]['全部']['correct'] for s in map(str,X.E.M.SEEDS)}),
        '该对照是此前多标签实验中重训的ModernNCA-LN categorical控制组，不是历史RetrievalScorer的959/1761（54.46%）。不能混用模型名称或把两种训练协议称为同协议。','',
        '二、正确集合是否进入候选（全部为封存预测后的Gold诊断，不能回灌训练）',
        '人群 | 第一版有正确候选 | 第二版有正确候选 | 第二版没有正确候选 | 有候选但选错 | 选对']
    for g in ['全部','龙华区','龙岗区','单校','多校','语义待核','其余']:
        a=c0[g];b=cov[g]
        lines.append(f"{g} | {a['exact_candidate_present']}/{a['n']} | {b['candidate_present']}/{b['n']} | {b['candidate_absent']} | {b['present_but_selected_wrong']} | {b['selected_correct']}")
    lines += ['',f"第二版固定候选库的事后可达上限为{cov['全部']['candidate_present']}/1761（{pct(cov['全部']['candidate_present'],1761)}）。这是用Gold计算的诊断上限，不是模型成绩，也不是所有可能的政策输入方式的上限。",
        f"现有输入正关系与固定地图Gold不一致的坐标组有{cov['全部']['lower_bound_conflicts']}个。这里可能有输入估计错误，也可能有政策/地图语义差异；不能直接宣称Gold或官方文字错了。",
        '242条语义待核样本全部保留并单报；41个地图范围外坐标组仍输出，按原协议不参与1761条计分。','',
        '三、这次具体改了什么',
        '1. 官方候选：按202所小学现有冻结原文、共享规则及范围表达构造；保留年份、学段、学校ID、出处、条件和适用状态。',
        '2. 第一版使用整段范围一致、单向包含与共享条款。第二版再提取每个独立充分OR子范围，例如几校均含同一社区时，直接证明该社区条件分别支持各校。',
        '3. 不沿学校关系图做传递闭包；AND、排除、户籍等条件不可删去。大学区成员、资格联动和待核学校不因这一步自动扩展。',
        '4. 固定Modern学校支持概率，使用集合内几何平均支持与源验证选出的集合大小校正比较候选。beta=1.0，在第一版源验证选择后固定，第二版未重新调参。',
        '5. 小区适用性分satisfied/not_satisfied/unknown；unknown可作为模型假设比较，但不能当作已确认事实或完整训练标签。已有社区点/道路估计仍保留原有不确定性。',
        '6. 同一学校集合的不同条件不合并为一条无条件招生事实；部分已知正关系只构成下界。所有模型输出均不构成家庭录取资格证明。','',
        '四、补证据检查与边界',
        '严格完整小区名检查得到龙华2条、龙岗215条充分原文关系。这217条均已在现有输入中；新增关系为0。故补实名证据两臂与各自原输出完全相同，不构成新的有效数据干预。',
        '第二版新增的是已有原文的组合表达，不是新收集的行政边界或住宅身份数据。剩余缺口不能靠改输出数量自动补齐。',
        '源验证1246条中仅60条多校；候选由源训练完整标签生成，而深圳候选由政策构造，两者分布不一致。源校准参数位于本次预定范围上沿，不能宣称优化已充分。',
        '本次没有训练独立的集合条件编码器，也没有加入新的社区行政隶属事实。结果不能证明所有集合学习方法无效。','',
        '五、验证',
        f"第一版独立重放{a1['output_rows_independently_replayed']}条输出；第二版独立重放{a2['output_rows_independently_replayed']}条输出。",
        f"第二版对{a2['unique_scope_implications_exhaustively_checked']}个不同布尔蕴含证明枚举了{a2['boolean_assignments_checked']}种真值赋值；这只验证解析表达式的逻辑，不替代官方原文或地理实体身份核验。",
        '新增单元测试11项通过；输入/代码/权重/候选/预测哈希已保存，原Gold哈希不变。adopted=false；independent_validation=false。','',
        '文件位置：',str(X.E.OUT),str(X.OUT),str(doc),'',
        '后续证据优先级：先核实住宅属于哪个官方社区/子范围及排除条件，补齐可适用的完整集合参照；另用固定完整候选训练和验证集合评分。不得按本轮测试Gold拼候选或回填参照。']
    (doc/'结论与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    source_lines=['官方条件集合及原文索引','源自现有冻结年度政策/解析材料；条件集合是模型候选，不表示该区任一住宅都适用，也不表示自由择校。','']
    diagnostics=[]
    for d in X.E.M.TARGETS:
        cat=X.read(X.OUT/'catalogues'/f'{d}.json');names_by=cat['school_names']
        source_lines += [d,f"学校目录{len(names_by)}所；新增充分子范围条件{len(cat['new_scope_cells'])}条。",'']
        for cell in cat['new_scope_cells']:
            source_lines += [cell['candidate_id'],'条件：'+json.dumps(cell['condition'],ensure_ascii=False),
                '学校：'+'；'.join(names_by[s] for s in cell['school_ids'])]
            for proof in cell['evidence']:
                source_lines.append('原文 '+proof['school_name']+'：'+proof['source_text'])
                for ev in proof['evidence']:
                    source_lines.append('出处：'+ev.get('source_position','')+' '+ev.get('source_url',''))
            source_lines.append('')
        source_lines += ['该区全部学校原文范围（包括未进入多校候选和待核学校）：','']
        for r in cat['all_official_school_scopes']:
            source_lines += [r['school_name'],r['original_scope_text'],'原范围状态：'+r['status']]
            for ev in r['evidence']:source_lines.append(ev.get('source_position','')+' '+ev.get('source_url',''))
            source_lines.append('')
        sites={s['site_id']:s for s in X.read(X.OUT/'site_candidates'/f'{d}.json')}
        byschool={s['school_id']:s for s in cat['all_official_school_scopes']}
        for row in X.read(X.OUT/'scored_rows/scope_cell_sets.json'):
            if row['d']!=d or row['correct']:continue
            site=sites[row['site_id']];excluded_conflict=sorted(set(row['truth'])&set(site['exclusions']))
            reason=('candidate_present_but_ranked_wrong' if row['candidate_present'] else
                'input_positive_lower_bound_conflicts_with_map' if row['lower_bound_conflicts'] else
                'input_exclusion_conflicts_with_map' if excluded_conflict else 'missing_conditional_set_template')
            diagnostics.append(dict(row,diagnostic_only_not_input=True,failure_bucket=reason,
                excluded_gold_ids=excluded_conflict,truth_school_names=[names_by[s] for s in row['truth']],
                predicted_school_names=[names_by[s] for s in row['prediction']],
                official_scope_evidence=[byschool[s] for s in row['truth']]))
    (doc/'官方条件集合及原文.txt').write_text('\n'.join(source_lines)+'\n',encoding='utf8')
    X.write(doc/'逐条错误诊断_禁止作为输入.json',diagnostics)
    buckets=Counter(r['failure_bucket'] for r in diagnostics)
    X.write(doc/'诊断分类.json',dict(all_errors=dict(buckets),multi_errors=dict(Counter(r['failure_bucket'] for r in diagnostics if r['size']>1))))
    X.write(doc/'交付清单.json',dict(files={str(p):X.sha(p) for p in doc.iterdir() if p.is_file() and p.name!='交付清单.json'},
        default_model_changed=False,original_gold_changed=False,adopted=False))
    print('REPORTS WRITTEN',doc,'ERROR BUCKETS',dict(buckets),flush=True)


if __name__=='__main__':main()
