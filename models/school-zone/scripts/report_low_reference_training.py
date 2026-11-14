"""Describe frozen results; never selects a new candidate or changes predictions."""
from pathlib import Path
import json
from experiment_low_reference_training import OUT, read, write, ARMS

NAMES={'global_full':'已有多标签结构／普通训练','global_sparse':'已有多标签结构／低参照训练',
       'local_full':'按学校构造查询／普通训练','local_sparse':'按学校构造查询／低参照训练',
       'archived_modern_tuned':'旧目标1 Modern微调','archived_modern_frozen':'旧目标1 Modern未微调'}


def main():
    result=read(OUT/'results.json');audit=read(OUT/'independent_audit.json');assert audit['passed']
    selection=read(OUT/'training_selection.json');thresholds=read(OUT/'thresholds.json')
    lines=['深圳低参照训练与逐校查询实验','',
        '新增核验：0。沿用目标1的84个模拟核验小区，已有文字参照、弱参照及政策处理保留。',
        '两区固定1095题，参照小区及其同楼盘组不进入测试。目标1是补到至少1个，不是部署时总共只能看1个。',
        '每个新版本3个种子，完全相同的训练标签、损失、超参数预算和源区验证集。全部随机初始化，不读取512参照权重。',
        '源区普通版本保留全部合法参照；稀疏版本轮换8份源区训练视图，每所有参照学校最多1条关系。另1份固定源区稀疏视图验证。',
        '稀疏源区选择是训练信息遮罩，不是新增人工核验。深圳参照不轮换。验证楼盘组不进入源区参照池。',
        '重新计算道路分界、楼盘权重、已知行名称遮罩和6项周边统计；官方政策、道路几何及社区证据保留。',
        '只用原本单校的可见观察推导道路分界；多校记录被遮掉部分关系后不会被当成已确认单校。','',
        '完整学校集合准确率（多一校、少一校都算错）：',
        '版本 | 全部 | 龙华 | 龙岗 | 单校 | 多校 | 新参照未覆盖组合']
    for arm in ARMS+['archived_modern_tuned','archived_modern_frozen']:
        mm=result['metrics'][arm+'_policy']
        vals=[f"{mm[k]['correct']}/{mm[k]['n']}={mm[k]['exact']:.2f}%" for k in ['all','龙华区','龙岗区','single','multi','unseen']]
        lines.append(NAMES[arm]+' | '+' | '.join(vals))
    lines += ['', '配对变化（百分点；95%空间块区间；修正／误伤；原单校正确被误伤）：']
    for key,v in result['comparisons'].items():
        lines.append(f"{key}: {v['delta_pp']:+.2f}; [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]; {v['corrected']}/{v['harmed']}; {v['single_harmed']}"+
                     ('；同训练条件' if v['matched_training'] else '；旧结果仅作成绩参照，非单因素因果对照'))
    p=result['comparisons']['global_sparse_policy minus global_full_policy']
    if p['ci95'][0]>0: verdict='预设主要对照在当前开发数据上出现正向改善。'
    elif p['ci95'][1]<0: verdict='预设主要对照在当前开发数据上有害。'
    else: verdict='预设主要对照未确认改善；区间跨零不能解释为证明无效。'
    lines += ['',verdict,
        '结构变化及交互为补充对照，results.json另列4项对照的98.75%区间，不能只挑最好一行宣称方法成功。',
        '旧46.94%采用预训练后目标区微调及目标参照验证。本轮采用从零训练及源区验证；旧成绩只判断实际竞争力，不隔离训练改动。',
        '未覆盖组合按新增完整参照集合事后分组；不是整个政策系统的信息上限，也不参与训练或阈值选择。','',
        '错误变化（冻结预测后的描述，不据此重新调参）：']
    for a in ARMS:
        mm=result['metrics'][a+'_policy']
        lines.append(f"{NAMES[a]}：错误添加学校{mm['all']['fp']}次，漏掉学校{mm['all']['fn']}次。")
    g=result['metrics']['global_full_policy'];s=result['metrics']['global_sparse_policy'];local=result['metrics']['local_full_policy']
    lines += [f"已有多标签结构的低参照训练：错误添加学校增加{s['all']['fp']-g['all']['fp']}次，漏校变化{s['all']['fn']-g['all']['fn']:+d}次。这里主要是误加学校，没有换来相应的漏校改善。",
        f"普通训练时按学校构造查询：单校多判对{local['single']['correct']-g['single']['correct']}题，多校变化{local['multi']['correct']-g['multi']['correct']:+d}题，不能用总分上涨宣称解决多校。"]
    if thresholds['local_full']['threshold']==1.:
        lines.append('按学校构造查询＋普通训练的源区最优阈值为1.0，原始输出实际退回单校；其多校预测来自既有政策补校。这是本轮最好的新版本，但不是新的多校能力。')
    lines += ['',
        '源区选择（每行同一个阈值用于两目标区）：']
    for arm in ARMS:
        chosen=[selection[f'{arm}/{s}']['best']['step'] for s in [20261231,20261232,20261233]]
        atcap=sum(selection[f'{arm}/{s}']['at_step_limit'] for s in [20261231,20261232,20261233])
        lines.append(f"{NAMES[arm]}：步骤{chosen}，阈值{thresholds[arm]['threshold']:.8f}；{atcap}/3在320步上限。")
    seeds=read(OUT/'seed_metrics.json')['metrics']
    lines += ['', '三个单种子深圳准确率（统一用该版本源区集成阈值，不挑种子）：']
    for a in ARMS:
        lines.append(NAMES[a]+'：'+' / '.join(f"{v['all']['exact']:.2f}%" for v in seeds[a].values()))
    lines += ['', '局限：固定学习率及320步上限，不宣称充分收敛；反复使用的开发地区，不是独立泛化验证。',
        '独立复核：逐题集合计分、FP/FN、三种子均值、原始解码、政策重放及哈希检查通过；未替换默认模型。',
        '本地重跑：python scripts/experiment_low_reference_training.py prepare / fit / score / audit（依次分别运行）。',
        'prepare需使用新的输出目录，避免覆盖本轮冻结结果。完整训练依赖原历史源区及目标区特征输入；公开副本只提供轻量结果复核，不称为从零重训。']
    (OUT/'实验结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
