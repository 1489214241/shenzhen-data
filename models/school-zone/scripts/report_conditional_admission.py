"""Report both development stages, including failed primary and adaptive followup."""
import hashlib
from pathlib import Path
import numpy as np
import experiment_conditional_admission as A
import experiment_conditional_admission_hybrid as H

def compare(root,new,old):
    after=A.read(root/'scored_rows'/f'{new}.json');before={(r['d'],r['site_id']):r for r in A.read(root/'scored_rows'/f'{old}.json')}
    delta=[int(r['correct'])-int(before[r['d'],r['site_id']]['correct']) for r in after]
    return dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=100*float(np.mean(delta)),
        ci95=A.M.evaluate.bootstrap(after,delta,20261104,3000))

def main():
    roots=[H.BASE,H.OUT];results=[];audits=[]
    for root in roots:
        A.OUT=root;A.verify();A.check('weights_freeze.json');A.check('prediction_freeze.json')
        ia=A.read(root/'input_audit.json');oa=A.read(root/'output_audit.json');assert ia['passed'] and oa['passed']
        results.append(A.read(root/'results.json'));audits.append(dict(input=ia,output=oa))
    before,after=results;c=A.read(H.OUT/'protocol.json')
    for d in c['sources']+c['targets']:
        p=dict(np.load(H.OUT/'cache/control'/f'{d}.npz'))['prob']
        q=dict(np.load(H.OUT/'cache/routed'/f'{d}.npz'))['prob']
        np.testing.assert_array_equal(p,q)
    for arm in ['control_bce','control_zlpr']:
        for seed in c['seeds']:
            assert A.sha(H.OUT/'weights'/arm/f'{seed}.pt')==A.sha(H.BASE/'weights'/arm/f'{seed}.pt')
        assert before['metrics'][arm+'_legacy']==after['metrics'][arm+'_legacy']
    comparisons={
        'hybrid_bce minus matched_control_bce':compare(H.OUT,'routed_bce_legacy','control_bce_legacy'),
        'hybrid_bce minus native_policy':compare(H.OUT,'routed_bce_legacy','control_base_legacy'),
        'hybrid_zlpr minus matched_control_zlpr':compare(H.OUT,'routed_zlpr_legacy','control_zlpr_legacy'),
        'hybrid_bce_raw minus native_raw':compare(H.OUT,'routed_bce_raw','control_base_raw')}
    doc=A.DOC;doc.mkdir(exist_ok=True)
    lines=['深圳多校继续调研：已执行两阶段实验（2026-11-13）','',
        f"实际结果：第二阶段BCE主候选{after['metrics']['routed_bce_legacy']['全部']['correct']}/1761，"
        f"同类旧证据BCE对照{after['metrics']['control_bce_legacy']['全部']['correct']}/1761，原Modern政策方案898/1761。",
        '主比较（新证据对同类旧证据补校）：'+str(comparisons['hybrid_bce minus matched_control_bce']),
        '相对原Modern政策方案：'+str(comparisons['hybrid_bce minus native_policy']),
        '已经观察到可复现的补校能力，但整体提升尚未确认；区间仍跨零、单校有误加，且本轮是自适应开发。未替换默认模型。','',
        '本次不把一次失败解释成无解：先核查分数尺度和排序可表达性，阅读全文，再运行条件补校及输入路径分离两阶段。',
        '结论必须同时看预定主比较、对照和负面结果；这些是已使用过来源验证和深圳数据上的开发实验，不是独立验证。','',
        '一、实际机制证据',
        '原softmax排序对逐样本公共分数偏移不敏感，原始sigmoid分数敏感；不能直接把排序分数视为校准后的入选概率。',
        '来源60条多校中只有21条可由原分数前缀截断得到；51条的第一名在正确集合内，其余30条需要跳过错误的中间候选。',
        '固定第一名、逐校学习入选可绕开固定集合目录与排序前缀限制；也会保留第一名原有错误。',
        '阅读全文：Calibrated Label Ranking（21页）；ZLPR（17页，含附录）。详细方法与适用边界见文献与实验依据.txt。','',
        '二、第一阶段：条件补校（保留旧规则作为估计输出）',
        '方案 | 全部1761 | 单校803 | 多校958 | 三校及以上364 | 龙华627 | 龙岗1134']
    names={'control_base_legacy':'原Modern＋旧规则','control_bce_legacy':'旧输入＋BCE补校＋旧规则',
        'control_zlpr_legacy':'旧输入＋ZLPR补校＋旧规则','routed_base_legacy':'新输入＋旧规则',
        'routed_bce_legacy':'新输入＋BCE补校＋旧规则','routed_zlpr_legacy':'新输入＋ZLPR补校＋旧规则（第一阶段预定主候选）'}
    fields=['全部','单校','多校','三校及以上','龙华区','龙岗区']
    for arm,name in names.items():lines.append(name+' | '+' | '.join(str(before['metrics'][arm][g]['correct']) for g in fields))
    lines+=['第一阶段主比较：'+str(before['comparisons']['routed_zlpr_legacy']['全部']),
        '旧输入BCE补校对原方案：'+str(before['comparisons']['control_bce_legacy']['全部']),
        '903不是第一阶段预定主方案；只作为预先安排的对照结果，不以事后挑最高组合冒充成功主试验。','',
        '三、第二阶段：旧输入负责第一名，新证据只供补校',
        '第一阶段深圳结果已经看过：新输入提升裸模型补校，却改变原第一名。基于这一证据提出第二阶段，明确为自适应开发。',
        '两个输入组逐样本使用完全相同的旧Modern概率与第一名；新参照与范围特征仅进入139维补校输入。',
        '重新构造排除第一名后的部分标签并训练；旧输入对照权重按字节复用，已有实验和Gold保持。',
        '预定主比较是新证据BCE补校减旧输入BCE补校（903），同时报告相对原方案（898）。',
        '方案 | 全部1761 | 单校803 | 多校958 | 三校及以上364 | 龙华627 | 龙岗1134']
    for arm,name in [('control_base_legacy','原方案'),('control_bce_legacy','旧输入BCE对照'),
                     ('routed_bce_legacy','保留旧第一名＋新证据BCE补校（第二阶段主候选）'),
                     ('routed_zlpr_legacy','保留旧第一名＋新证据ZLPR补校')]:
        lines.append(name+' | '+' | '.join(str(after['metrics'][arm][g]['correct']) for g in fields))
    for name,metric in comparisons.items():lines.append(name+'：'+str(metric))
    for arm in ['control_bce_legacy','routed_bce_legacy','routed_zlpr_legacy']:
        lines.append(arm+'补校初始化种子结果：'+str({seed:m['全部']['correct'] for seed,m in after['seeds'][arm].items()}))
    lines+=['','四、模型自己判多校与规则补校必须分开','裸输出 | 全部 | 单校 | 多校 | 三校及以上']
    for stage,r in [('第一阶段',before),('第二阶段',after)]:
        for arm in ['control_base_raw','control_bce_raw','routed_bce_raw','routed_zlpr_raw']:
            lines.append(stage+'/'+arm+' | '+' | '.join(str(r['metrics'][arm][g]['correct']) for g in fields[:4]))
    lines+=['模型补对的多校可能已被旧规则补对，所以裸模型提高不等于完整系统获得同等增益。',
        '原规则在对照中是估计输出，不意味着其中位置推断被确认为真值。仅文字约束对照也保存在完整结果中。','',
        '五、来源选择与限制']
    for stage,root,r in zip(['第一阶段','第二阶段'],roots,results):
        for arm in ['control_bce','control_zlpr','routed_bce','routed_zlpr']:
            sel=A.read(root/'selection'/f'{arm}.json')
            lines.append(stage+'/'+arm+'选中步数='+str(sel['best']['step'])+'；来源集成='+str(r['source'][arm]))
    lines+=['来源验证反复使用，来源完整多校仅60条验证样本且只有两校，不能将小规模来源改善当作泛化结论。',
        '控制组ZLPR选择320步位于预算上限，不能宣称最优训练长度已确认。',
        '旧第一名错误、新证据误差、条件适用性未闭合、非随机缺失、三校及以上迁移仍是限制。',
        '掩码ZLPR是本地部分标签适配，不享有原完整标签论文未经验证的保证。',
        '空间块bootstrap区间用于开发描述；没有为多次开发和多种对照选择进行多重比较校正。',
        'adopted=false；independent_validation=false；默认模型未替换。不能把观测零新增错判说成未来零风险。','',
        '六、验证与复现',
        '4项针对性测试通过：未知零梯度、全未知零损失、第一名保留且可跳过中间候选、学校置换一致性。',
        '两阶段分别重放24组保存概率，核对57664条裸/文字约束输出；重计全部集成分数，核查Gold哈希不变。',
        '两阶段均保留1802条输出；1761条地图命中计分，41条地图外不计分，242条语义待核仍计入。',
        '第一阶段对原898条正确结果逐条精确重现；第二阶段旧第一名概率及复用对照权重逐项/逐字节相等。',
        '推理接口曾补齐旧补充器要求的decision字段，修复前代码及前后哈希保留在第一阶段integration_fix_receipt.json；未改训练或选模。',
        '第一阶段：'+str(H.BASE),'第二阶段：'+str(H.OUT),
        '执行脚本：scripts/experiment_conditional_admission.py；scripts/experiment_conditional_admission_hybrid.py。',
        '已有输出均封存，prepare拒绝覆盖。不要按本轮评分样本回填训练。']
    (doc/'继续调研与实验结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    A.write(doc/'两阶段结果.json',dict(first=before,hybrid=after,hybrid_comparisons=comparisons,adopted=False))
    A.write(doc/'独立复算记录.json',audits)
    A.write(doc/'实际使用入口.json',dict(first=str(H.BASE),hybrid=str(H.OUT),production_changed=False,adopted=False))
    old={(r['d'],r['site_id']):r for r in A.read(H.OUT/'scored_rows/control_bce_legacy.json')}
    changes=[dict(r,before_prediction=old[r['d'],r['site_id']]['prediction'],before_correct=old[r['d'],r['site_id']]['correct'],diagnostic_only_not_training=True)
        for r in A.read(H.OUT/'scored_rows/routed_bce_legacy.json') if r['prediction']!=old[r['d'],r['site_id']]['prediction']]
    A.write(doc/'第二阶段逐条变化_禁止作为训练输入.json',changes)
    A.write(doc/'交付校验.json',dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in doc.iterdir() if p.is_file() and p.name!='交付校验.json'},
        reporter_sha256=A.sha(Path(__file__)),adopted=False))
    print('RESEARCH REPORT WRITTEN',doc,comparisons,flush=True)

if __name__=='__main__':main()
