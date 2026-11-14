"""Describe the completed validation pilot; never select additional settings."""
import json
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_support as run
from run_modernnca_relation import digest,save
from zoning import evaluate


def main():
    results=json.loads((run.DOC/'验证结果.json').read_text(encoding='utf8'))
    rows=json.loads((run.OUT/'validation_cases.json').read_text(encoding='utf8'))
    audit=json.loads((run.DOC/'重放审计.json').read_text(encoding='utf8'))
    interventions=json.loads((run.OUT/'trained_intervention_cases.json').read_text(encoding='utf8'))
    extra={}; names=dict(scalar='整体尺度',diagonal='逐维尺度',covariance='完整协方差',ridge='岭回归分类头')
    for m in names:
        delta=[int(r['predictions'][m]==r['truth'])-int(r['predictions']['identity']==r['truth']) for r in rows]
        clustered=[dict(d=r['validation'],block=r['block']) for r in rows]
        ci=evaluate.bootstrap(clustered,delta,seed=20261113,draws=10000)
        sels=[json.loads(f.read_text(encoding='utf8')) for f in (run.OUT/'selection').glob(f'*_{m}.json')]
        extra[m]=dict(descriptive_postselection_ci=ci,selected_at_80=sum(s['steps']==80 for s in sels),
            caution='Selection-set interval, conditional on already selected settings; not an independent significance test. Duplicated validation cases share spatial bootstrap blocks.')
    intervention_summary={}
    for kind in ['covariance','ridge']:
        for switch in ['identity','scalar','diagonal'] if kind=='covariance' else ['identity']:
            rr=[r for r in interventions if r['kind']==kind and r['switch']==switch]
            a=np.array([r['trained_prediction']==r['truth'] for r in rr]); b=np.array([r['switched_prediction']==r['truth'] for r in rr])
            intervention_summary[f'{kind}-{switch}']=dict(n=len(rr),delta=float((a.astype(float)-b).mean()*100),fixed=int((a&~b).sum()),broken=int((~a&b).sum()))
    save(run.DOC/'补充统计.json',dict(methods=extra,trained_weight_interventions=intervention_summary,
        reporter_sha256=digest(Path(__file__))))
    promoted=[m for m,v in results['verdict'].items() if v['promoted']]
    lines=['参照条件化改进：本轮研究判断','',
        ('通过验证区筛查：'+', '.join(promoted)+'；仍需后续冻结开发评价，不能据此替换主模型。') if promoted else '两个主要候选均未通过事先规定的验证区晋级门槛。本轮停止扩跑目标评价集；研究基线仍为ModernNCA-LN。',
        '', '一、实际做了什么',
        '先在同一组LN权重上，仅替换距离/分类头；再让各改法与编码器、查询生成器、温度共同训练。全参照、全关系与学校证据、全候选、训练标签、排除规则保持一致。',
        '8组目标—验证情景，覆盖6个不同验证区；每种方法4组学习率/正则，20/40/80步检查，1个选择种子。不是K1三批的完整确认实验。',
        'LN使用旧四组曲线截取相同步数预算，重新训练验证重放；新方法各自选择，未借用LN最优参数代替调参。',
        '', '二、重新训练后的结果（验证题，重复情景合并口径）']
    for m,name in names.items():
        z=results['trained'][m]['两城']; lo,hi=extra[m]['descriptive_postselection_ci']
        lines.append(f"{name}：{z['accuracy_a']:.3f}%；LN {z['accuracy_b']:.3f}%；差 {z['delta']:+.3f} 点；改对/误伤 {z['fixed']}/{z['broken']}。选择后描述区间 [{lo:+.3f}, {hi:+.3f}]。")
        city=results['trained'][m]
        lines.append(f"  长沙差 {city['长沙']['delta']:+.3f} 点；杭州差 {city['杭州']['delta']:+.3f} 点。8组中 {extra[m]['selected_at_80']} 组选在80步上限。")
    lines += ['以上区间是选择后描述统计，不校正选择乐观偏差，不构成确认提升或证明有害；同一验证区的重复题按相同空间块一起重抽。',
        '', '三、两个主要候选的事先判定',json.dumps(results['verdict'],ensure_ascii=False,indent=2),
        '完整协方差额外要求优于整体尺度和逐维尺度；不能把温度/尺度校准收益归为相关性建模。']
    for c in ['scalar','diagonal']:
        z=results['trained']['covariance'][f'vs_{c}']; lines.append(f"完整协方差减{names[c]}：{z['delta']:+.3f}点；改对/误伤 {z['fixed']}/{z['broken']}。")
    lines += ['', '四、固定权重和重新训练要分开解释']
    for m in ['covariance','ridge']:
        z=results['frozen_intervention'][m]['两城']; lines.append(f"原LN权重直接换{names[m]}：{z['delta']:+.3f}点，改对/误伤 {z['fixed']}/{z['broken']}。")
    for k,z in intervention_summary.items():
        lines.append(f"候选训练后，保留其权重：{k} 读出差 {z['delta']:+.3f}点，改对/误伤 {z['fixed']}/{z['broken']}。")
    lines += ['上述后训练干预不等于从头重训消融，不能将差值当作模块相对原LN的独立贡献。',
        '', '五、结论边界与完整性',
        '数值公式与梯度检查通过；零/单参照、全屏蔽、候选保留、标签隔离、换序、特征梯度等检查通过。',
        f"40组所选权重/概率全部重新推理核对，最大概率误差 {audit['max_error']:.9g}。新方法128次验证调参拟合，LN另做8次所选设置重放；没有将历史实验算作本轮新训练。",
        '80步短预算及单个选择种子限制了结论，尤其选在上限的配置尚不能排除进一步训练改善。未测试学校级协方差、SVM或其它收缩/正则值，不能宣布Simple CNAPS/MetaOptNet整类方法无效。',
        '本轮不是主表排名更新，没有重新比较GBT/RetrievalScorer，也没有新的K0/K1目标准确率。开发数据反复使用、杭州答案未冻结。没有据此宣称新算法贡献。',
        '复现见研究说明.txt；数值见验证结果.json、补充统计.json；逐题记录和权重位于独立outputs目录。']
    (run.DOC/'研究判断.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines))


if __name__=='__main__': main()
