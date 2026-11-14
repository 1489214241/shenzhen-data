"""Post-run integrity/description only: never changes choices or trains models."""
import json
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_interaction as run
from run_modernnca_relation import digest,save


def main():
    complete=json.loads((run.OUT/'completed.json').read_text(encoding='utf8'))
    frozen=json.loads((run.OUT/'frozen.json').read_text(encoding='utf8'))
    assert digest(run.OUT/'frozen.json')==complete['frozen_sha256']
    assert all(digest(REPO/p)==h for p,h in frozen['source_hashes'].items())
    rows=json.loads((run.OUT/'validation_cases.json').read_text(encoding='utf8'))
    result=json.loads((run.DOC/'验证结果.json').read_text(encoding='utf8'))
    replay=json.loads((run.DOC/'重放审计.json').read_text(encoding='utf8'))
    assert len(replay['models'])==32
    archive=REPO/'outputs/modernnca_mechanism_20261112/selection'
    audits=[];runtime={};caps={};choices={};initial_identity={}
    for kind in run.METHODS:
        times=[];atcap=0
        for d in frozen['validation']:
            chosen=json.loads((run.OUT/'selection'/f'{d}_{kind}.json').read_text(encoding='utf8'))
            assert chosen['validation']==frozen['validation'][d]
            best=max(chosen['trials'],key=lambda t:run.rank(t['best']))
            assert chosen['hp']==best['hp'] and chosen['steps']==best['best']['step']
            atcap+=chosen['steps']==160
            for i,t in enumerate(chosen['trials']):
                assert t['complete'] and [h['step'] for h in t['history']]==frozen['checkpoints']
                assert t['hp']==frozen['grid'][i]
                path=run.OUT/'trials'/f'{d}_{kind}_{i}'
                assert digest(path.with_suffix('.pt'))==t['state_hash']
                assert digest(path.with_suffix('.npz'))==t['probability_hash']
                prob=np.load(path.with_suffix('.npz'))['P']
                assert np.isfinite(prob).all() and (prob>=0).all()
                assert np.allclose(prob.sum(1),1,atol=2e-5)
                times.append(t['seconds']);audits.append(str(path.relative_to(run.OUT)))
            if kind=='identity':
                old=json.loads((archive/f'{d}_layernorm.json').read_text(encoding='utf8'))
                errs=[]
                for t in chosen['trials']:
                    prior=next(z for z in old['trials'] if z['hp']==t['hp'])
                    history={h['step']:h for h in prior['history']}
                    for h in t['history']:
                        errs.append((abs(h['accuracy']-history[h['step']]['accuracy']),abs(h['nll']-history[h['step']]['nll'])))
                initial_identity[d]=dict(max_accuracy_error=max(x[0] for x in errs),max_nll_error=max(x[1] for x in errs))
                assert initial_identity[d]['max_accuracy_error']==0
                assert initial_identity[d]['max_nll_error']<2e-5
            choices[f'{d}_{kind}']=dict(hp=chosen['hp'],steps=chosen['steps'])
        runtime[kind]=dict(total_seconds=sum(times),median_trial_seconds=float(np.median(times)))
        caps[kind]=atcap
    assert len(audits)==128
    interventions={}
    for k in run.METHODS[1:]:
        actual=np.array([r['predictions'][k]==r['truth'] for r in rows]);bypass=np.array([r['bypass'][k]==r['truth'] for r in rows])
        interventions[k]=dict(delta=float((actual.astype(float)-bypass).mean()*100),fixed=int((actual&~bypass).sum()),broken=int((~actual&bypass).sum()))
    diagnosis=json.loads((run.OUT/'diagnosis.json').read_text(encoding='utf8'))
    dr=diagnosis['rows'];strata={}
    for name,rr in [('true_reference_mass_zero',[r for r in dr if r['true_reference_mass']==0]),
                    ('true_reference_mass_0_to_2',[r for r in dr if 0<r['true_reference_mass']<=2]),
                    ('true_reference_mass_above_2',[r for r in dr if r['true_reference_mass']>2])]:
        strata[name]=dict(n=len(rr),accuracy=float(np.mean([r['pred']==r['truth'] for r in rr])*100))
    metadata={(r['target'],r['q']):r for r in dr}
    diagnostic_comparisons={}
    for k in run.METHODS[1:]:
        z={}
        for name,condition in [('zero',lambda r:r['true_reference_mass']==0),('low',lambda r:0<r['true_reference_mass']<=2),('many',lambda r:r['true_reference_mass']>2)]:
            rr=[r for r in rows if condition(metadata[(r['target'],r['q'])])]
            z[name]=run.comparison(rr,k,'identity')
        diagnostic_comparisons[k]=z
    audit=dict(completed_trials=len(audits),selected_models=32,source_hashes_verified=True,
        identity_history_replay=initial_identity,cap_selections=caps,runtime=runtime,
        bypass_interventions=interventions,old_LN_diagnosis=strata,diagnostic_comparisons=diagnostic_comparisons,
        report_sha256=digest(Path(__file__)),choices=choices)
    save(run.DOC/'完整性与诊断.json',audit)
    names=dict(identity='原ModernNCA-LN',rowmlp='逐条MLP残差',statistics='学校集合统计残差',isab='ISAB参照交互')
    promoted=[k for k,v in result['verdict'].items() if v['promoted']]
    if complete['target_evaluated']:
        decisions=json.loads((run.DOC/'目标判定.json').read_text(encoding='utf8'))
        confirmed=[names[k] for k,v in decisions.items() if v['confirmed']]
        outcome=('冻结后目标比较通过：'+', '.join(confirmed)+'；仍仅为开发数据结论。') if confirmed else '全部比较完成，三个改动均未确认优于原ModernNCA-LN；保留原研究基线。逐条MLP虽通过验证筛查，但收益未稳定保持到目标区。'
    else:
        outcome='三个新增模块均未通过事先冻结的验证筛查条件，没有扩跑目标题；研究基线仍为原ModernNCA-LN。'
    lines=['ModernNCA参照交互：本轮研究判断','',
        outcome,
        '', '一、本轮实际完成',
        '四种方法全部从头训练：每种8个训练情景×4组学习率/正则，128次拟合；20/40/80/160步选择，单个选择种子。8个情景覆盖6个不同验证区。',
        '全部合法参照、逐条关系证据、学校证据、候选、训练题与标签保持一致。自项目参照在编码和交互前排除；三个残差模块的新增参数相差不到0.7%。',
        '', '二、验证结果（重复情景合并的选择集分数）']
    for k in run.METHODS[1:]:
        z=result['comparisons'][f'{k}-identity'];a=z['两城'];lo,hi=z['descriptive_postselection_ci']
        lines.append(f"{names[k]}：{a['accuracy_a']:.3f}%；原LN {a['accuracy_b']:.3f}%；差 {a['delta']:+.3f} 点，改对/误伤 {a['fixed']}/{a['broken']}；选择后描述区间 [{lo:+.3f}, {hi:+.3f}]。")
        lines.append(f"  长沙 {z['长沙']['delta']:+.3f} 点；杭州 {z['杭州']['delta']:+.3f} 点；{result['verdict'][k]['positive_scenarios']}/8个情景正向。")
    for k in ['rowmlp','statistics']:
        z=result['comparisons'][f'isab-{k}']['两城'];lines.append(f"ISAB减{names[k]}：{z['delta']:+.3f}点，改对/误伤 {z['fixed']}/{z['broken']}。")
    lines+=['以上区间按同一验证区的空间块共同重抽，重复情景不会被当成独立地理块；但没有消除调参选择偏差，所以仍不能作为独立提升证明。',
        '', '三、事先规则与判定',json.dumps(result['verdict'],ensure_ascii=False,indent=2),
        'ISAB须同时通过对原LN和两个简单对照的筛查，单独胜过原LN不足以归功于参照交互。',
        '', '四、机制诊断边界',
        '旧LN重放显示参照较少的题准确率较低；这只提示难题分布，没有证明原因是集合交互不足。按真实学校参照质量分组仅用于开发诊断，不是模型可偷看的选择特征。',
        json.dumps(diagnostic_comparisons,ensure_ascii=False,indent=2),
        '训练后关闭残差的差值另存完整性与诊断.json。它只描述已训练模型对模块的依赖，不能替代重新训练对照。',
        '', '五、核对与限制',
        f"128次拟合的设置/历史/权重和预测哈希均核对；32个所选权重重新推理，最大概率误差 {replay['max_error']:.9g}。原LN全部32组训练曲线与归档重叠检查点的准确率完全一致。",
        '160步上限所选情景数：'+', '.join(f'{names[k]} {caps[k]}/8' for k in run.METHODS)+'。到上限的配置不能据此认为已经收敛。',
        '本轮只测一个ISAB结构、一个统计残差结构和一个容量对照；没有证明所有交互方法无效。',
        '本轮未重新比较GBT/RetrievalScorer，不能更新整个主表最强模型排名。pre_v1数据反复用于开发，杭州答案未冻结，独立地区/时间验证仍未完成。',
        '实际训练时间：'+', '.join(f'{names[k]} {runtime[k]["total_seconds"]/60:.1f}分钟' for k in run.METHODS)+'。',
        '复现见研究说明.txt；训练前规则见方案.json；逐情景分数见验证结果.json；权重/逐题概率在独立outputs目录。']
    if complete['target_evaluated']:
        lines.insert(7,'验证筛查后，按冻结计划追加四种方法×八区×四批×三个种子的384次拟合；本轮合计512次新训练。只有逐条MLP晋级，其余两个版本保留为固定对照，不能按目标结果重新选择。')
        lines+=['','六、冻结后的目标评价',
            (run.DOC/'目标判定.json').read_text(encoding='utf8'),
            'K0/K1、分城及规则+模型分数见目标结果.json；这仍是重复使用过的开发数据，不是新盲测。']
        target=json.loads((run.DOC/'目标结果.json').read_text(encoding='utf8'))
        for batch,z in target.items():
            lines.append(batch+' 模型准确率：'+', '.join(f"{names[m]} {z['model'][m]['两城']['accuracy']:.3f}%" for m in run.METHODS))
            lines.append(batch+' 规则+模型（剔除核验参照）：'+', '.join(f"{names[m]} {z['overall_excluding_verified'][m]['两城']['accuracy']:.3f}%" for m in run.METHODS))
            cc=z['contrasts']['rowmlp-identity']['两城']
            lines.append(f"逐条MLP减原LN：{cc['delta']:+.3f}点，99%配对空间块区间 {cc['ci']}，改对/误伤 {cc['fixed']}/{cc['broken']}。")
        ta=run.DOC/'目标完整性审计.json'
        if ta.exists():
            z=json.loads(ta.read_text(encoding='utf8'))
            lines.append(f"目标产物审计：{z['artifacts']}组单种子权重/预测/设置核对，抽查重放{z['replayed']}组，最大概率误差{z['max_error']:.9g}；逐种子结果见目标完整性审计.json。")
    (run.DOC/'研究判断.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines[:20]))


if __name__=='__main__':main()
