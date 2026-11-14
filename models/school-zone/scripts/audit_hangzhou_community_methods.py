"""Independent frozen-output counts, feature isolation checks and readable report."""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from zoning import data
from experiment_hangzhou_community_methods import OUT,ROOT,HZ,ARMS,read,write,load,sha,verify_hashes


def main():
    verify_hashes(OUT/'input_hashes.json',ROOT)
    verify_hashes(OUT/'prepared_hashes.json',OUT)
    verify_hashes(OUT/'weights_hashes.json',OUT)
    verify_hashes(OUT/'prediction_hashes.json',OUT)
    result=read(OUT/'results.json');counts={};checks=0
    for d in HZ:
        x=data.with_list_gold(data.load(d));base=load(OUT/'arms/hangzhou_E'/f'{d}.pkl')
        cs=load(OUT/'arms/changsha_E'/f'{d}.pkl')
        assert base[1].known==cs[1].known
        for key in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            assert np.array_equal(getattr(base[2],key),getattr(cs[2],key)),(d,key)
        for name,sl in base[2].pair_blocks.items():
            if name!='same_community':assert np.array_equal(base[2].pair[...,sl],cs[2].pair[...,sl])
        for name,sl in base[2].anchor_blocks.items():
            if name!='community_clause':assert np.array_equal(base[2].anchor[...,sl],cs[2].anchor[...,sl])
        for arm in ARMS:
            rr=read(OUT/'predictions'/arm/f'{d}.json')
            _,rules,features=load(OUT/'arms'/arm/f'{d}.pkl')
            archive=np.load(OUT/'predictions'/arm/f'{d}.npz')
            probs=archive['probs'];seeds=['ensemble']+[str(s) for s in archive['seeds']]
            assert len(rr)==x.n*len(seeds)
            for i,seed in enumerate(seeds):
                items=[r for r in rr if r['seed']==seed]
                assert [r['q'] for r in items]==list(range(x.n))
                p=probs.mean(0) if i==0 else probs[i-1]
                total=0
                for r in items:
                    q=r['q'];pred=r['prediction']
                    assert len(pred)==len(set(pred)) and all(0<=s<x.school_n for s in pred)
                    assert pred==rules.known.get(q,[int(p[q].argmax())])
                    if q in x.answers:total+=set(pred)==set(x.answers[q])
                counts.setdefault(arm,{}).setdefault(seed,0);counts[arm][seed]+=total
                if seed=='ensemble':assert total==result['metrics'][arm][d]['correct']
                checks+=len(items)
    for arm in ARMS:
        for seed,total in counts[arm].items():assert total==result['per_seed'][arm][seed]['correct']
    historical=read(REPO/'outputs/hz_comm_ER/eval.json')['overall']['杭州']
    current=result['metrics']['hangzhou_ER']['杭州']
    assert historical['n']==current['n']
    assert abs(historical['strict']-current['historical_scoped_strict'])<1e-8
    assert abs(historical['accuracy']-current['historical_any_hit'])<1e-8
    changes={}
    for key in result['comparisons']:
        a,b=key.split(' minus ');ra=read(OUT/'scored_rows'/f'{a}.json');rb=read(OUT/'scored_rows'/f'{b}.json')
        delta=[int(x['prediction']==x['truth'])-int(y['prediction']==y['truth']) for x,y in zip(ra,rb)]
        expect=result['comparisons'][key]['杭州']
        assert delta.count(1)==expect['corrected'] and delta.count(-1)==expect['harmed']
        changes[key]={}
        for by in ['community','model','other_rule']:
            ix=[i for i,r in enumerate(ra) if (r['by'] if r['by'] in ('community','model') else 'other_rule')==by]
            changes[key][by]=dict(n=len(ix),corrected=sum(delta[i]>0 for i in ix),harmed=sum(delta[i]<0 for i in ix))
    audit=dict(all_checks_passed=True,prediction_rows_checked=checks,seed_totals=counts,changes_by_new_arm_output=changes,
               note='Recomputed from frozen predictions and original Gold. E arms checked to differ only in community feature columns.')
    write(OUT/'independent_audit.json',audit)
    inputs=read(OUT/'input_audit.json');reconstruction=read(OUT/'baseline_reconstruction.json')
    labels={'hangzhou_E':'杭州方式：只作特征','changsha_E':'长沙方式：只作特征','hangzhou_ER':'杭州方式：特征＋规则参照（现用）',
            'changsha_ER':'长沙方式：特征＋规则参照','changsha_keep_current_R':'只换长沙特征，保留现用杭州规则参照'}
    lines=['杭州四区社区处理方式对照，2026-11-13','',
        '本轮结论：现用杭州模型下，长沙方式更适合作为特征替换候选；不支持将社区直接规则整体换成长沙方式。',
        '保留现用杭州规则参照，只换长沙社区特征：1692/2272→1712/2272，74.47%→75.35%；纠正33、损害13；差值95%空间区间[+0.18,+1.64]个百分点。',
        '这一候选的分区净变化为拱墅-2、萧山+19、富阳0、临安+3，主要提升来自萧山。十个种子的城市净变化均为正。',
        '在不含社区规则的共同参照库下，长沙特征比杭州特征高0.88个百分点，但区间[-0.14,+2.01]跨零，尚不能确认稳定提升。',
        '连同社区直接规则整体替换时，长沙比杭州仅高0.18个百分点，区间[-0.97,+1.32]跨零，且拱墅、富阳下降。',
        '以上为同批杭州开发数据上的固定权重对照；最优分数候选未作独立验证，未自动修改默认配置，也不证明长沙方法普遍优于杭州方法。','',
        '这是现用杭州模型固定权重下的输入替换实验，不是两种流程分别重训后的最佳成绩比较。',
        '重建现用模型的十个原始种子，每个目标区只使用本区规则标签及其余七区合法监督；本区官方答案不进训练。',
        '固定相同小区、学校、政策文本、MCA社区/村点及EPSG:32651坐标。长沙方式增加道路阻隔筛选；杭州方式用最近两个不同社区的距离比。',
        'E：共同的政策参照库，关闭社区直接判定，只换社区特征。ER：共同政策参照＋各方式自己的社区规则，并重建相关参照及分界特征。',
        '另有保留现用杭州规则参照、只换长沙特征的诊断对照。模型权重源自现用杭州ER，结论仅针对这组现用模型。',
        '主指标：固定2272个有官方答案的小区，预测完整学校集合与Gold完全一致，不截断到有地图学校。历史scoped strict和任意命中指标另列。','',
        '完整集合正确率：','方式 | 杭州 | 拱墅 | 萧山 | 富阳 | 临安']
    for arm in ARMS:
        row=[labels[arm]]
        for g in ['杭州']+HZ:
            m=result['metrics'][arm][g];row.append(f"{m['correct']}/{m['n']}（{m['accuracy']:.2f}%）")
        lines.append(' | '.join(row))
    lines+=['','配对比较（差值为前者减后者；按区分层、1公里空间块重抽3000次）：']
    for key,v in result['comparisons'].items():
        a,b=key.split(' minus ');m=v['杭州'];ci=m['ci95']
        seed_delta=[counts[a][s]-counts[b][s] for s in counts[a] if s!='ensemble']
        lines.append(f"{labels[a]} 对 {labels[b]}：{m['delta_pp']:+.2f}个百分点；纠正{m['corrected']}，损害{m['harmed']}；95%区间[{ci[0]:+.2f}, {ci[1]:+.2f}]；十种子正确数差{seed_delta}。")
    lines+=['','社区定位与直接规则覆盖：','区 | 杭州严格/宽松 | 长沙严格/宽松 | 杭州新增规则 | 长沙新增规则']
    for d,v in inputs.items():
        lines.append(f"{d} | {v['hangzhou_strict']}/{v['hangzhou_loose']} | {v['changsha_strict']}/{v['changsha_loose']} | {v['hangzhou_new_rules']} | {v['changsha_new_rules']}")
    lines+=['','新增社区规则与Gold的一致性（仅评分后诊断，不用于挑选输入）：']
    for arm,dd in result['direct_rule_postscore_audit'].items():
        lines.append(labels[arm]+'：'+'；'.join(f"{d} {v['consistent']}/{v['scored']}（全部规则{v['all_rules']}）" for d,v in dd.items()))
    lines+=['','历史评价口径（不能与上述完整集合指标混用）：']
    for arm in ARMS:
        m=result['metrics'][arm]['杭州'];lines.append(f"{labels[arm]}：历史scoped strict {m['historical_scoped_strict']:.2f}%，历史任意命中 {m['historical_any_hit']:.2f}%。")
    lines+=['','复现及完整性：',json.dumps(reconstruction,ensure_ascii=False),f'已逐条核对{checks}条预测输出、全部种子计数、固定输入/权重/预测哈希；所有推断重复运行逐位相同。',
        '复现原有杭州社区特征逐位相同；E对照除了社区六列特征外其他输入逐位相同。',
        '原始default.json、community_mca.npz、Gold及历史结果均未修改。',
        '局限：固定权重继承现用杭州社区规则的训练影响；此实验回答现用模型换社区处理是否更好，不证明方法在重新训练或独立地区上普遍更优。坐标点是代理证据，并非官方行政边界。',
        '运行：D:/ana/python.exe -X utf8 scripts/experiment_hangzhou_community_methods.py prepare（然后train、infer、score）；最后运行scripts/audit_hangzhou_community_methods.py。',
        '所有结果文件在本报告同目录；results.json含分区、单/多校、地图/名单、逐种子和空间区间；independent_audit.json含直接规则/模型传播的纠正损害分解。']
    (OUT/'杭州两种社区方式对照.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=1))


if __name__=='__main__':main()
