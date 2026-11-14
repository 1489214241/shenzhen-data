"""Write the completed pilot report without relabeling diagnostic slices as gains."""
import json
import collections
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'outputs/shenzhen_membership_trees_20261114'
DEST=REPO.parent/'科研推进/深圳整体方法核查与学校成员预测_20261114'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    r=read(OUT/'results.json');audit=read(OUT/'audit.json')
    assert audit['passed'] and read(OUT/'independent_local_feature_audit.json')['passed']
    coverage=read(OUT/'reference_coverage_diagnostic_only.json')['groups']
    inventory=read(DEST/'逐校输入缺口_不使用评价答案.json')
    schools={(s['district'],s['school_id']):s for s in inventory['schools']}
    bins=collections.defaultdict(list)
    for row in read(OUT/'scored_rows/baseline_current.json'):
        kinds=[]
        for sid in row['truth']:
            refs=[a for a in schools[row['d'],sid]['references'] if a['site_id']!=row['site_id']]
            kinds.append((bool(refs),any(a['kind']=='textual_proof' for a in refs)))
        key='all_textual' if all(k[1] for k in kinds) else 'all_have_ref_some_estimated_only' if all(k[0] for k in kinds) else 'missing_at_least_one_school_reference'
        bins[key].append(row)
    replay={k:dict(n=len(v),correct=sum(a['correct'] for a in v),multi=sum(a['size']>1 for a in v)) for k,v in bins.items()}
    assert replay==coverage and sum(v['n'] for v in replay.values())==1761
    text='''深圳整体方法核查与学校成员预测实验（2026-11-14）

结论：尚未把深圳整体准确率提高到70%。本轮已完成全学校成员预测对照，结果有害，不采用。
不改默认模型，不改Gold，不删难题；之前道路试验917/1761=52.07%仍是未验证别名条件下的开发候选。
本轮基线为当前Modern补校流程912/1761=51.79%；历史其他流程曾有959/1761=54.46%的开发候选，不能把不同版本混作同条件提升。

一、杭州的70%是什么
重新读取杭州现用历史RetrievalScorer输出：1692/2272=74.47%。单校1690/2131=79.31%，多校2/141=1.42%。
当前深圳Modern补校流程：912/1761=51.79%。单校569/803=70.86%，多校343/958=35.80%。
两者均按完整学校集合完全相等计分，但不是同模型、同训练条件的城市对照。杭州78.43%的旧数字是命中任意学校，不能混用。
杭州多校占6.21%，深圳占54.40%；不能用杭州的总体高分证明它已经会预测多校。
深圳单校仍比杭州这一版本低8.45个百分点，也需要修。若仅作样本比例的算术重加权，深圳在杭州单/多校比例下为68.68%；这不是新性能、也不是因果解释。

二、现模型到底错在哪里
深圳849条错误：只漏学校361；预测集合与答案完全不相交361；只多报45；既漏又多报82。
因此488条包含错误学校，仅靠A校命中后补B校不可能修完这些错误。首校排序、范围适用性与多校输出需要一起处理。
当前源训练的完整多校样本94条，全为双校；来源验证只有60条双校。深圳有364条三校以上，但没有完整深圳集合监督进入本次训练。
这是一项已观察到的监督差异，不证明单凭补多校标签即可解决。

三、本轮实际做的模型试验
使用已有Modern三教师集成概率、学校证据与参照证据，对每个小区的全部学校分别预测成员概率；不锁死原首校，不截断Top10，不提供正确学校数量。
mean组沿用139维证据；local组另加每校最近两个合法参照记录的完整关系特征、距离、计数、可用性和排名，共235维。
同一小区组的参照排除；不完整标签中的未知项不作负例；两组均恢复首校为普通可监督候选。
模型为项目已有梯度提升树家族的工程试验，并非新提出的架构，也不是XGBoost论文复现。
以前第11/14/15轮已试过树模型且迁移不理想；本轮的区别是更新输入、所有学校直接成员判定、原首校可被替换以及最近两个参照特征。没有把旧方法包装成首次尝试。
两组固定300轮、31叶子、学习率0.05、L2=1、seed=20261114。每组训练一次，未做多训练种子稳定性验证。
阈值只从来源1246条验证选：balanced平衡单/多校集合准确率；protected要求来源单校不比教师低超过0.5个百分点。
两种阈值分别输出裸预测、仅文字证明约束、原政策约束，共12组；所有目标预测封存后才读评分答案。
预定主方案local_balanced_legacy在来源已不满足单校保护，仍依协议报告目标诊断结果；不能称为通过保护的方案。

四、1761条原固定题的完整结果
方案 | 正确数 | 准确率 | 单校正确/803 | 多校正确/958 | 龙华正确/627 | 龙岗正确/1134 | 改对 | 误伤
'''
    for arm,m in r['metrics'].items():
        c=r['comparisons'].get(arm,{}).get('all',{})
        text+=f"{arm} | {m['all']['correct']} | {m['all']['exact']:.2f}% | {m['single']['correct']} | {m['multi']['correct']} | {m['龙华区']['correct']} | {m['龙岗区']['correct']} | {c.get('corrected','-')} | {c.get('harmed','-')}\n"
    text+='\n主要对比及95%空间块配对区间（百分点）：\n'
    for arm in ['local_balanced_legacy','local_protected_legacy','mean_protected_legacy']:
        c=r['comparisons'][arm]['all'];text+=f"{arm}: {c['delta_pp']:.3f}, {c['ci95']}\n"
    text+='''
主方案纠正57条、误伤274条，净减217条。召回率64.74%→71.56%，但错报关系724→1645，完整集合准确率反而51.79%→39.47%。
因此“让模型多报学校”没有解决适用性判断，新增假阳性代价更大。来源保护也不能保证迁移后深圳受保护，protected两组仍退步。
这证明本轮设置有害，不证明所有树模型、多校学习或补参照方式都无效。

五、输入端的可执行缺口
现有参照648条，覆盖109/202校；其中391条有文字证据、257条为估计，不能全部叫核验事实。
龙华：71校，40校无参照，50校无文字证据参照。
龙岗：131校，53校无参照，61校无文字证据参照。
合计93校没有参照，18校只有估计参照，91校具有至少一条文字证据参照。
全量逐校原文、官方URL、表格行/页、已有参照及缺口已保存到“逐校输入缺口与官方出处.txt”和JSON。
排序仅按无参照/无文字参照/校名字典序，不读取评价答案选择需要补的学校；未冒称本轮新增了核验事实。

六、参照覆盖与当前错误的关系（仅评分后诊断）
'''
    names={'all_textual':'正确学校均有文字证据参照','all_have_ref_some_estimated_only':'正确学校均有参照，但部分只有估计','missing_at_least_one_school_reference':'至少一所正确学校没有参照'}
    for k,v in coverage.items():text+=f"{names[k]}：{v['correct']}/{v['n']}={100*v['correct']/v['n']:.2f}%；其中多校样本{v['multi']}条。\n"
    text+='''这里按答案学校分组、排除了查询自身参照，属于事后条件统计；不能用它挑评价题、路由上线模型或宣称补齐后整体必到75%。
缺参照与复杂性等因素会混杂。它支持优先检验覆盖和证据质量，不构成因果提升证明。

七、适合应用论文的下一步方法与边界
方法组合应同时覆盖“事实接入—范围执行—参照推断—集合输出”，本次否定了直接用全局树模型替换输出流程的这一个候选。
1. 事实接入：按逐校缺口建立核验台账，优先处理无参照学校。保留原每校K=0/1/3/5、原seed=1/2/3及references.pick/apply规则；同K方法使用完全相同参照、监督和评价题。不得用每区32个参照悄悄替代。
2. 范围执行：社区成员关系使用能证明行政归属的官方名单/边界；道路范围核对实体类型、别名、连通与排除作用域。社区办公点和学校点只做位置线索。按官方条款构造A且B、A或B、A减B，而不是只增加一个距离特征。
3. 学校集合：官方明确同一完整基础范围的学校可作为同范围候选一起判定；同名社区、共享关系、部分交叠都不能直接变成全范围等价。局部共享必须先判定该小区落在共享分支；大学区保留原地段。禁止传递闭包A→B→C自动扩校。
4. 本地适配：先验证原K协议下新增单校参照对首校排序与漏校的收益。原K候选只允许整组单校，因此它本身不提供完整多校监督；不能假装已经解决三校以上学习。若单列完整集合核验扩展，必须另立协议，明确核验成本、完整性证明、查询退出和独立评价，不混进原K结果。该扩展本轮未执行，也未把保留答案转为训练标签。
5. 证据不足的案例保留不确定与核验任务。应用系统可以报告自动处理覆盖率和人工成本，但论文主表仍保留全部原题，不得把拒判样本删掉后称整体准确率70%。

研究贡献应围绕这些可检验的应用环节：同预算参照是否提升覆盖，文字作用域是否减少错扩校，完整集合监督是否改善三校以上，处理成本是否下降。
每一步独立消融，避免将补数据、改模型、换评价集混在一起。当前结果仍来自反复使用的开发数据，不能作为独立确认。

八、审核与复现
原Gold哈希未改；1802条全部输出，1761条评分，41条地图外保留输出，242条语义待核仍计分并单列。
20个来源/目标概率数组逐项重放、来源选阈值重放、全部正确数及改对误伤复核通过。
两个目标区的最近参照特征从原输入重新计算并逐元素一致；针对未知标签、首校可替换、同组参照排除的3项测试通过。
一次环境兼容修复：threadpoolctl2.2.0查询OpenBLAS版本失败，首次训练尚未运行；改用进程环境变量限制线程，无全局升级。保存修复前源码、原因及哈希收据，未改变数据或参数。
单次树训练和已反复使用的验证集限制了外推；来源训练特征含原教师训练内分数，本轮不是严格折外stacking确认。
adopted=false；independent_validation=false。没有上线，也没有达到70%的新结果。

实验制品：学区归属模型_代码库/outputs/shenzhen_membership_trees_20261114/
入口：scripts/experiment_shenzhen_school_membership_trees.py（prepare、fit、infer、score、audit）。已有封存结果拒绝覆盖。
逐校核查与特征复验：scripts/report_shenzhen_membership_tree_inputs.py。
报告：scripts/report_shenzhen_membership_trees.py。
'''
    DEST.mkdir(parents=True,exist_ok=True)
    p=DEST/'实验结论与下一步方法.txt';p.write_text(text,encoding='utf-8')
    paths=[p,DEST/'逐校输入缺口与官方出处.txt',DEST/'逐校输入缺口_不使用评价答案.json',
           OUT/'results.json',OUT/'audit.json',OUT/'protocol.json',OUT/'reference_coverage_diagnostic_only.json',
           OUT/'independent_local_feature_audit.json',OUT/'city_comparison_diagnostic_only.json',Path(__file__)]
    (DEST/'交付文件哈希.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},ensure_ascii=False,indent=2),encoding='utf-8')
    print(p)


if __name__=='__main__':main()
