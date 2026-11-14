"""Write a readable report from sealed scores, without choosing new settings."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from run_full_evidence import DOC, OUT, METHODS, BATCHES

names = dict(reference='我们的 RetrievalScorer（本轮重训）', matching='Matching 关系输入适配版',
             modernnca='ModernNCA 关系输入适配版（独立模型）', proto='ProtoNet 关系输入适配版', deepsets='DeepSets 关系输入适配版')
results = json.loads((DOC / '结果.json').read_text(encoding='utf8'))
verdict = json.loads((DOC / '判定.json').read_text(encoding='utf8'))
audit = json.loads((DOC / '完整性审计.json').read_text(encoding='utf8'))
assert audit['pass'] and audit['complete']
lines = ['完整同信息的参照模型比较：开发实验结果', '',
         '范围：长沙四区＋杭州四区；同一冻结数据、全部合法参照、完整逐参照和学校证据、全部学校候选、相同训练题与标签。',
         '各对照均为明确标注的任务适配版。ModernNCA 是独立距离匹配网络，没有混入 RetrievalScorer 的分数或权重。', '',
         '模型部分准确率（%，K=0 / K=1第一批 / 第二批 / 第三批）']
for m in METHODS:
    vals = [results[f'k{k}_s{s}']['model'][m]['两城']['accuracy'] for k,s in BATCHES]
    lines.append(names[m] + '：' + ' / '.join(f'{v:.2f}' for v in vals))
lines += ['', '规则＋模型准确率（排除人工核验参照；%，同上顺序）']
for m in METHODS:
    vals = [results[f'k{k}_s{s}']['overall_excluding_verified'][m]['两城']['accuracy'] for k,s in BATCHES]
    lines.append(names[m] + '：' + ' / '.join(f'{v:.2f}' for v in vals))
lines += ['', '逐对照差值：对照减我们的本轮重训模型（百分点）']
for m in METHODS[1:]:
    lines.append(names[m])
    for k, s in BATCHES:
        z = results[f'k{k}_s{s}']; a = z['comparisons'][m]['两城']; ci = z['familywise_ci'][m]['两城']
        lines.append(f"  K={k} 批{s}：{a['delta']:+.2f}；95%区间[{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}]；四方法校正区间[{ci[0]:+.2f}, {ci[1]:+.2f}]；改对/改坏 {a['fixed']}/{a['broken']}")
        for city in ['长沙', '杭州']:
            b = z['comparisons'][m][city]
            lines.append(f"    {city}：{b['delta']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]")
    lines.append('  K=1三批稳定超过的事先判定：' + ('通过' if verdict[m]['confirmed'] else '未通过'))
lines += ['', '严格集合准确率（%，模型部分，K=0 / K=1三批）']
for m in METHODS:
    vals = [results[f'k{k}_s{s}']['model'][m]['两城']['strict'] for k,s in BATCHES]
    lines.append(names[m] + '：' + ' / '.join(f'{v:.2f}' for v in vals))
lines += ['', '已通过审计',
          '四批输入哈希、训练题和标签、完整参照列表、全部学校候选、输入打包可逆性、目标评价题不进入训练、所有预测形状/归一化/种子集成/文件哈希、评分题与冻结答案一致。',
          '机制测试：参照置换不变、学校编号置换等变、查询标签不影响预测、屏蔽参照无影响、每一列完整关系与学校输入获得梯度、ModernNCA读出与官方核心数值一致。', '',
          '限制与解释',
          '1. 这是真正相同信息来源且不预先删掉参照/候选的比较；四个外部方法仍需关系输入适配和学校锚点，不能把成绩写成论文原版的结论。共享输入编码器架构是本轮控制条件，并非穷尽每种方法的最优表达。',
          '2. 每方法四组学习率/正则配置，验证区选步数，三个最终训练种子。RetrievalScorer保留100步本区线性热启动；对照独立初始化。选择规则一致，总计算量不完全相等。',
          '3. K=0、K=1各批评价数量不同；批间准确率不能作为相同题上的因果提升。K=1只补充每校核验预算，规则参照一直存在。',
          '4. 原主表多校规则命中任一正确学校即算对；另有严格集合指标。规则合并成绩不包含人工核验参照的已知答案。',
          '5. 目标评价题ID没有进入训练，但少数评价点与政策已知训练点同属楼盘组；每题同楼盘参照被屏蔽，不能将本实验称作完全按楼盘隔离的独立测试。逐区组数见完整性审计。',
          '6. 全部地区已用于开发，杭州答案未正式定版。输入快照冻结不等于独立测试集。结果不能推出任何原论文方法在所有设置下都不如我们。']
caps = [n for n, a in audit['selection'].items() if a['at_hard_cap']]
lines.append('7. 选择落在1280步硬上限的配置：' + ('、'.join(caps) if caps else '无。') + ' 不据此声称所有配置达到理论最优。')
lines += ['', '原始数字、逐种子、逐区结果见 结果.json；事先方案见 方案.json/说明.txt；全部模型与预测在 outputs/full_evidence_20261112。']
(DOC / '结果.txt').write_text('\n'.join(lines), encoding='utf8')
print('\n'.join(lines[:20]))
