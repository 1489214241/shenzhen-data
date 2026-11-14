"""Report every frozen normalization variant, including negative results."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
from run_modernnca_normalization import DOC, OUT, OLD
from run_modernnca_relation import save

NAMES={'modernnca':'ModernNCA独立全证据适配基线','reference':'现有RetrievalScorer',
       'layernorm':'ModernNCA仅改LayerNorm','pooled_bn':'ModernNCA固定权重重新校准BatchNorm'}
BATCHES=['k0_s1','k1_s1','k1_s2','k1_s3']


def report():
    results=json.loads((DOC/'结果.json').read_text(encoding='utf8'))
    verdict=json.loads((DOC/'判定.json').read_text(encoding='utf8'))
    audit=json.loads((DOC/'完整性审计.json').read_text(encoding='utf8'))
    assert audit['pass_'] and audit['complete']
    diagnosis=json.loads((DOC/'诊断.json').read_text(encoding='utf8'))
    bn=json.loads((DOC/'归一化诊断.json').read_text(encoding='utf8'))
    records=json.loads((DOC/'诊断逐题.json').read_text(encoding='utf8'))
    lines=['ModernNCA机制诊断与归一化改进：全部结果','',
           '批次顺序：K=0 / K=1第一、二、三批。单位：准确率百分比；差值为百分点。',
           'K是补充核验参照预算，不是标准K-shot。所有结果都是pre_v1开发结果。','',
           '模型部分（规则未判定，保留所有候选）']
    for method,name in NAMES.items():
        lines.append(name+'：'+' / '.join(f"{results[b]['model'][method]['两城']['accuracy']:.4f}" for b in BATCHES))
    lines+=['','规则+模型（排除人工核验后成为参照的小区）']
    for method,name in NAMES.items():
        lines.append(name+'：'+' / '.join(f"{results[b]['overall_excluding_verified'][method]['两城']['accuracy']:.4f}" for b in BATCHES))
    lines+=['规则多校输出沿用命中任一正确学校的历史口径；严格集合准确率另见结果.json。','']
    for b in BATCHES:
        lines.append(b)
        for method in NAMES:
            for city in ['长沙','杭州']:
                row=results[b]['model'][method][city]
                lines.append(f"  {NAMES[method]} {city}：{row['accuracy']:.4f}，n={row['n']}")
        for pair,comparison in results[b]['comparisons'].items():
            lines.append('  '+pair)
            for city in ['两城','长沙','杭州']:
                row=comparison[city]
                lines.append(f"    {city}：{row['delta']:+.4f}，95%区间[{row['ci'][0]:+.4f},{row['ci'][1]:+.4f}]，改对/改坏={row['fixed']}/{row['broken']}")
            lo,hi=comparison['adjusted_ci']
            lines.append(f'    两城四对比较校正后98.75%区间：[{lo:+.4f},{hi:+.4f}]')
        lines.append('')
    lines+=['事先判定（K=1三批全部校正区间下界>0，且无分城确认有害）']
    lines += [f"{pair}：{'达到开发集稳定提升标准' if value['confirmed'] else '未达到开发集稳定提升标准'}" for pair,value in verdict.items()]
    lines+=['','错误机制诊断（K0与K1第一批；唯一真校题；以下是三个种子的题次之和，非独立样本数）']
    for b,ds in diagnosis['counts'].items():
        counts=sum((Counter(v) for v in ds.values()),Counter())
        lines.append(b+'：'+json.dumps(dict(counts),ensure_ascii=False))
        grouped=defaultdict(list)
        for r in records:
            if r['batch']==b: grouped[(r['district'],r['q'])].append(r)
        allwrong=[rr for rr in grouped.values() if all(r['pred']!=r['truth'] for r in rr)]
        lines.append(f'  唯一题数={len(grouped)}；三个种子全错={len(allwrong)}；其中全都有真校参照={sum(all(r["true_reference"] for r in rr) for rr in allwrong)}。')
    n=sum(v['n'] for v in bn.values())
    switches=sum(v['forward_reverse_prediction_changes'] for v in bn.values())
    lines += ['','BatchNorm固定权重诊断：',
        f'8个验证设置共{n}题次；仅改变BN统计量更新的区顺序，{switches}题次的预测不同。部分设置使用同一个验证区，不可视为独立样本。',
        '重估统计量会在一些区改善、一些区变差；不能把所有错误归因于BatchNorm。',
        '已核对官方距离读出、完整输入、掩码及预测回放；已有基线选中步数未卡在1280上限。并未穷尽所有优化器和超参数。',
        '', '验证选参（LayerNorm与原基线各四配置）']
    for f in sorted((OUT/'selection').glob('*_layernorm.json')):
        z=json.loads(f.read_text(encoding='utf8')); d=f.stem.rsplit('_',1)[0]
        old=json.loads((OLD/'selection'/f'{d}_modernnca.json').read_text(encoding='utf8'))
        lines.append(f"  {d}，验证={z['validation']}，原基线={100*old['best']['accuracy']:.4f}，LayerNorm={100*z['best']['accuracy']:.4f}，步数={z['steps']}，配置={z['hp']}")
    lines+=['','审计：',f"64组集成预测、192组种子预测/权重完整；回放最大误差={max(v['replay_max_error'] for v in audit['predictions'].values()):.9g}。",
        '其中96次是LayerNorm最终训练，另外96组是原ModernNCA权重的BN校准，不是额外梯度训练。',
        '校准只读既有合法训练题输入；校准标签不参与计算，训练参数逐项不变；全部参照和候选保留。',
        '默认模型、正式数据、历史基线结果均未替换。']
    (DOC/'结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    short=['ModernNCA归一化改进：结论','',
        '起点是独立全证据ModernNCA任务适配版。原有RetrievalScorer不是ModernNCA改进版。',
        '本轮先发现固定权重下BN统计会受区顺序影响，再比较只换LayerNorm与只重估BN统计的两个最小改动。',
        '距离学习、学校锚点、全部参照/关系/学校证据、训练标签和候选未变。','']
    for method in ['modernnca','layernorm','pooled_bn','reference']:
        short.append(NAMES[method]+'，模型准确率：'+' / '.join(f"{results[b]['model'][method]['两城']['accuracy']:.2f}" for b in BATCHES))
    short+=['']
    for pair in verdict:
        rr=[results[b]['comparisons'][pair]['两城'] for b in BATCHES[1:]]
        short.append(pair+'，K=1差值：'+' / '.join(f"{r['delta']:+.2f}" for r in rr)+'个百分点；'+('达到事先开发判定。' if verdict[pair]['confirmed'] else '未达到事先稳定提升判定。'))
    short+=['','这轮属于适配/训练机制排查与修正，不构成新方法创新性证明。',
        '若区间跨零，只能写未确认差异，不能写等效；负结果也不能推为所有ModernNCA改进无效。',
        '两城已反复用于开发；杭州标签未正式定版；少量评价点与政策训练点同楼盘组。需要未参与开发的地区或时段验证后才能提出正式泛化结论。',
        '完整数字、分城损益、校正区间和审计见结果.txt、结果.json、完整性审计.json。']
    (DOC/'结论.txt').write_text('\n'.join(short)+'\n',encoding='utf8')
    print('\n'.join(short),flush=True)


if __name__=='__main__': report()
