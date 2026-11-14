"""Human-readable report and descriptive error-category audit; no model selection."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_targeted as run
from run_modernnca_relation import digest,save

NAMES={'layernorm':'ModernNCA-LN基线','reference':'RetrievalScorer独立对照',
       'schoolquery':'学校条件化查询','adaptivegate':'输入相关融合','constantgate':'固定系数融合'}

def ci(v): return '['+', '.join(f'{x:+.2f}' for x in v)+']'


def main():
    result=json.loads((run.DOC/'结果.json').read_text(encoding='utf8'))
    verdict=json.loads((run.DOC/'判定.json').read_text(encoding='utf8'))
    audit=json.loads((run.DOC/'完整性审计.json').read_text(encoding='utf8'))
    assert audit['complete'] and audit['pass_']
    records=json.loads((run.DOC/'基线诊断逐题.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for r in records:groups[(r['batch'],r['district'],r['q'])].append(r)
    descriptive={}
    for m in run.METHODS:
        cache={}; counts={}; availability={}
        for r in records:
            key=(r['batch'],r['district'])
            if key not in cache: cache[key]=np.load(run.OUT/r['batch']/m/'probs'/f"{r['district']}.npz")['probs']
            j=run.base.SEEDS.index(r['seed']); pred=int(cache[key][j,r['q']].argmax())
            old=r['pred']==r['truth']; new=pred==r['truth']
            ckey=f"{r['batch']}/{r['city']}/{r['category']}"
            co=counts.setdefault(ckey,Counter()); co['n']+=1
            co['fixed']+=int(new and not old); co['broken']+=int(old and not new)
            co['changed_prediction']+=int(pred!=r['pred'])
            skey=f"{r['batch']}/{r['city']}/true_reference={r['true_reference']}"
            sub=availability.setdefault(skey,Counter());sub['n']+=1
            sub['baseline_correct']+=int(old);sub['candidate_correct']+=int(new)
            sub['fixed']+=int(new and not old);sub['broken']+=int(old and not new)
        stable={}; ensemble={key:value.mean(0).argmax(-1) for key,value in cache.items()}
        for (batch,d,q),rs in groups.items():
            assert len(rs)==3
            if len({r['pred'] for r in rs})!=1 or len({r['category'] for r in rs})!=1:continue
            truth=rs[0]['truth'];before=rs[0]['pred']==truth
            after=int(ensemble[(batch,d)][q])==truth
            for city in ['两城',rs[0]['city']]:
                key=f"{batch}/{city}/{rs[0]['category']}";co=stable.setdefault(key,Counter())
                co['unique_queries']+=1;co['candidate_ensemble_correct']+=int(after)
                co['fixed']+=int(after and not before);co['broken']+=int(before and not after)
        descriptive[m]=dict(error_category={k:dict(v) for k,v in counts.items()},
            reference_availability={k:dict(v) for k,v in availability.items()},stable_subgroups={k:dict(v) for k,v in stable.items()})
    save(run.DOC/'错误类型变化.json',dict(note='Descriptive only. error_category/reference_availability count seed-cases; stable_subgroups counts unique queries where all three baseline seeds predicted the same school in the same diagnostic category, evaluated with the candidate probability-averaged ensemble. K0_s1/K1_s1 unique-truth cases only. No causal proof or independent validation.',data=descriptive,source_sha256=digest(Path(__file__))))
    lines=['ModernNCA-LN针对性改进：全部结果','',
        '范围：长沙、杭州共8区；K0一批，K1三批；每格为3个训练种子概率平均后评分。K是补充核验参照预算，不是总参照数。',
        '输入、监督、候选、验证规则相同。下面先列纯模型准确率，再列规则+模型；两者分母不同。',
        '区间：普通95%空间块重抽；同时列出7比较校正区间(99.2857%)。单位均为百分点。','']
    for batch,res in result.items():
        lines += [batch]
        for m in ['layernorm','reference']+run.METHODS:
            model=res['model'][m]; whole=res['overall_excluding_verified'][m]
            v=model['两城']; w=whole['两城']
            lines += [f"  {NAMES[m]}：纯模型 {v['accuracy']:.4f}% (n={v['n']}; 严格集合{v['strict']:.4f}%)；长沙 {model['长沙']['accuracy']:.4f}%，杭州 {model['杭州']['accuracy']:.4f}%；规则+模型 {w['accuracy']:.4f}% (n={w['n']}; 严格集合{w['strict']:.4f}%)"]
        for key,v in res['comparisons'].items():
            a,b=key.split('-'); z=v['两城']
            lines += [f"  {NAMES[a]} − {NAMES[b]}：{z['delta']:+.4f}，95%CI {ci(z['ci'])}，校正CI {ci(v['adjusted_ci'])}；改对/改错 {z['fixed']}/{z['broken']}",
                f"    长沙 {v['长沙']['delta']:+.4f} {ci(v['长沙']['ci'])}；杭州 {v['杭州']['delta']:+.4f} {ci(v['杭州']['ci'])}"]
        lines.append('')
    lines += ['预设判定']+[f"  {key}: {v['confirmed_development_benefit']}" for key,v in verdict.items()]
    lines += ['', '审计：24份设置选择、96份三种子预测、288份权重重放通过；详见完整性审计.json。',
        '限制：全部为复用pre_v1开发数据；方案诊断用过K0_s1/K1_s1答案，LN基线此前看过开发结果才选定；杭州答案未冻结。三个K1参照批次不等于三个独立地区验证。',
        '规则+模型排除人为补充核验的参照样本，历史规则口径为规则输出任一学校命中；严格集合口径另报。部分政策训练/评价小区同楼盘，参照已按原协议遮蔽，但不属于完全楼盘隔离测试。',
        'FEAT和GMU只提供改法启发，本轮是自定义ModernNCA-LN适配扩展。未验证新算法贡献，未替换默认模型。']
    (run.DOC/'结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    passed=[m for m in run.METHODS if verdict[f'{m}-layernorm']['confirmed_development_benefit']
        and verdict[f'{m}-reference']['confirmed_development_benefit']]
    ln_passed=[m for m in run.METHODS if verdict[f'{m}-layernorm']['confirmed_development_benefit']]
    headline=('达到同时优于两个对照的开发判定：'+ '、'.join(NAMES[m] for m in passed)) if passed else (
        '达到优于LN基线的开发判定：'+'、'.join(NAMES[m] for m in ln_passed)+'；尚未同时确认优于RetrievalScorer。' if ln_passed else
        '本轮三个尝试均未达到稳定优于ModernNCA-LN的预设开发判定。')
    concise=[headline,'本轮固定三个尝试，全部结果已完成并通过完整性审计。','']
    for m in run.METHODS:
        key=f'{m}-layernorm'
        ds=[result[f'k1_s{s}']['comparisons'][key]['两城']['delta'] for s in [1,2,3]]
        concise += [f"{NAMES[m]}：K1相对LN基线 {' / '.join(f'{d:+.2f}' for d in ds)} 个百分点；预设稳定提升判定：{verdict[key]['confirmed_development_benefit']}。"]
    concise += ['', '输入相关融合优于固定系数对照的判定：'+str(verdict['adaptivegate-constantgate']['confirmed_development_benefit']),
        '', '完整数字、分城差异、区间和改对/改错见结果.txt；结构、公式与原文依据见文献与改法.txt。',
        '当前结果只决定哪些方向值得继续验证，不构成独立确认或自动替换研究基线的依据。']
    (run.DOC/'结论.txt').write_text('\n'.join(concise)+'\n',encoding='utf8')
    print('\n'.join(concise),flush=True)


if __name__=='__main__': main()
