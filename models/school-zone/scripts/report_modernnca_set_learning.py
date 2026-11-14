"""Human-readable post-score report; not imported by training/inference."""
import json
import experiment_modernnca_set_learning as E


def main():
    r=E.read(E.OUT/'results.json');audit=E.read(E.OUT/'audit.json');assert audit['passed']
    sel=r['source_selection'];source=r['source_ensemble'];doc=E.DOC;doc.mkdir(parents=True,exist_ok=True)
    names={'categorical_policy':'原Modern+现有政策','untrained_set_readout':'未训练集合评分头（同集成方式）','learned_set_readout':'训练后的集合表示与评分头'}
    lines=['ModernNCA完整学校集合学习实验（2026-11-13）','',
        '这轮实际训练了新集合表示与评分网络。ModernNCA-LN证据编码器、查询和距离匹配权重冻结；并非整个Modern端到端重训。',
        '训练输入为现有完整源标签。学校证据向量先经可学习变换，再按集合均值和方差聚合，并学习集合残差分数。训练目标是选中完整集合，而不是逐校独立BCE或手调深圳阈值。','',
        '一、目标区结果（相同1761个固定地图Gold样本）',
        '方案 | 全部 | 龙华627 | 龙岗1134 | 单校803 | 多校958']
    for arm,name in names.items():
        m=r['metrics'][arm]
        lines.append(name+' | '+' | '.join(f"{m[k]['correct']}/{m[k]['n']} ({m[k]['exact']:.2f}%)" for k in ['全部','龙华区','龙岗区','单校','多校']))
    lines += ['','多校数量分层（封存预测后的统计）：','方案 | 两校 | 三校及以上']
    cardinality={}
    for arm,name in names.items():
        rr=E.read(E.OUT/'scored_rows'/f'{arm}.json');cardinality[arm]={}
        values=[]
        for key,pred in [('两校',lambda r:r['size']==2),('三校及以上',lambda r:r['size']>=3)]:
            rows=[r for r in rr if pred(r)];n=len(rows);correct=sum(r['correct'] for r in rows)
            cardinality[arm][key]=dict(n=n,correct=correct)
            values.append(f'{correct}/{n} ({100*correct/n:.2f}%)')
        lines.append(name+' | '+' | '.join(values))
    E.write(doc/'按学校数量分层.json',cardinality)
    for baseline in ['untrained_set_readout','categorical_policy']:
        z=r['comparisons'][f'learned_set_readout minus {baseline}']['全部']
        lines += ['',f"相对{names[baseline]}：{z['delta_pp']:+.3f}个百分点；纠正{z['corrected']}、改错{z['harmed']}；95%空间块bootstrap区间{z['ci95']}。",
            '三种子正确数变化：'+str({s:r['seeds']['learned_set_readout'][s]['全部']['correct']-r['seeds'][baseline][s]['全部']['correct'] for s in map(str,E.M.SEEDS)})]
    lines += ['',
        '原Modern对照来自此前相同完整输入的categorical训练组（898/1761），不是历史RetrievalScorer的959/1761。未训练头采用与学习头相同的集合概率集成，因此不应拿上一轮“先平均学校概率再组集合”的875直接作严格学习消融。','',
        '二、训练与选择',
        '完整源训练样本4884个，其中完整多校94个；来自长沙和杭州。当前固定参照池能对应到完整多校源标签的仅19个住宅；深圳不存在已确认完整的目标训练集合。没有新增参照或把部分正关系伪造为完整标签。',
        '源训练4790单校、94两校，源验证1186单校、60两校：两者都没有三校及以上完整标签。深圳三校及以上结果单独报告；这是重要分布差异，但不能单凭这个观察断言失败原因。',
        '已有源验证1246条，其中多校60条；源训练/验证同楼盘分离。源校准和深圳均已用于开发，不算新独立验证。',
        '4组学习率/正则组合，每组最高640步；检查80/160/320/640步。按源完整集合正确数选择，并要求单校准确率相对同一冻结教师下降不超过0.5个百分点。',
        '选中配置：'+json.dumps(sel,ensure_ascii=False),
        '最终三种子源验证结果：'+json.dumps(source,ensure_ascii=False),
        '源验证保护条件通过不等于深圳单校一定不受损；目标区单校结果已单列。','',
        '三、信息边界与局限',
        '没有读取目标Gold构造输入、候选、特征或选择模型。所有目标输出封存后才评分；原Gold哈希未变。',
        '深圳候选、正关系下界、排除事实和政策解释均沿用上一轮冻结版本。大学区不扩成员；未确认的住宅适用条件不升级为事实。',
        '同一候选库仍有417条完整正确集合缺失（其中多校404条）；本次单独检验能否解决已有正确候选的评分问题，不声称训练可突破候选缺失。',
        '训练目标增加了完整集合学习，但没有新增独立住宅行政隶属事实，也没有更新原Modern编码器。源候选形成方式和深圳政策候选存在分布差异。',
        '所有1802条住宅输出保留；41条地图外不计分；242条政策/地图语义待核样本保留并单报。','',
        '四、验证与采用状态',
        f"5项单元测试通过；独立重放{audit['target_probability_arrays_bitwise_replayed']}组目标概率、核对{audit['output_rows_independently_recounted']}条输出，重放{audit['teacher_latent_rows_bitwise_replayed']}条教师证据缓存。",
        '源模型选择、最终源验证指标、文件哈希及原Gold均已审计。',
        '默认模型未修改；adopted=false；independent_validation=false。无论本次正负，均不能据此宣布在新的城市/年份已稳定有效。','',
        '完整产物目录：'+str(E.OUT)]
    (doc/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    before={(x['d'],x['site_id']):x for x in E.read(E.OUT/'scored_rows/categorical_policy.json')}
    changed=[]
    for row in E.read(E.OUT/'scored_rows/learned_set_readout.json'):
        old=before[row['d'],row['site_id']]
        if row['prediction']!=old['prediction']:
            changed.append(dict(row,baseline_prediction=old['prediction'],baseline_correct=old['correct'],diagnostic_only_not_training=True))
    E.write(doc/'逐条变化_禁止作为训练输入.json',changed)
    E.write(doc/'交付清单.json',dict(files={str(p):E.sha(p) for p in doc.iterdir() if p.is_file() and p.name!='交付清单.json'},adopted=False))
    print('REPORT WRITTEN',doc,flush=True)


if __name__=='__main__':main()
