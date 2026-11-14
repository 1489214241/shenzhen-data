"""Independently audit frozen predictions and generate the experiment report."""
import sys
import json
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiment_modernnca_shenzhen_multilabel import (OUT,DOC,ROOT,OLD,CURRENT,TARGETS,METHODS,ARMS,
    read,write,verify,check_manifest,inputs,metric,MultiLabelModernNCA,task,predict,DEVICE)
from shenzhen_partial_model import predict_sets
from shenzhen_reviewed_supplements import ReviewedSupplements
import shenzhen_full_inputs as F


def main():
    c=verify()
    assert F.sha(OUT/'protocol.json')==read(OUT/'protocol_receipt.json')['protocol_sha256']
    for name in ['selection_freeze.json','weights_freeze.json','prediction_freeze.json']:check_manifest(name)
    xs,ys=inputs(c);result=read(OUT/'results.json')
    threshold=read(OUT/'source_calibration.json')['selections']['balanced_exact']['threshold']
    base=REPO/'outputs/shenzhen_aligned_model_20261111'
    gold={(r['district'],r['site_id']):r for r in read(base/'scored_cases.json')}
    total=0;independent={};replayed=0
    for method in METHODS:
        payload=[dict(np.load(OUT/'probabilities'/method/f'{seed}.npz')) for seed in c['seeds']]
        mean={d:np.stack([p[d] for p in payload]).mean(0) for d in xs}
        for label,pp in [('ensemble',mean)]+list(zip(map(str,c['seeds']),payload)):
            for d in TARGETS:
                meta=read(CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
                policy=read(ROOT/'round08_reference_retraining/policies'/f'{d}.json');graph=read(OLD/'source_graphs'/f'{d}.json')
                actual={suffix:read(OUT/'predictions'/f'{method}_{suffix}'/label/f'{d}.json') for suffix in ['raw','policy']}
                expected=[]
                for q,site in enumerate(meta['sites']):
                    p=np.asarray(pp[d][q],np.float64)
                    selected=[int(p.argmax())] if method=='categorical' else np.flatnonzero(p>=threshold).tolist()
                    if not selected:selected=[int(p.argmax())]
                    assert actual['raw'][q]['after']==sorted(ids[j] for j in selected)
                    pos=[ids[j] for j in meta['known'].get(str(q),[])]
                    dec=predict_sets(p,2. if method=='categorical' else threshold,ids,policy,graph,site,pos)
                    expected.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],after=dec['predicted_school_ids'],decision=dec))
                assert ReviewedSupplements(d).apply(expected)==actual['policy']
                for suffix,rows in actual.items():
                    arm=f'{method}_{suffix}';all_ids={r['site_id'] for r in rows}
                    assert len(all_ids)==len(rows)==xs[d].n
                    correct=0;fp=fn=0
                    for row in rows:
                        assert len(set(row['after']))==len(row['after']);assert set(row['after'])<=set(ids)
                        g=gold[d,row['site_id']]
                        if g['has_map_gold']:
                            truth=set(g['gold']);pred=set(row['after']);correct+=truth==pred;fp+=len(pred-truth);fn+=len(truth-pred)
                    expected_metric=result['seeds'][arm][label][d]
                    assert correct==expected_metric['correct'] and fp==expected_metric['fp'] and fn==expected_metric['fn']
                    independent.setdefault(arm,{}).setdefault(label,0);independent[arm][label]+=correct;total+=len(rows)
        # Independently reload every checkpoint and replay both target arrays.
        for i,seed in enumerate(c['seeds']):
            m=MultiLabelModernNCA(41,27,seed,c['dim']).to(DEVICE)
            m.load_state_dict(torch.load(OUT/'weights'/method/f'{seed}.pt',map_location=DEVICE,weights_only=True))
            for d in TARGETS:
                actual=predict(m,task(xs[d],np.arange(xs[d].n)),method)
                assert np.array_equal(actual,payload[i][d]);replayed+=1
    for arm,byseed in independent.items():
        for label,n in byseed.items():assert n==result['seeds'][arm][label]['全部']['correct']
    source=read(OUT/'source_calibration.json');valpred=[];vallabel=[]
    for d in c['sources']:
        assert not(set(xs[d].groups[ys[d]['train_ids']])&set(xs[d].groups[ys[d]['valid_ids']]))
        p=np.stack([np.load(OUT/'probabilities/multilabel'/f'{s}.npz')[d] for s in c['seeds']]).mean(0).astype(np.float64)
        pred=p>=threshold;empty=~pred.any(1);pred[empty,p[empty].argmax(1)]=True
        valpred.extend(pred.tolist());vallabel.extend(ys[d]['valid_labels'].tolist())
    nc=sum(a==b for a,b in zip(valpred,vallabel));assert nc==source['selections']['balanced_exact']['correct']
    for d in TARGETS:assert (ys[d]['labels']!=0).all() and not len(ys[d]['valid_ids'])
    cardinality={}
    for arm in ARMS:
        rr=read(OUT/'scored_rows'/f'{arm}.json')
        cardinality[arm]={str(k):metric([r for r in rr if (r['size']==k if k<3 else r['size']>=3)]) for k in [1,2,3]}
    audit=dict(passed=True,output_rows_checked=total,reloaded_target_probability_arrays_bitwise_equal=replayed,
        independent_correct_counts=independent,source_threshold_recount=nc,source_train_validation_groups_disjoint=True,
        target_unknown_not_negative=True,target_no_validation_labels=True,original_gold_unchanged=True,adopted=False,
        cardinality_diagnostics=cardinality,
        torch=torch.__version__,device=DEVICE,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    write(OUT/'audit.json',audit)
    names={'categorical_raw':'Modern单校输出（纯模型）','categorical_policy':'Modern单校输出＋现有政策',
           'multilabel_raw':'Modern多标签输出（纯模型）','multilabel_policy':'Modern多标签输出＋现有政策'}
    main=result['comparisons']['multilabel_policy minus categorical_policy']['全部']
    current=result['historical_incumbent']['multilabel_policy']
    headline=f"多标签＋政策相对单校＋政策：{main['delta_pp']:+.2f}个百分点；95%区间{main['ci95']}。相对历史现用模型：{current['delta_pp']:+.2f}个百分点（训练条件不同）。"
    lines=['ModernNCA-LN 深圳多校适配验证，2026-11-13','',headline,
        '已完成实际训练、来源区选参、冻结预测和独立重放；未自动替换现用模型。',
        '主模型是ModernNCA-LN全证据关系适配版，保留编码器、查询网络、欧氏距离及学校聚合；没有使用RetrievalScorer的权重或热启动。',
        '单校对照使用原Modern类别概率及argmax；为同信息比较，多个已知正例按均匀权重重复单标签NLL。它不是旧的仅单校训练历史成绩复现。',
        '多标签版输出归一化前的逐校log-mass，加一个共享偏置，使用masked BCE；未知标签不参与二元损失。',
        '两组均使用同一来源区完整标签、深圳部分正例、参照、学校锚点、特征、空间划分及50/50完整来源单/多校加权。未把后续政策补充冒充模型已用训练输入。',
        '训练来源为长沙杭州八区；94条完整多校训练记录。1246条来源空间留出记录中60条多校，用于选参和阈值，不是独立确认集。',
        '每种方法4组lr/reg、40/80/160/320步检查点，来源验证选定后训练三个相同种子。320步是本轮预设上限；若最优在边界，不能声称优化充分。',
        '政策版复用当前同一套大学区、共享范围、部分正例及独立事实补充。没有新造共享关系，也没有用目标polygon构造候选或参照。',
        '评分为固定1761个原图命中坐标组的完整学校集合；另41个图外组仍输出。原图多命中不自动解释为共享资格；语义待核组单列，不删除或改写答案。','',
        '方式 | 全部 | 龙华 | 龙岗 | 单校 | 多校 | 误报/漏报']
    for arm in ARMS:
        row=[names[arm]]
        for g in ['全部']+TARGETS+['单校','多校']:
            m=result['metrics'][arm][g];row.append(f"{m['correct']}/{m['n']}（{m['exact']:.2f}%）")
        m=result['metrics'][arm]['全部'];row.append(f"{m['fp']}/{m['fn']}");lines.append(' | '.join(row))
    lines+=['','主要配对比较：']
    for key,dd in result['comparisons'].items():
        a,b=key.split(' minus ');v=dd['全部'];ci=v['ci95']
        ds=[result['seeds'][a][str(s)]['全部']['correct']-result['seeds'][b][str(s)]['全部']['correct'] for s in c['seeds']]
        lines.append(f"{names[a]} 对 {names[b]}：{v['delta_pp']:+.2f}个百分点，95%空间区间[{ci[0]:+.2f},{ci[1]:+.2f}]，纠正{v['corrected']}、损害{v['harmed']}；三个种子净正确数变化{ds}。")
    lines+=['','历史现用深圳模型为959/1761=54.46%，不是Modern；训练协议与本轮不同，只作为当前实际性能参照。']
    for arm,v in result['historical_incumbent'].items():
        lines.append(f"{names[arm]}相对现用：{v['delta_pp']:+.2f}个百分点；纠正{v['corrected']}、损害{v['harmed']}；95%区间{v['ci95']}。")
    lines+=['','来源验证选定参数：']
    for method in METHODS:
        lines.append(method+'：'+json.dumps(read(OUT/'selection'/f'{method}.json'),ensure_ascii=False))
    lines+=['','按完整答案学校数拆分（3表示3所及以上）：']
    for arm,dd in cardinality.items():
        lines.append(names[arm]+'：'+json.dumps(dd,ensure_ascii=False))
    lines+=['多标签最终来源校准：'+json.dumps(source['selections']['balanced_exact'],ensure_ascii=False),
        '',f'核验：{total}条输出独立核对；{replayed}个目标概率数组从权重重新推断逐位相同；阈值只用来源答案；原Gold哈希未变。',
        '局限：这是一轮开发适配实验，不是原版ModernNCA论文复现、不是独立泛化证明，也不是新共享规则发现。多校训练记录稀少且来自旧城市；不能把一次低分说成Modern无法解决多校。',
        '运行顺序：scripts/experiment_modernnca_shenzhen_multilabel.py prepare/select/fit/infer/score；然后scripts/audit_modernnca_shenzhen_multilabel.py。',
        '详细结果：outputs/modernnca_shenzhen_multilabel_20261113/results.json、scored_rows、selection、weights、predictions、audit.json。']
    text='\n'.join(lines)+'\n'
    (OUT/'结果与限制.txt').write_text(text,encoding='utf8');DOC.mkdir(parents=True,exist_ok=True);(DOC/'结果与限制.txt').write_text(text,encoding='utf8')
    print(audit)


if __name__=='__main__':main()
