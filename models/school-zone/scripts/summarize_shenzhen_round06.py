"""Source ranking diagnostics and terminal integrity for the frozen round06."""
import numpy as np
import torch
import shenzhen_full_inputs as F
from experiment_shenzhen_balanced_ranking import OUT, OLD, ARMS, OBJECTIVES, verify
from shenzhen_exact_calibration import calibration_intervals, threshold_metrics


def main():
    if (OUT/'completion_integrity.json').exists():
        raise FileExistsError('Keep completed audit')
    c = verify()
    checks = {}
    for name in ['input_freeze.json','probability_freeze.json','prediction_freeze.json']:
        files = F.read(OUT/name)['files']
        for rel,h in files.items():
            assert F.sha(OUT/rel)==h,rel
        checks[name]=len(files)
    history = F.read(OUT/'training_history.json')
    assert len(history)==30
    for seed in c['seeds']:
        for arm in ARMS:
            p=OUT/'checkpoints'/f'{arm}_{seed}.pt'
            ck=torch.load(p,map_location='cpu',weights_only=False)
            assert ck['stage']=='full' and ck['step']==40 and len(ck['history'])==40
            assert all(np.isfinite(list(v.values())).all() for v in ck['history'])
            assert history[f'{arm}/{seed}']['checkpoint_sha256']==F.sha(p)
    ys={d:dict(np.load(OLD/'inputs'/f'{d}_labels.npz')) for d in c['sources']}
    diagnostics={}
    lines=['第六轮：训练权重与多校排序诊断','',
        '30个模型，同父模型、同40步，另有继续原BCE的对照。未改原图Gold、分母或当前默认模型。',
        '所有校准仅使用来源区。以下结果仍是开发证据，不是独立验证。','']
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{seed}.npz')) for seed in c['seeds']]
        mean={d:np.stack([x[d] for x in loaded]).mean(0) for d in loaded[0]}
        pp=[mean[d+'_train_multi'] for d in c['sources']]
        yy=[z['labels'][np.flatnonzero((z['labels']>=0).all(1)&((z['labels']==1).sum(1)>1))]==1 for z in ys.values()]
        intervals,sizes=calibration_intervals(pp,yy)
        assert len(sizes)==94
        vp=[mean[d] for d in c['sources']];vy=[ys[d]['valid_labels'] for d in c['sources']]
        vi,vs=calibration_intervals(vp,vy)
        cal=F.read(OUT/'source_calibration'/f'{arm}.json')['selections']
        report=dict(train_multi_count=94,train_multi_attainable=len(intervals),
            train_at_original_threshold=threshold_metrics(pp,yy,.9),
            train_at_source_calibrated_threshold={o:threshold_metrics(pp,yy,cal[o]['threshold']) for o in OBJECTIVES},
            source_validation=cal,validation_multi_count=int((vs>1).sum()),
            validation_multi_attainable=sum(k>1 for lo,hi,k in vi))
        diagnostics[arm]=report
        lines.append(f"{arm}: 94条训练多校在原0.90阈值下完整正确{report['train_at_original_threshold']['correct']}条；任意逐条阈值的原始排序容量{len(intervals)}/94。")
        lines.append(f"来源区60条多校留出样本，原始排序容量{report['validation_multi_attainable']}/60；这不是部署准确率。")
        for o in OBJECTIVES:
            s=cal[o]
            lines.append(f"{o}: t={s['threshold']:.8f}; 来源区全部{s['correct']}/{s['n']}，其中多校{s['multi_correct']}/60。")
    F.write(OUT/'source_ranking_diagnostics.json',dict(reports=diagnostics,target_gold_read=False))
    results=F.read(OUT/'results.json')['reports']
    lines += ['', '深圳严格完整集合评测（1761条，41条图外不计分）：']
    for arm in ARMS:
        for o in OBJECTIVES:
            r=results[arm][o]['ensemble']['全部'];a=r['after'];b=r['before'];p=r['parent']
            seeds=[results[arm][o][str(s)]['全部']['before']['delta_pp'] for s in c['seeds']]
            lines.append(f"{arm}/{o}: {a['correct']}/1761={a['exact_accuracy']:.2%}; 误报{a['fp']}，漏报{a['fn']}; 对同校准继续BCE改正{b['corrected']}/损害{b['harmed']}，95%差值区间{b['delta_95_pp']}；对冻结父模型改正{p['corrected']}/损害{p['harmed']}，95%差值区间{p['delta_95_pp']}。")
            lines.append(f"十种子相对继续BCE的变化范围[{min(seeds):.3f},{max(seeds):.3f}]个百分点。")
    lines += ['', '没有独立验证，不自动采用。后续结论需同时检查两区、多校/单校、误报漏报与损害，不能只挑最高分。',
              '范围解析的6校修复单独保存在scoped_ring_source_audit_v2，未并入本实验。独立几何审计新增可判定样本为0，尚不能把语法修复称作输入已补齐。']
    (OUT/'第六轮诊断总结.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'completion_integrity.json',dict(passed=True,verified_files=checks,
        checkpoint_count=30,steps_each=40,loss_history_finite=True,original_gold_check=F.read(OUT/'verification.json'),
        adopted=False,independent_validation=False))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
