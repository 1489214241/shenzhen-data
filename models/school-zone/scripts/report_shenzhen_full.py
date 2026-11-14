"""Describe and verify the frozen full-input run; never fit or alter predictions."""
from pathlib import Path
import collections, json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import shenzhen_full_inputs as F
import experiment_shenzhen_full as R

OUT=R.OUT


def main():
    result=F.read(OUT/'results.json');rows=F.read(OUT/'scored_cases.json')
    cfg=F.read(OUT/'protocol.json');audit=F.read(OUT/'input_audit.json')
    coverage=F.read(OUT/'reference_coverage.json')
    freeze=F.read(OUT/'prediction_freeze.json')
    for p,h in freeze['prediction_sha256'].items():assert F.sha(OUT/p)==h
    for d,h in freeze['probability_sha256'].items():assert F.sha(OUT/'probabilities'/f'{d}.npz')==h
    R.B.check_frozen()
    history=F.read(OUT/'training_history.json')
    assert len(history)==len(cfg['targets'])*len(cfg['seeds'])==20
    for key,v in history.items():
        d,seed=key.split('/');assert len(v['loss'])==40 and np.isfinite(v['loss']).all()
        assert F.sha(OUT/'checkpoints'/f'{d}_{seed}.pt')==v['checkpoint_sha256']
    assert len(rows)==1802 and sum(r['has_map_gold'] for r in rows)==1761
    assert len({(r['district'],r['site_id']) for r in rows})==len(rows)
    old=F.read(F.REPO/'outputs/shenzhen_current_model_20261111/results.json')
    assert old['gold_sha256']==result['gold_sha256']
    names={};policies={};known={};coords={}
    for d in cfg['targets']:
        meta=F.read(OUT/'inputs'/f'{d}.json')
        names.update({s['school_id']:s['school_name'] for s in meta['schools']})
        known[d]={meta['sites'][int(q)]['coordinate_group_id']:v for q,v in meta['known_kind'].items()}
        coords.update({(d,s['coordinate_group_id']):xy for s,xy in zip(meta['sites'],F.project([s['representative_point_wgs84'] for s in meta['sites']]))})
        policies[d]={r['rule_id']:r for r in F.SZ._load(d,'政策规则/规则包.json')['relations']}
        pp=np.load(OUT/'probabilities'/f'{d}.npz')
        assert pp['probs'].shape==(10,len(meta['sites']),len(meta['schools']))
        assert pp['seeds'].tolist()==cfg['seeds'] and np.isfinite(pp['probs']).all()
        assert np.allclose(pp['probs'].sum(-1),1,atol=1e-5)
    # B.score retains a legacy 2km display key; the full scorer already bootstraps
    # 1km from coordinates. Align only this derived display field, not predictions.
    for r in rows:
        xy=coords[r['district'],r['site_id']]
        r['block']=f"{r['district']}/{int(xy[0]//1000)}/{int(xy[1]//1000)}"
        r['block_metres']=1000
    F.write(OUT/'scored_cases.json',rows)
    wrong=[]
    for r in rows:
        if not r['has_map_gold'] or not ((set(r['after'])-set(r['before']))-set(r['gold'])):continue
        item=dict(r)
        for k in ['before','after','gold']:item[k+'_school_names']=[names[s] for s in r[k]]
        item['official_rules_and_quotes']=[policies[r['district']][v['rule_id']] for v in r['applied_rules']]
        item['before_had_any_map_school']=bool(set(r['before'])&set(r['gold']))
        wrong.append(item)
    F.write(OUT/'错加诊断与政策原文.json',dict(cases=wrong,count=len(wrong),
        before_disjoint=sum(not r['before_had_any_map_school'] for r in wrong),
        singleton_map_wrong_add=sum(len(r['gold'])==1 for r in wrong)))
    diagnostics={}
    for d in cfg['targets']:
        rr=[r for r in rows if r['district']==d and r['has_map_gold']]
        counter=collections.Counter()
        for r in rr:
            p,g=set(r['after']),set(r['gold'])
            label='correct' if p==g else 'disjoint' if not p&g else 'omission_only' if p<g else 'overreport_only' if g<p else 'both_omission_and_overreport'
            counter[label]+=1
        by={}
        for kind in ['named','boundary','community','model']:
            subset=[r for r in rr if known[d].get(r['site_id'],'model')==kind]
            by[kind]=R.B.compare(subset) if subset else dict(n=0)
        # Native overall has a permissive rule-known branch; report separately
        # from our predeclared strict-set primary metric, never swap the primary.
        native_ok=sum(bool(set(r['before'])&set(r['gold'])) if r['site_id'] in known[d]
                      else set(r['raw_argmax'])==set(r['gold']) for r in rr)
        diagnostics[d]=dict(final_errors=dict(counter),by_input_rule_kind=by,
            native_overall_before_output_expansion=dict(correct=native_ok,n=len(rr),accuracy=native_ok/len(rr),
                semantics='Native overall: any correct school for rule-known sites; exact top1 set for model sites. Supplementary metric only.'))
    F.write(OUT/'diagnostics.json',diagnostics)
    lines=['深圳完整输入模型重跑结果（2026-11-11）','',
        '本次为当前 RetrievalScorer 的完整输入外测。41列参照特征、27列学校特征、10个固定种子、40步，来源为长沙和杭州八区。',
        '深圳Gold和学区polygon不参与输入构建、训练、参照选择或预测；全部预测固定后才评分。',
        '主指标：输出完整学校集合必须与原图全部命中学校相等。长沙当前changsha_sets主指标也是完整集合；旧overall中的宽松规则命中分数应另列。','',
        '地区\t旧缺输入结果（3种子）\t本次完整输入（10种子）\t规则扩充前\t原图有命中组数']
    for d in cfg['targets']+['合计']:
        v=result['overall'] if d=='合计' else result['districts'][d]['all_map']
        o=old['overall'] if d=='合计' else old['districts'][d]['all_map']
        lines.append(f"{d}\t{o['after']['exact_accuracy']:.2%}\t{v['after']['exact_accuracy']:.2%}\t{v['before']['exact_accuracy']:.2%}\t{v['after']['n']}")
    lines+=['','旧、新实验输入及种子数不同；上表是历史运行对照，不是单项输入的严格消融。','',
        '本次输入覆盖（输入类别与默认配置相同，实际覆盖率并不相同）：']
    for d in cfg['targets']:
        a=audit[d]
        lines.append(f"{d}：{a['sites']}组小区，{a['school_coordinates']}/{a['schools']}所学校有坐标；社区候选点{a['community_points']}个，严格社区关联{a['strict_community_sites']}组；政策社区特征覆盖{a['community_feature_sites']}组；路名方向条件找到{a['road_sides_resolved']}/{a['road_sides']}项；规则参照{a['known_sites']}组，覆盖{coverage[d]['schools_with_reference']}/{a['schools']}所学校，单校训练参照{a['training_singleton_sites']}组。")
    lines+=['坐标链：BD09→WGS84→EPSG32650；独立OSM道路以WGS84投影到同一米制坐标系。',
        '百度社区点是办公地或社区位置线索，不是行政边界；不确定别名、部分范围、未解析条件仍保留未知。','',
        '单校、多校与输出规则风险：']
    for d,v in result['districts'].items():
        for k,label in [('single','原图单校'),('multi','原图多校'),('semantic_pending','语义待核')]:
            a=v[k]['after'];lines.append(f"{d} {label}：{a['correct']}/{a['n']}，{a['exact_accuracy']:.2%}。")
    v=result['overall'];b,a=v['before'],v['after'];ci=result['paired_interval']['delta_95_percentile']
    lines += [f"同一套完整模型的输出规则扩充：修正{v['corrected']}组，原本完整正确而被改错{v['harmed']}组，新增原图外学校{v['wrongly_added_sites']}组。",
        f"学校级错报{b['false_positive_school_assignments']}→{a['false_positive_school_assignments']}；漏报{b['false_negative_school_assignments']}→{a['false_negative_school_assignments']}。",
        f"最终学校级precision={a['precision']:.2%}，recall={a['recall']:.2%}。",
        f"输出规则增量的1公里分区分层空间bootstrap 95%区间：{100*ci[0]:.2f}至{100*ci[1]:.2f}个百分点（3000次）。",
        '修正/损害衡量完整集合对错，新增错误学校单独计算；损害为0也不代表没有新增错误。','',
        '10个种子重复（完整输入，规则扩充后）：']
    for seed,v in result['per_seed'].items():
        if seed!='ensemble':lines.append(f"{seed}：{v['after']['exact_accuracy']:.2%}；错加{v['wrongly_added_sites']}组。")
    lines+=['','评价边界：原图标签仍是点落面结果，242组存在基础学区/大学区/派位范围等语义待核项。',
        '全量预测1802组，计分1761组；图外41组不当作可靠负例。没有改Gold或按本轮得分调参。',
        '结果不构成独立验证后的正式采用。错加案例中的官方原文、URL、表行/页码见“错加诊断与政策原文.json”。','',
        '入口：实验结果.txt；详细统计：results.json；逐例：scored_cases.json；输入覆盖：input_audit.json；核验：verification.json。']
    (OUT/'重跑结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    F.write(OUT/'verification.json',dict(passed=True,trained_models=20,steps_per_model=40,
        predictions_unchanged_after_scoring=True,original_gold_unchanged_vs_previous_run=True,
        candidate_sites=1802,scored_sites=1761,scored_case_block_metres=1000,
        integrity=F.read(OUT/'integrity_audit.json'),chunk_gradient=F.read(OUT/'chunk_verification.json'),
        report_code_sha256=F.sha(Path(__file__))))
    print(json.dumps(dict(overall=result['overall'],diagnostics=diagnostics),ensure_ascii=False))


if __name__=='__main__':main()
