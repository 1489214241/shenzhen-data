"""Post-freeze descriptive comparison. Never changes model inputs or predictions."""
from pathlib import Path
import pickle
import shenzhen_full_inputs as F
from zoning.evaluate import bootstrap

OUT=F.REPO/'outputs/shenzhen_aligned_model_20261111'
OLD=F.REPO/'outputs/shenzhen_full_model_20261111'


def metric(rows,field):
    return dict(n=len(rows),correct=sum(set(r[field])==set(r['gold']) for r in rows),
        accuracy=sum(set(r[field])==set(r['gold']) for r in rows)/len(rows) if rows else None)


def main():
    assert F.read(OUT/'verification.json')['passed']
    new=F.read(OUT/'results.json');old=F.read(OLD/'results.json')
    cs=F.read(F.REPO/'outputs/hz_comm_ER/eval_recheck.json')['changsha_sets']
    rows=[r for r in F.read(OUT/'scored_cases.json') if r['has_map_gold']]
    priors={(r['district'],r['site_id']):r for r in F.read(OLD/'scored_cases.json')}
    known=set();coords={}
    for d in new['districts']:
        m=F.read(OUT/'inputs'/f'{d}.json')
        known.update((d,m['sites'][int(q)]['coordinate_group_id']) for q in m['known'])
        with (F.REPO/'data/深圳市/规则对齐输入_20261111'/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        coords.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    grouped={};lines=['深圳对齐重跑：差距与错误分组（评分后分析，不回调参数）','',
        '比较采用固定地图命中集合及完整学校集合相等口径。统计关联不能单独证明某条规则的因果贡献。','',
        '分组\t样本数\t旧完整输入正确数\t本轮正确数\t本轮准确率']
    selectors={
        '全部':lambda r:True,
        '龙华区':lambda r:r['district']=='龙华区',
        '龙岗区':lambda r:r['district']=='龙岗区',
        '原图单校':lambda r:len(r['gold'])==1,
        '原图多校':lambda r:len(r['gold'])>1,
        '本轮规则参照覆盖位置':lambda r:(r['district'],r['site_id']) in known,
        '本轮其余位置':lambda r:(r['district'],r['site_id']) not in known,
        '原图语义待核':lambda r:r['semantic_pending'],
        '原图未标记语义待核':lambda r:not r['semantic_pending'],
    }
    for key,fn in selectors.items():
        rs=[r for r in rows if fn(r)];bs=[priors[r['district'],r['site_id']] for r in rs]
        a,b=metric(rs,'after'),metric(bs,'after');grouped[key]=dict(before=b,after=a)
        delta=[int(set(r['after'])==set(r['gold']))-int(set(p['after'])==set(p['gold'])) for r,p in zip(rs,bs)]
        grouped[key].update(corrected=delta.count(1),harmed=delta.count(-1))
        if key in ['全部','龙华区','龙岗区']:
            blocks=[dict(d=r['district'],block=str(tuple(int(v//1000) for v in coords[r['district'],r['site_id']]))) for r in rs]
            grouped[key]['delta_95_percentile_pp']=bootstrap(blocks,delta,seed=20261104,draws=3000)
        value='不适用' if a['accuracy'] is None else f"{a['accuracy']:.2%}"
        lines.append(f"{key}\t{len(rs)}\t{b['correct']}\t{a['correct']}\t{value}")
    seed_deltas={k:new['per_seed'][k]['after']['exact_accuracy']-old['per_seed'][k]['after']['exact_accuracy']
        for k in new['per_seed'] if k!='ensemble'}
    lines+=['','相较旧完整输入：']
    for d in ['龙华区','龙岗区','全部']:
        g=grouped[d];ci=g['delta_95_percentile_pp']
        lines.append(f"{d}：改正 {g['corrected']}，损害 {g['harmed']}；区分层 1km 配对重抽样 95%变化区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点。")
    lines+=['','逐种子准确率变化（百分点；全部种子保留，不选择最优种子）：',
        ', '.join(f'{s}: {v*100:+.2f}' for s,v in seed_deltas.items()),'',
        '长沙现有严格口径参考：outputs/hz_comm_ER/eval_recheck.json 中 3362/3858 = 87.14%。',
        '这不是两城同难度的因果对照：长沙训练有同城其他区，深圳来源只有长沙/杭州八区；政策类型和多校比例也不同。',
        '长沙该最终输出还包含逐小区政策核验覆盖；不能把它与深圳外测的差距全部归咎于模型。',
        '学校参照覆盖：长沙 198/251；本轮龙华 17/71、龙岗 63/131。深圳参照较少仍限制近邻参照特征。',
        '龙岗仍有 29 所学校包含未完全解析的子句；独立行政边界仅解决龙华 5/53、龙岗 5/83 个查询，闭合道路范围解决 2/15 个查询。',
        '上述是输入覆盖及模型任务的差别，不等于官方没有发布文字，不据此宣称已经解释全部性能差距。',
        '规则能解析不代表空间条件已确定；社区坐标是位置线索，不能代替行政边界。',
        '本轮是开发修复后的复测，不能据此宣称独立验证通过。',
        '结论：整体点估计小幅提高，但空间区间跨 0，且误报增加、龙华下降；未确认稳定提升，不据此自动替换旧默认模型。']
    F.write(OUT/'错误分组与种子对照.json',dict(groups=grouped,seed_accuracy_deltas=seed_deltas))
    (OUT/'差距与错误分组.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'评分后分析来源.json',dict(script_sha256=F.sha(Path(__file__)),
        new_result_sha256=F.sha(OUT/'results.json'),old_result_sha256=F.sha(OLD/'results.json'),
        changsha_result_sha256=F.sha(F.REPO/'outputs/hz_comm_ER/eval_recheck.json'),changsha_summary=cs['overall'],
        target_inputs_modified=False,predictions_modified=False))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
