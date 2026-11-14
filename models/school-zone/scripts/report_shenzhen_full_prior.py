"""Report all arms and both stages, with source quotes and independent replay."""
import collections
import numpy as np
import experiment_shenzhen_full_prior_repair as R
from zoning.full_policy_prior import choose

F=R.F;read,write,sha=F.read,F.write,F.sha


def main():
    R.install();c=F.verify();F.check()
    result=read(R.OUT/'results.json');audit=read(R.OUT/'output_audit.json');assert audit['passed']
    before=read(R.BASE/'results.json');first_audit=read(R.BASE/'output_audit.json');assert first_audit['passed']
    pp=[dict(np.load(F.H.OUT/'probabilities/routed_bce'/f'{s}.npz')) for s in c['seeds']]
    threshold=read(F.H.OUT/'source_calibration.json')['routed_bce']['threshold']
    replay=0;names={};observations={};university_additions=0
    for d in c['targets']:
        meta=read(F.INPUT/d/'metadata.json');ids=[s['school_id'] for s in meta['schools']]
        names.update({s['school_id']:s['school_name'] for s in meta['schools']})
        observations.update({x['school_id']:x for x in read(F.INPUT/d/'full_scope_inventory.json')})
        policy=read(F.INPUT/d/'reviewed_policy.json');cons=read(F.INPUT/d/'constraints.json')
        inv=read(R.OUT/'inventory'/f'{d}.json');priors=read(R.OUT/'priors'/f'{d}.json')
        universities=[r['school_ids'] for r in policy['relations'] if r['kind']=='university']
        shared=[r['school_ids'] for r in inv['records'] if len(r['school_ids'])>1]
        mean=np.stack([p[d] for p in pp]).mean(0)
        baseline=read(R.OUT/'predictions/baseline_current/ensemble'/f'{d}.json')
        raw=read(F.H.OUT/'predictions/routed_bce_raw/ensemble'/f'{d}.json')
        supplemented=read(R.OUT/'predictions/full_scope_minimum/ensemble'/f'{d}.json')
        university_ids=set().union(*map(set,universities)) if universities else set()
        university_additions+=sum(bool((set(a['after'])-set(b['after']))&university_ids) for a,b in zip(supplemented,baseline))
        for arm,compose in [('full_prior_single_condition',False),('full_prior_composed',True)]:
            predicted=read(R.OUT/'predictions'/arm/'ensemble'/f'{d}.json')
            for q in [0,len(meta['sites'])//2,len(meta['sites'])-1]:
                con=cons[baseline[q]['site_id']];prior=priors[q]
                pool=R.original_pool(inv['records'],prior,[],[],[],con['negative'],compose)
                extra=R.corrected_pool([],prior,baseline[q]['after'],raw[q]['after'],prior['estimated_members'],con['negative'],False)
                merged=dict(pool);merged.update(dict(extra));allpool=sorted(merged.items(),key=lambda x:(len(x[0]),sorted(x[0])))
                after,decision=choose(allpool,mean[q],ids,threshold,universities,shared)
                assert after==predicted[q]['after'],(d,q,arm)
                assert abs(decision['score']-predicted[q]['decision']['score'])<1e-9
                replay+=1
    baseline={(x['d'],x['site_id']):x for x in read(R.OUT/'scored_rows/baseline_current.json')}
    oldpriors={(d,x['site_id']):x for d in c['targets'] for x in read(F.BASE/'priors'/f'{d}.json')}
    strata={};changes=[];rows_by={}
    for arm in F.ARMS:
        rows=read(R.OUT/'scored_rows'/f'{arm}.json');rows_by[arm]=rows;strata[arm]={}
        for label,test in [('原104条文字多校',lambda p:len(p['verified_members'])>=2),
                           ('原345条模型推测多校',lambda p:len(p['verified_members'])<2 and len(p['estimated_members'])>=2),
                           ('原509条未接提前条件多校',lambda p:len(p['estimated_members'])<2)]:
            chosen=[x for x in rows if x['size']>1 and test(oldpriors[x['d'],x['site_id']])]
            strata[arm][label]=dict(n=len(chosen),before=sum(baseline[x['d'],x['site_id']]['correct'] for x in chosen),
                after=sum(x['correct'] for x in chosen))
    for x in rows_by['full_scope_minimum']:
        old=baseline[x['d'],x['site_id']]
        if x['prediction']==old['prediction']:continue
        added=set(x['prediction'])-set(old['prediction'])
        changes.append(dict(district=x['d'],site_id=x['site_id'],name=x['name'],
            before=[names[s] for s in old['prediction']],after=[names[s] for s in x['prediction']],
            map_scoring_set=[names[s] for s in x['truth']],added=[names[s] for s in sorted(added)],
            before_correct=old['correct'],after_correct=x['correct'],semantic_pending=x['semantic_pending'],
            original_scope_quotes=[dict(school=names[s],text=observations[s]['source_text'],
                source=observations[s]['evidence']) for s in sorted(added)]))
    audit_extra=dict(passed=True,scalar_candidate_replays=replay,university_member_addition_sites=university_additions,
        note='Full population recomputed; scalar exhaustive ranking on12stratified sampled query/arm combinations. Does not constitute independent validation.')
    write(R.OUT/'independent_replay.json',audit_extra)
    labels={'baseline_current':'当前模型与规则','baseline_native':'原始Modern与规则','full_text_minimum':'全部文字确定关系补充',
        'full_scope_minimum':'保留原预测＋全部范围匹配线索补充','full_prior_single_condition':'单个范围候选重新选集合（已修复）',
        'full_prior_composed':'范围组合候选重新选集合（已修复，预定主比较）'}
    lines=['深圳全量多校提前输入实验（2026-11-14）','',
        '范围：现用龙华、龙岗全部202所学校、1802个小区坐标组。福田、南山继续保留为历史资料。',
        '同一1761条地图命中样本评分；41条仅输出；242条语义待核样本仍计分。学校集合须完全一致。',
        '所有样本均处理，不按Gold挑选双校或多校。未把地图校数或学校集合送入模型。',
        '固定此前Modern及补校头分数、三种子、source阈值；本次改变提前条件与集合选择，不是重训网络。','',
        '结果：范围匹配线索补充912→919，净增7条、+0.3975个百分点；修正8、误伤1。空间95%区间[0.0550,0.8359]个百分点。',
        '全部8条修正均处于已有语义待核层；其余1519条从873→872。故不能宣称政策语义已解决或可正式采用。',
        '预定主比较（完整范围组合重新选集合）修复后仅876，比912少36；修正2、误伤38。主比较失败。',
        '较好的是保留原模型后补充线索这一对照臂；不把它冒充预定主实验成功。没有修改正式模型默认值。','',
        '方法 | 全对/1761 | 准确率 | 单校/803 | 多校/958 | 三校及以上/364 | 龙华/627 | 龙岗/1134 | 修正/误伤']
    for arm,m in result['metrics'].items():
        diff=result['comparisons'].get(arm,{}).get('all',{})
        lines.append(f"{labels[arm]} | {m['all']['correct']} | {m['all']['exact']:.2f}% | {m['single']['correct']} | {m['multi']['correct']} | {m['three_plus']['correct']} | {m['龙华区']['correct']} | {m['龙岗区']['correct']} | {diff.get('corrected','—')}/{diff.get('harmed','—')}")
    lines.extend(['','前一轮未覆盖多校的结果（用冻结的前轮分层，事后诊断，不用于输入）：'])
    for arm,groups in strata.items():
        lines.append(labels[arm])
        for label,s in groups.items():lines.append(f"  {label}：{s['before']}→{s['after']}，分母{s['n']}")
    lines.extend(['','三次补校头初始化结果（不是三次新训练的主模型）：'])
    for arm in ['baseline_current','full_scope_minimum','full_prior_single_condition','full_prior_composed']:
        lines.append(labels[arm]+'：'+str({s:m['all']['correct'] for s,m in result['seeds'][arm].items()}))
    lines.extend(['','输入及方法限制：',
        '全量扫描不等于所有小区都已知校数：文字与独立归属证据仍覆盖105条多校关系；结合范围估计共205条。其余依赖模型候选判断。',
        '两区共162个多校范围条件；查询候选约龙华1891组、龙岗9460组，包含两条件组合。未知条件保留为未知。',
        '范围估计含社区点/道路近似，不能视作官方确认的行政归属；两条件组合只是假设，不能通过A—B、B—C关系证明A—B—C同时适用。',
        f'大学区未按成员名单扩展；实查范围补充臂新增大学区成员的样本数={university_additions}。',
        '修复前原始模型集合可绕过范围展开，在可加评分下占优，两种重新选集合结果均706。已将原始模型根学校按同一规则直接展开一次，再生成该候选；修复后879/876。',
        '修复前输出、协议、审计完整保留；修复是看过目标结果后的自适应工程修复，不是新的独立确认。',
        '扩大候选后仍下降，说明当前集合评分不能可靠地在候选间选择；此结论限于本轮固定评分方法，不证明多校问题无解。','',
        '变对/变错的具体案例与官网原文（地图只作固定评分，不等于入学资格）：'])
    for change in changes:
        if change['before_correct']==change['after_correct']:continue
        lines.extend(['',f"{change['district']}｜{change['name']}｜{'修正' if change['after_correct'] else '误伤'}｜语义待核={change['semantic_pending']}",
            '新增学校：'+'；'.join(change['added']),'原预测：'+'；'.join(change['before']),
            '本轮预测：'+'；'.join(change['after']),'地图评分集合：'+'；'.join(change['map_scoring_set'])])
        for quote in change['original_scope_quotes']:
            lines.append(quote['school']+'原文：'+quote['text'])
            for src in quote['source']:lines.append(f"{src.get('source_position')}｜{src.get('source_url')}")
    lines.extend(['','验证：10项单元测试通过；原始根候选展开专项检查通过；两阶段各39398条输出/评分一致性检查通过；12个查询/实验臂组合用标量选择器独立复算通过。',
        '所有输入、代码、预测和Gold哈希校验通过；无正式模型默认值修改。',
        f'修复后结果：{R.OUT}',f'修复前记录：{R.BASE}',
        f'运行入口：{F.REPO / "scripts/experiment_shenzhen_full_prior_repair.py"}',
        '顺序：prepare → infer → score → audit。已封存目录拒绝覆盖。'])
    F.DOC.mkdir(parents=True,exist_ok=True)
    (F.DOC/'全量实验结果与政策原文.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(F.DOC/'逐条变化_仅诊断.json',dict(changes=changes,not_training_input=True))
    write(F.DOC/'全量结果.json',dict(repaired=result,before_repair=before,previous_multi_strata=strata,additional_audit=audit_extra))
    write(F.DOC/'实际使用入口.json',dict(script=str(F.REPO/'scripts/experiment_shenzhen_full_prior_repair.py'),
        outputs=str(R.OUT),previous_stage=str(R.BASE),adopted=False,independent_validation=False))
    write(F.DOC/'交付校验.json',{str(p):sha(p) for p in F.DOC.iterdir() if p.is_file() and p.name!='交付校验.json'})
    print('DELIVERED',F.DOC,'STRATA',strata['full_scope_minimum'],'AUDIT',audit_extra,flush=True)


if __name__=='__main__':main()
