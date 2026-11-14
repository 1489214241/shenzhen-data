"""Frozen same-budget rule-alignment rerun. Separate prepare/train/score processes."""
import argparse
import collections
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import experiment_shenzhen_full as R
import shenzhen_full_inputs as F
from zoning.evaluate import bootstrap

OLD=R.OUT
OUT=F.REPO/'outputs/shenzhen_aligned_model_20261111'
INPUT=F.REPO/'data/深圳市/规则对齐输入_20261111'
R.OUT=R.B.OUT=OUT
F.OUT=INPUT


def prepare():
    F.guard()
    manifest=F.read(INPUT/'manifest.json')
    for rel,h in manifest['files'].items():assert F.sha(INPUT/rel)==h,rel
    # The source cache and all training hyperparameters are identical to the prior full run.
    cache=F.read(R.CACHE/'manifest.json')
    assert cache['config']==F.read(F.REPO/'configs/default.json')
    for rel,h in cache['files'].items():assert F.sha(R.CACHE/rel)==h,rel
    audits={}
    for d in R.B.TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:new=pickle.load(f)
        with (F.REPO/'data/深圳市/完整输入_20261111'/d/'district.pkl').open('rb') as f:old=pickle.load(f)
        assert new.site_ids==old.site_ids and new.school_ids==old.school_ids
        np.testing.assert_array_equal(new.coords,old.coords)
        np.testing.assert_array_equal(new.school_xy,old.school_xy)
        assert new.answers=={} and new.train_answers=={}
        rows=F.read(INPUT/d/'policy_observations.json')
        assert len(rows)==len(new.school_ids)
        admin=F.read(INPUT/d/'admin_geometry_audit.json')
        ring=F.read(INPUT/d/'ring_geometry_audit.json')
        audits[d]=dict(schools=len(rows),school_population_unchanged=True,site_population_unchanged=True,coordinates_unchanged=True,
            known_before=F.read(F.REPO/'data/深圳市/完整输入_20261111'/d/'input_summary.json')['known_sites'],
            known_after=F.read(INPUT/d/'input_summary.json')['known_sites'],
            root_ops=dict(collections.Counter(r['ast']['op'] for r in rows)),
            unknown_syntax_schools=sum(any(v['op']=='unknown' for v in r['unresolved']) for r in rows),
            pending_status_schools=sum(r['scope_status'].endswith('_pending') for r in rows),
            independent_admin_resolved=sum(r['status']=='independent_osm_boundary' for r in admin),admin_queries=len(admin),
            unique_rings=sum(r['status']=='unique_closed_ring' for r in ring),ring_queries=len(ring))
    R.prepare()
    c=F.read(OUT/'protocol.json')
    c.update(status='frozen_rule_alignment_before_training_and_target_scoring',
        experiment='Changsha shared policy primitives + Shenzhen grammar adapter + independent OSM admin boundaries',
        original_full_run=str(OLD.relative_to(F.REPO)),
        common_rule_executor='zoning.rules.school / zoning.rules.sites, unified_20261106 (no Tianxin-specific school-index repair)',
        common_name_executor='Exact source functions from name_matching.py, longhua/rules.py, address_scope.py, v3_names.py; source hashes in input manifest',
        preserved_shenzhen_semantics='No university expansion; local directed shared-scope supplementation, no transitive closure',
        alignment_boundary='Executable evidence and safeguards aligned; missing geometry, person/household facts, ambiguous identities, and unsupported composite clauses stay unknown. Unknown is not lack of official text.',
        limitations='OSM admin polygons are independent public-map observations, not certified enrollment truth. Office POIs remain proxy features when no exact administrative polygon exists. This bundled adapter/data correction is not a parser-only causal ablation.',
        adoption='development rerun after prior scores were known; no post-score tuning and no independent-validation claim')
    F.write(OUT/'protocol.json',c)
    alignment=dict(checks=audits,shared_modules=dict(names='native functions',phase='native functions',address='native functions',exclusion_similarity='native functions',
        side='native strict_line',three_valued='native AND/OR',clause_conflict='native school/clause',site_priority='native sites',
        admin='same contains/covers + 50m boundary band, independent OSM source',ring='exact-name closed line network, unique face, 50m band; no guessed directions',
        school_references='explicit source name reference, ambiguity/cycles stay unknown',shared='existing sourced local-only rules',university='no member expansion'),
        unresolved_is_not_negative=True,additional_verified_target_supports=0,target_gold_read=False,target_catchment_geometry_read=False)
    F.write(OUT/'对齐核查.json',alignment)
    code=[Path(__file__),F.REPO/'scripts/build_shenzhen_aligned_inputs.py',F.REPO/'scripts/shenzhen_aligned_policy.py',F.REPO/'scripts/shenzhen_admin_from_pbf.py',F.REPO/'scripts/audit_shenzhen_alignment_inputs.py',F.REPO/'tests/test_shenzhen_aligned_policy.py']
    for p in code:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    frozen=F.read(OUT/'input_manifest.json')
    for p in [OUT/'protocol.json',OUT/'对齐核查.json']+list((OUT/'code').rglob('*.py')):
        frozen['frozen_sha256'][str(p.relative_to(OUT))]=F.sha(p)
    F.write(OUT/'input_manifest.json',frozen)
    text=['深圳与长沙规则对齐核查（2026-11-12 完成输入复核）','',
        '直接调用长沙原有名称、分期、门牌、排除相似性函数，以及现用统一规则执行器。',
        '复用原学校、小区坐标和道路快照；从同一OSM PBF补取独立行政边界。',
        '模型、41/27特征、8个来源区、10种子、40步、额外人工核验参照K=0与旧完整输入运行一致。',
        '大学区不扩展原地段；共享只按官方条款指定范围单向补充，不传递。',
        '规则能力对齐不代表全部官方文字都能确定执行：缺失边界、户籍条件、别名及复杂路段保持未知。',
        '这是既有目标分数已知后的开发修复实验；不作为独立验证或自动采纳。','']
    for d,a in audits.items():text.append(f"{d}：参照位置 {a['known_before']}→{a['known_after']}；仍有未解析子句的学校 {a['unknown_syntax_schools']}；独立行政边界 {a['independent_admin_resolved']}/{a['admin_queries']}；唯一闭合范围 {a['unique_rings']}/{a['ring_queries']}。")
    (OUT/'对齐清单.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import audit_shenzhen_alignment_inputs
    audit_shenzhen_alignment_inputs.main()
    R.B.log('ALIGNED_INPUTS_FROZEN',audits)


def score():
    R.B.interval=R.interval
    R.B.score()
    result=F.read(OUT/'results.json');old_result=F.read(OLD/'results.json')
    assert result['gold_sha256']==old_result['gold_sha256']
    rows=F.read(OUT/'scored_cases.json');old_rows=F.read(OLD/'scored_cases.json')
    old={(r['district'],r['site_id']):r for r in old_rows}
    assert len(rows)==len(old)==1802
    coords={}
    for d in R.B.TARGETS:
        with (INPUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        coords.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    paired=[];deltas=[];examples=[]
    for r in rows:
        prior=old[r['district'],r['site_id']]
        assert r['gold']==prior['gold'] and r['has_map_gold']==prior['has_map_gold']
        if not r['has_map_gold']:continue
        p=coords[r['district'],r['site_id']]
        a=set(r['after'])==set(r['gold']);b=set(prior['after'])==set(prior['gold'])
        paired.append(dict(d=r['district'],block=str((int(p[0]//1000),int(p[1]//1000)))))
        deltas.append(int(a)-int(b))
        examples.append(dict(district=r['district'],site_id=r['site_id'],building_name=r['building_name'],before=prior['after'],after=r['after'],gold=r['gold'],corrected=bool(a and not b),harmed=bool(b and not a)))
    assert len(paired)==1761
    ci=bootstrap(paired,deltas,seed=20261104,draws=3000)
    comparison=dict(old_run=str(OLD),new_run=str(OUT),n=len(paired),corrected=sum(v==1 for v in deltas),harmed=sum(v==-1 for v in deltas),
        delta_accuracy=float(np.mean(deltas)),delta_95_percentile_pp=list(ci),gold_sha256_unchanged=True,
        same_source_cache=True,same_training_budget=True,independent_validation=False)
    F.write(OUT/'与旧完整输入对照.json',comparison);F.write(OUT/'逐例改正与损害.json',examples)
    lines=['深圳规则对齐后重跑结果（2026-11-12；沿用 20261111 实验目录）','',
        '严格整套学校集合评分；同一1761个有地图命中的坐标组，41个图外组不评分；全部1802组均有预测。',
        '旧运行保留；本次名称/范围适配和独立行政边界共同变化，不能归因给某一单项。','',
        '地区\t旧完整输入\t规则对齐后\t正确数/样本数']
    for d in R.B.TARGETS:
        a=result['districts'][d]['all_map']['after'];b=old_result['districts'][d]['all_map']['after']
        lines.append(f"{d}\t{b['exact_accuracy']:.2%}\t{a['exact_accuracy']:.2%}\t{a['correct']}/{a['n']}")
    a=result['overall']['after'];b=old_result['overall']['after']
    lines.append(f"合计\t{b['exact_accuracy']:.2%}\t{a['exact_accuracy']:.2%}\t{a['correct']}/{a['n']}")
    lines+=['',f"相较旧完整输入，改正{comparison['corrected']}，损害{comparison['harmed']}；变化{comparison['delta_accuracy']*100:+.2f}个百分点。",f"按区分层1km空间块配对bootstrap 95%区间（百分点）：{ci}",
        f"学校误报数 {b['false_positive_school_assignments']}→{a['false_positive_school_assignments']}；漏报数 {b['false_negative_school_assignments']}→{a['false_negative_school_assignments']}。",
        '','仍有无法完整执行的官方条款。详见对齐清单及输入目录逐校policy_observations.json、道路/行政边界审计。',
        '保持原图Gold，不使用学区polygon制造输入参照；结果是开发复测，不代表独立验证成功或正式采纳。']
    (OUT/'实验结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result['status']='completed_rule_alignment_development_rerun';F.write(OUT/'results.json',result)
    print('\n'.join(lines),flush=True)


def verify():
    R.B.check_frozen()
    m=F.read(INPUT/'manifest.json')
    for rel,h in m['files'].items():assert F.sha(INPUT/rel)==h,rel
    frozen=F.read(OUT/'prediction_freeze.json')
    for rel,h in frozen['prediction_sha256'].items():assert F.sha(OUT/rel)==h,rel
    for d,h in frozen['probability_sha256'].items():assert F.sha(OUT/'probabilities'/f'{d}.npz')==h
    history=F.read(OUT/'training_history.json');assert len(history)==20
    for key,v in history.items():
        d,seed=key.split('/');assert len(v['loss'])==40 and np.isfinite(v['loss']).all()
        assert F.sha(OUT/'checkpoints'/f'{d}_{seed}.pt')==v['checkpoint_sha256']
    for d in R.B.TARGETS:
        z=np.load(OUT/'probabilities'/f'{d}.npz');assert z['probs'].shape[0]==10
        assert np.isfinite(z['probs']).all() and np.allclose(z['probs'].sum(-1),1,atol=1e-5)
    F.write(OUT/'verification.json',dict(passed=True,checkpoints=20,steps_each=40,source_input_hashes_unchanged=True,all_prediction_hashes_unchanged=True,aligned_input_files=len(m['files'])))
    print('VERIFIED 20 CHECKPOINTS; INPUTS AND PREDICTIONS UNCHANGED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','score','verify']);a=p.parse_args()
    dict(prepare=prepare,train=R.B.train,score=score,verify=verify)[a.phase]()
