"""Round 01: freeze threshold-free policy-set posterior then score separately."""
import argparse
import pickle
import shutil
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
from zoning import shenzhen_policy as SP
from zoning.evaluate import bootstrap
from shenzhen_set_decode import decode

BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
OUT=F.REPO/'outputs/shenzhen_resolution_20261112/round01_set_decode'
TARGETS=['龙华区','龙岗区']


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Keep frozen round')
    assert F.read(BASE/'verification.json')['passed']
    frozen=F.read(BASE/'prediction_freeze.json')
    for d,h in frozen['probability_sha256'].items():assert F.sha(BASE/'probabilities'/f'{d}.npz')==h
    for rel,h in frozen['prediction_sha256'].items():assert F.sha(BASE/rel)==h
    config=F.read(BASE/'protocol.json')
    protocol=dict(round=1,date='2026-11-12',hypothesis='Identical policy sets split posterior mass across schools; set MAP may repair decoding without retraining.',
        baseline=str(BASE),candidate='sum frozen school probabilities ONLY over roots with identical complete policy-expanded sets; MAP; ties keep baseline',
        labels=['ensemble']+[str(s) for s in config['seeds']],hyperparameters=[],retraining=False,
        protected='all existing rule-known outputs; unchanged population, coordinates, original map Gold and policy rules',
        limitations='Pushforward assumes existing single-school-root policy mappings can represent the answer. It is not calibrated multilabel probability or eligibility truth. Candidate coverage audit is diagnostic only.',
        target_gold_used_to_build=False,independent_validation=False,
        adoption_gate='no automatic adoption; inspect exact-set improvement, both districts, FP/FN, corrected/harmed and 1km paired intervals; independent confirmation still required',
        research_basis=[dict(title='Estimating Multi-label Accuracy using Labelset Distributions',url='https://arxiv.org/abs/2209.04163',use='joint mode corresponds to exact match; our pushforward is an application hypothesis, not reproduction'),
            dict(title='Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains',url='https://icml.cc/Conferences/2010/papers/589.pdf',use='loss-sensitive joint-label inference; no PCC training claim')])
    F.write(OUT/'protocol.json',protocol)
    for src in [Path(__file__),F.REPO/'scripts/shenzhen_set_decode.py',F.REPO/'tests/test_shenzhen_set_decode.py',F.REPO/'zoning/shenzhen_policy.py']:
        dest=OUT/'code'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    changes={};inputs={}
    for d in TARGETS:
        p=SP.load_policy(d);meta=F.read(BASE/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
        dummy=dict(coordinate_group_id='source_only_empty_name',building_name='',coordinate_group_building_names=[])
        root_sets=[SP.apply(p,dummy,[s])['predicted_school_ids'] for s in sid]
        grounded=SP.predict_district(d)
        fixed={q:r['predicted_school_ids'] for q,r in enumerate(grounded) if r['predicted_school_ids']}
        for q,ss in meta['known'].items():
            q=int(q);fixed[q]=SP.apply(p,meta['sites'][q],[sid[s] for s in ss])['predicted_school_ids']
        z=np.load(BASE/'probabilities'/f'{d}.npz');probabilities=z['probs']
        assert probabilities.shape==(10,len(meta['sites']),len(sid))
        changes[d]={}
        for label,probs in [('ensemble',probabilities.mean(0))]+[(str(seed),p0) for seed,p0 in zip(z['seeds'],probabilities)]:
            chosen,mass,sets=decode(probs,root_sets)
            old=F.read(BASE/'predictions'/label/f'{d}.json')
            prior={r['coordinate_group_id']:r['predicted_school_ids'] for r in old['decisions']}
            rows=[]
            for q,site in enumerate(meta['sites']):
                key=site['coordinate_group_id'];oldpred=fixed[q] if q in fixed else root_sets[int(probs[q].argmax())]
                assert set(oldpred)==set(prior[key]),('baseline parity',d,label,key)
                pred=fixed[q] if q in fixed else list(chosen[q])
                candidate_index=next(j for j,s in enumerate(root_sets) if set(s)==set(chosen[q]))
                actual=SP.apply(p,site,[sid[candidate_index]])['predicted_school_ids']
                if q not in fixed:assert set(actual)==set(pred),('site condition parity',key)
                rows.append(dict(district=d,site_id=key,building_name=site['building_name'],before=prior[key],after=pred,
                    rule_known=q in fixed,candidate_sets=[fixed[q]] if q in fixed else sets,
                    changed=set(pred)!=set(prior[key])))
            changes[d][label]=sum(r['changed'] for r in rows)
            F.write(OUT/'predictions'/label/f'{d}.json',rows)
        inputs[d]=dict(policy_hash=F.sha(F.SZ.DATA/d/'政策规则/规则包.json'),probabilities_hash=F.sha(BASE/'probabilities'/f'{d}.npz'),
            metadata_hash=F.sha(BASE/'inputs'/f'{d}.json'),rule_protected_sites=len(fixed),policy_root_sets=len(sets),schools=len(sid))
    F.write(OUT/'input_audit.json',dict(districts=inputs,changed_predictions=changes,baseline_parity_passed=True,target_gold_read=False))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('PREDICTIONS FROZEN',changes,flush=True)


def metrics(rows,key):
    return dict(n=len(rows),correct=sum(set(r[key])==set(r['gold']) for r in rows),
        exact_accuracy=sum(set(r[key])==set(r['gold']) for r in rows)/len(rows) if rows else None,
        fp=sum(len(set(r[key])-set(r['gold'])) for r in rows),fn=sum(len(r['gold']-set(r[key])) for r in rows))


def score():
    freeze=F.read(OUT/'prediction_freeze.json')
    for rel,h in freeze['files'].items():assert F.sha(OUT/rel)==h,rel
    gold_rows=F.read(BASE/'scored_cases.json');gold={(r['district'],r['site_id']):r for r in gold_rows}
    assert len(gold)==1802
    baseline_results=F.read(BASE/'results.json')
    for d,h in baseline_results['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    xy={}
    for d in TARGETS:
        with (F.REPO/'data/深圳市/规则对齐输入_20261111'/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    reports={};case_audit=[]
    for label in F.read(OUT/'protocol.json')['labels']:
        rows=[]
        for d in TARGETS:
            pred=F.read(OUT/'predictions'/label/f'{d}.json')
            for r in pred:
                g=gold[d,r['site_id']]
                if not g['has_map_gold']:continue
                r['gold']=set(g['gold']);r['semantic_pending']=g['semantic_pending']
                pt=xy[d,r['site_id']];r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));r['d']=d
                rows.append(r)
        assert len(rows)==1761
        out={}
        for name,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
            ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
            ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
            delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
            out[name]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'),corrected=delta.count(1),harmed=delta.count(-1),
                delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000),
                representable=sum(any(set(s)==r['gold'] for s in r['candidate_sets']) for r in rs))
        reports[label]=out
        if label=='ensemble':
            for r in rows:
                case_audit.append(dict(district=r['district'],site_id=r['site_id'],building_name=r['building_name'],
                    gold=sorted(r['gold']),before=r['before'],after=r['after'],changed=r['changed'],rule_known=r['rule_known'],
                    representable=any(set(s)==r['gold'] for s in r['candidate_sets']),semantic_pending=r['semantic_pending']))
    F.write(OUT/'results.json',dict(reports=reports,independent_validation=False,adopted=False))
    F.write(OUT/'diagnostic_cases_not_training.json',case_audit)
    lines=['第一轮：固定模型概率，按已有政策映射聚合完整集合（2026-11-12）','',
        '原图Gold不变；没有重训、阈值选择或根据本轮目标分数选择方案。既往目标分数已知，属于开发实验。',
        '集合可表示数是评分后诊断上限，不是可部署准确率，不作为参照或训练输入。','',
        '分组\t旧正确数\t新正确数\t样本数\t新准确率\t改正/损害\t候选集合可表示数']
    for key,r in reports['ensemble'].items():
        lines.append(f"{key}\t{r['before']['correct']}\t{r['after']['correct']}\t{r['after']['n']}\t{r['after']['exact_accuracy']:.2%}\t{r['corrected']}/{r['harmed']}\t{r['representable']}")
    r=reports['ensemble']['全部'];lines += ['',f"变化 {r['delta_pp']:+.2f} 个百分点；95%空间配对区间 {r['delta_95_pp']}。",f"误报 {r['before']['fp']}→{r['after']['fp']}；漏报 {r['before']['fn']}→{r['after']['fn']}。",'未自动采用；还需解决无法表示的范围、训练监督和独立确认。']
    (OUT/'结果与候选覆盖诊断.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'verification.json',dict(passed=True,frozen_files=len(freeze['files']),seeds=10,scored_sites=1761,gold_hashes_unchanged=True,baseline_parity=True))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);a=p.parse_args()
    dict(prepare=prepare,score=score)[a.phase]()
