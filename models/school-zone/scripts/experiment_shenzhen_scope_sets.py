"""Round 02: source-only Boolean scope inclusion, fixed weights and population."""
import argparse
import pickle
import shutil
from pathlib import Path
import numpy as np
import shenzhen_full_inputs as F
from zoning import shenzhen_policy as SP
from zoning.evaluate import bootstrap
from shenzhen_set_decode import decode
from shenzhen_scope_implication import build, expand_original
from experiment_shenzhen_set_decode import metrics
from shenzhen_aligned_policy import Parser
from build_shenzhen_aligned_inputs import parse_scope

BASE = F.REPO / 'outputs/shenzhen_aligned_model_20261111'
PREVIOUS = F.REPO / 'outputs/shenzhen_resolution_20261112/round01_set_decode'
OUT = F.REPO / 'outputs/shenzhen_resolution_20261112/round02_scope_inclusion_v2'
INPUT = F.REPO / 'data/深圳市/规则对齐输入_20261111'
TARGETS = ['龙华区', '龙岗区']


def prepare():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve frozen experiment')
    oldfreeze = F.read(PREVIOUS / 'prediction_freeze.json')
    for rel, digest in oldfreeze['files'].items(): assert F.sha(PREVIOUS / rel) == digest, rel
    basefreeze = F.read(BASE / 'prediction_freeze.json')
    config = F.read(BASE / 'protocol.json')
    labels = ['ensemble'] + [str(s) for s in config['seeds']]
    protocol = dict(round=2, date='2026-11-12', baseline=str(PREVIOUS),
        candidate='Direct Boolean scope inclusion from complete unconditional source ASTs; root-only single pass, set-posterior MAP.',
        hypothesis='Literal whole-scope equality misses one school whose entire scope equals one branch of another school.',
        parser_review_fix='Terminal parenthesis on cardinal scope binds to the full conjunction, not an OR of individual sides. Reparse all source scopes; retain old frozen model features and weights.',
        labels=labels, hyperparameters=[], retraining=False, target_gold_used_to_build=False,
        exclusions='University, eligibility, pending, unknown syntax, references, unresolved aliases: no new implications.',
        protected='Same rule-known outputs as round01; do not promote partial references to full labels.',
        limitations='Soundness is relative to parsed source expressions. It does not establish correct school root, policy interpretation, map freshness or household eligibility.',
        independent_validation=False, adopted=False,
        adoption_gate='Full population, both districts, FP/FN, corrected/harmed, spatial intervals; independent confirmation needed. No automatic adoption.')
    F.write(OUT / 'protocol.json', protocol)
    for src in [Path(__file__), F.REPO / 'scripts/shenzhen_scope_implication.py',
                F.REPO / 'scripts/shenzhen_set_decode.py', F.REPO / 'tests/test_shenzhen_scope_implication.py',
                F.REPO / 'scripts/shenzhen_aligned_policy.py', F.REPO / 'scripts/build_shenzhen_aligned_inputs.py',
                F.REPO / 'tests/test_shenzhen_aligned_policy.py',
                F.REPO / 'zoning/shenzhen_policy.py']:
        dest = OUT / 'code' / src.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dest)
    audit = {}
    for d in TARGETS:
        policy = SP.load_policy(d)
        meta = F.read(BASE / 'inputs' / f'{d}.json')
        observations = F.read(INPUT / d / 'policy_observations.json')
        parser = Parser(observations, parse_scope)
        reparsed = [dict(r, ast=parser.parse(r['source_text'])) for r in observations]
        parse_changes = [dict(school_id=a['school_id'], school_name=a['school_name'], source_text=a['source_text'],
                              before=a['ast'], after=b['ast']) for a,b in zip(observations,reparsed) if a['ast']!=b['ast']]
        observations = reparsed
        F.write(OUT / 'source_graphs' / f'{d}_parse_review.json', parse_changes)
        graph = build(observations, policy)
        F.write(OUT / 'source_graphs' / f'{d}.json', graph)
        sid = [s['school_id'] for s in meta['schools']]
        dummy = dict(coordinate_group_id='empty', building_name='', coordinate_group_building_names=[])
        oldroots = [SP.apply(policy, dummy, [s])['predicted_school_ids'] for s in sid]
        newroots = [sorted(set(old) | set(expand_original([s], graph))) for s, old in zip(sid, oldroots)]
        assert F.sha(BASE / 'probabilities' / f'{d}.npz') == basefreeze['probability_sha256'][d]
        z = np.load(BASE / 'probabilities' / f'{d}.npz')
        assert z['seeds'].tolist() == config['seeds']
        changes = {}
        for label, probs in [('ensemble', z['probs'].mean(0))] + [(str(s), p) for s, p in zip(z['seeds'], z['probs'])]:
            prior = F.read(PREVIOUS / 'predictions' / label / f'{d}.json')
            oldchosen, _, _ = decode(probs, oldroots)
            chosen, _, candidates = decode(probs, newroots)
            rows = []
            assert len(prior) == len(meta['sites'])
            for q, r in enumerate(prior):
                assert r['site_id'] == meta['sites'][q]['coordinate_group_id']
                if not r['rule_known']: assert set(oldchosen[q]) == set(r['after']), (d, label, q)
                pred = r['after'] if r['rule_known'] else list(chosen[q])
                rows.append(dict(district=d, site_id=r['site_id'], building_name=r['building_name'],
                    before=r['after'], after=pred, rule_known=r['rule_known'],
                    candidate_sets=[pred] if r['rule_known'] else candidates,
                    changed=set(pred) != set(r['after'])))
            F.write(OUT / 'predictions' / label / f'{d}.json', rows)
            changes[label] = sum(r['changed'] for r in rows)
        audit[d] = dict(schools=len(sid), admitted_schools=sum(r['admitted'] for r in graph['inventory']),
            reparsed_scope_changes=len(parse_changes),
            direct_edges=len(graph['edges']), changed_root_sets=sum(set(a)!=set(b) for a,b in zip(oldroots,newroots)),
            changed_predictions=changes, source_observations_hash=F.sha(INPUT / d / 'policy_observations.json'),
            policy_hash=F.sha(F.SZ.DATA / d / '政策规则/规则包.json'),
            probabilities_hash=F.sha(BASE / 'probabilities' / f'{d}.npz'))
    F.write(OUT / 'input_audit.json', dict(districts=audit, target_gold_read=False, prior_prediction_parity=True))
    F.write(OUT / 'prediction_freeze.json', dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}, target_gold_read=False))
    print('FROZEN', audit, flush=True)


def score():
    freeze = F.read(OUT / 'prediction_freeze.json')
    for rel, digest in freeze['files'].items(): assert F.sha(OUT / rel) == digest, rel
    original = F.read(BASE / 'results.json')
    for d, h in original['gold_sha256'].items(): assert F.sha(F.SZ.DATA / d / 'Gold/小区_学校Gold.json') == h
    gold = {(r['district'],r['site_id']):r for r in F.read(BASE / 'scored_cases.json')}
    xy = {}
    for d in TARGETS:
        with (INPUT / d / 'district.pkl').open('rb') as f: dist = pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    reports = {}; cases = []
    for label in F.read(OUT / 'protocol.json')['labels']:
        rows = []
        for d in TARGETS:
            for r in F.read(OUT / 'predictions' / label / f'{d}.json'):
                g = gold[d,r['site_id']]
                if not g['has_map_gold']: continue
                r['gold'] = set(g['gold']); r['semantic_pending'] = g['semantic_pending']
                pt=xy[d,r['site_id']]; r['block']=str((int(pt[0]//1000),int(pt[1]//1000))); r['d']=d
                rows.append(r)
        assert len(rows)==1761
        report={}
        for name, rs in [('全部',rows)] + [(d,[r for r in rows if r['district']==d]) for d in TARGETS] + [
                ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
            delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
            report[name]=dict(before=metrics(rs,'before'),after=metrics(rs,'after'), corrected=delta.count(1),harmed=delta.count(-1),
                delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000),
                representable=sum(any(set(s)==r['gold'] for s in r['candidate_sets']) for r in rs))
        reports[label]=report
        if label=='ensemble':
            cases=[dict(district=r['district'],site_id=r['site_id'],building_name=r['building_name'],gold=sorted(r['gold']),
                before=r['before'],after=r['after'],changed=r['changed'],rule_known=r['rule_known'],semantic_pending=r['semantic_pending'],
                representable=any(set(s)==r['gold'] for s in r['candidate_sets'])) for r in rows]
    F.write(OUT / 'results.json',dict(reports=reports,independent_validation=False,adopted=False))
    F.write(OUT / 'diagnostic_cases_not_training.json',cases)
    lines=['第二轮：完整官方范围的直接包含关系（固定模型，不扩展大学区）','',
           '同一1761个样本，原图Gold不变。上一轮为集合概率合并，本轮只增加直接文字包含关系。',
           '尚属开发实验，未独立确认，不自动替换主线。','',
           '分组\t上轮正确\t本轮正确\t样本数\t本轮准确率\t改正/损害\t候选可表示数']
    for name,r in reports['ensemble'].items():
        lines.append(f"{name}\t{r['before']['correct']}\t{r['after']['correct']}\t{r['after']['n']}\t{r['after']['exact_accuracy']:.2%}\t{r['corrected']}/{r['harmed']}\t{r['representable']}")
    r=reports['ensemble']['全部']
    lines += ['',f"变化 {r['delta_pp']:+.2f} 个百分点；95%空间区间 {r['delta_95_pp']}。",
              f"误报 {r['before']['fp']}→{r['after']['fp']}；漏报 {r['before']['fn']}→{r['after']['fn']}。"]
    (OUT / '结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT / 'verification.json',dict(passed=True,frozen_files=len(freeze['files']),scored_sites=1761,seeds=10,
        target_gold_hashes_unchanged=True,prior_prediction_parity=True))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','score']);a=parser.parse_args()
    dict(prepare=prepare,score=score)[a.phase]()
