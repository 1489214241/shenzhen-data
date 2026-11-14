"""Descriptive validation error audit after the frozen screen; never selects models."""
import json
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_readout as run
from run_modernnca_relation import save,digest


def main():
    completed=json.loads((run.OUT/'completed.json').read_text(encoding='utf8'))
    seal=json.loads((run.OUT/'sealed_selection.json').read_text(encoding='utf8'))
    assert all(digest(run.OUT/p)==h for p,h in seal.items())
    frozen=json.loads((run.OUT/'frozen.json').read_text(encoding='utf8'))
    for p,h in frozen['source_hashes'].items():assert digest(REPO/p)==h,p
    for batch,h in frozen['input_hashes'].items():assert digest(run.base.SOURCE/batch/'inputs.pkl')==h,batch
    reference_checks=0
    for path in (run.OUT/'trials').glob('*_reference_*.json'):
        current=json.loads(path.read_text(encoding='utf8'))
        oldpath=REPO/'outputs/full_evidence_20261112/selection/trials'/path.name
        old=json.loads(oldpath.read_text(encoding='utf8'))
        lookup_steps={v['step']:v for v in old['history']}
        for row in current['history']:
            prior=lookup_steps[row['step']]
            assert all(row[k]==prior[k] for k in ['accuracy','nll']), (path,row,prior)
            reference_checks+=1
    assert reference_checks==128
    rows=json.loads((run.OUT/'validation_cases.json').read_text(encoding='utf8'))
    diagnosis=json.loads((run.OUT/'diagnosis.json').read_text(encoding='utf8'))
    lookup={(r['target'],r['q']):r for r in diagnosis['rows']}
    _,D,R,X=run.base.loaded(0,1)
    for r in rows:
        old=lookup[r['target'],r['q']]
        assert old['pred']==r['predictions']['identity']
        r.update(true_ref_mass=old['true_ref_mass'],baseline_margin=old['margin'])
        d=D[r['validation']]
        r.update(name=d.site_names[r['q']],truth_name=d.school_names[r['truth']],
                 predicted_names={k:d.school_names[v] for k,v in r['predictions'].items()})
    subsets={'all':rows,'zero_true_ref':[r for r in rows if r['true_ref_mass']==0],
             'with_true_ref':[r for r in rows if r['true_ref_mass']>0],
             'baseline_margin_below_0.2':[r for r in rows if r['baseline_margin']<.2],
             'baseline_margin_at_least_0.2':[r for r in rows if r['baseline_margin']>=.2]}
    report={}
    for name,rr in subsets.items():
        report[name]={f'{a}-{b}':run.common.comparison(rr,a,b) for a,b in run.PAIRS}
    changed=[r for r in rows if any(r['predictions']['content']!=r['predictions'][k] for k in ['identity','summary','reference'])]
    save(run.DOC/'验证错误核对.json',dict(subgroups=report,changed_cases=changed,
        input_and_frozen_source_hashes_checked=True,selection_seal_checked=True,
        reference_archive_identical_checkpoints=reference_checks,
        completed_sha256=digest(run.OUT/'completed.json'),script_sha256=digest(Path(__file__)),
        note='Descriptive postselection validation audit only. Repeated validation questions across training scenarios are not independent observations. True-school reference count is diagnostic, never a model input based on true query labels.'))
    lines=['验证错误核对（选择后描述，不作独立机制证明）','',
        '同一个验证区可能用于多个训练情景；下列数目是情景-题目，不是独立小区数。',
        '真实学校参照量只用于诊断，不作为偷看真实学校的模型特征。','']
    for name,z in report.items():
        lines.append(name)
        for p in ['content-identity','content-summary','content-reference']:
            v=z[p];lines.append(f"  {p}: n={v['n']}, 差{v['delta']:+.3f}点，改对/误伤{v['fixed']}/{v['broken']}")
    val=json.loads((run.DOC/'验证结果.json').read_text(encoding='utf8'))
    lines+=['','逐情景（括号内为实际评分的验证区）：']
    for d,v in run.base.VALIDATION.items():
        lines.append(f'{d}（{v}）')
        for pair in ['content-identity','content-summary','content-reference']:
            z=val['comparisons'][pair][d]
            lines.append(f"  {pair}: {z['accuracy_a']:.3f}% vs {z['accuracy_b']:.3f}%，改对/误伤{z['fixed']}/{z['broken']}")
    lines+=['','全部改判对象名称、学校和预测详见同目录验证错误核对.json。',
            f'RetrievalScorer与原归档重叠的{reference_checks}个检查点准确率和NLL完全一致。',
            '所有冻结输入、模型代码及所选配置哈希均已复核。没有利用这份诊断调整本轮模型。']
    (run.DOC/'验证错误核对.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
