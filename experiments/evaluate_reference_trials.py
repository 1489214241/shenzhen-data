"""Dependency-free scoring and budget audit for the complete-reference trials.

Uses district Gold, released reference lists and frozen predictions. This is
result verification, not training or evidence of independent generalization.
"""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def check_rows(rows,expected,gold,keys):
    by={(r['district'],r['site_id']):r for r in rows}
    assert len(by)==len(rows) and set(by)==keys
    correct=fp=fn=0
    for key,r in by.items():
        truth=gold[key];pred=set(r['prediction'])
        assert truth==set(r['truth']),key
        correct+=truth==pred;fp+=len(pred-truth);fn+=len(truth-pred)
    assert (len(rows),correct,fp,fn)==tuple(expected[k] for k in ['n','correct','fp','fn'])
    return dict(n=len(rows),correct=correct,exact_percent=100*correct/len(rows))

def main():
    gold={}
    for d in ['龙华区','龙岗区']:
        for r in read(ROOT/d/'Gold/小区_学校Gold.json'):gold[d,r['coordinate_group_id']]=set(r['point_school_ids'])
    output={};old=ROOT/'experiments/modern_complete_reference_20261114'
    if old.exists():
        plan=read(old/'public_plan.json');support=read(old/'released_support.json')
        keys={(d,r['site_id']) for d,v in plan.items() for r in v['rows'] if not r['selected']}
        assert len(keys)==1249 and sum(len(v) for v in support.values())==512
        for d,rr in support.items():
            for r in rr:assert set(r['schools'])==gold[d,r['site_id']]
        output['fixed512']={a:check_rows(read(old/'scored_rows'/f'{a}.json'),expected['all'],gold,keys) for a,expected in read(old/'results.json')['metrics'].items()}
    run=ROOT/'experiments/modern_school_gap_reference_20261114'
    plans=read(run/'released_plans.json');public=read(run/'public_population.json');verified=read(run/'verified_partial_references.json');expected=read(run/'results.json')['budgets']
    output['school_gap']={}
    for k in ['1','2','3']:
        keys=set();common=set();cost=0
        for d,budgets in plans.items():
            p=budgets[k];refs=p['support'];cost+=len(refs)
            assert len(refs)==len({r['group'] for r in refs})
            excluded=set(public[d]['base_reference_groups'])|{r['group'] for r in refs}
            assert not excluded&{r['group'] for r in p['query']}
            counts={s['school_id']:set() for s in public[d]['schools']}
            for r in verified[d]:counts[r['school']].add(r['group'])
            for r in refs:
                assert set(r['schools'])==gold[d,r['site_id']]
                for s in r['schools']:counts[s].add(r['group'])
            assert {s:len(v) for s,v in counts.items()}==p['counts']
            assert {s:int(k)-len(v) for s,v in counts.items() if len(v)<int(k)}==p['deficits']
            if k!='1':assert {r['site_id'] for r in budgets[str(int(k)-1)]['support']}<={r['site_id'] for r in refs}
            keys.update((d,r['site_id']) for r in p['query']);common.update((d,s) for s in p['common_ids'])
            nn={r['site_id']:set(r['prediction']) for r in read(run/'scored_rows'/k/'nearest_raw.json') if r['district']==d}
            for q in p['query']:
                legal=[r for r in refs if r['group']!=q['group']]
                if legal:
                    closest=min(legal,key=lambda r:math.dist(q['xy'],r['xy']))
                    assert nn[q['site_id']]==set(closest['schools'])
        result={'additional_unique_verifications':cost,'natural':{},'common':{}}
        for a,e in expected[k]['natural']['metrics'].items():
            rows=read(run/'scored_rows'/k/f'{a}.json')
            result['natural'][a]=check_rows(rows,e['all'],gold,keys)
            result['common'][a]=check_rows([r for r in rows if (r['district'],r['site_id']) in common],expected[k]['common']['metrics'][a]['all'],gold,common)
        output['school_gap'][k]=result
    zero=run/'zero_budget_control'
    if zero.exists():
        output['zero_new_reference_control']=check_rows(read(zero/'scored_rows.json'),read(zero/'results.json')['all'],gold,common)
    print(json.dumps(dict(passed=True,scope='Frozen scoring, reference costs and group isolation; not retraining',results=output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
