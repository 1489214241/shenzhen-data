"""Independent counts, complete-support budget and nearest-neighbor replay."""
from pathlib import Path
import json,hashlib,collections
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/modern_school_gap_reference_20261114'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for p,h in read(OUT/'prediction_freeze.json').items():assert sha(OUT/p)==h,p
    seal=read(OUT/'selection_freeze.json');assert sha(OUT/'released_plans.json')==seal['plans']
    plans=read(OUT/'released_plans.json');public=read(OUT/'public_population.json');verified=read(OUT/'verified_partial_references.json')
    gold={(r['district'],r['site_id']):set(r['gold']) for r in read(ROOT/'outputs/shenzhen_aligned_model_20261111/scored_cases.json') if r['has_map_gold']}
    results=read(OUT/'results.json')['budgets'];count=collections.Counter();replayed=0;audit={}
    for d,budgets in plans.items():
        common0=None;previous=set();audit[d]={}
        for k in ['1','2','3']:
            plan=budgets[k];support=plan['support'];common=set(plan['common_ids']);query={r['site_id']:r for r in plan['query']}
            assert common<=set(query)
            if common0 is not None:assert common==common0
            common0=common
            selected={r['site_id'] for r in support};assert previous<=selected;previous=selected
            assert len(selected)==len({r['group'] for r in support})==len(support)
            assert not {r['group'] for r in support if r['validation']}&{r['group'] for r in support if not r['validation']}
            excluded=set(public[d]['base_reference_groups'])|{r['group'] for r in support}
            assert not excluded&{r['group'] for r in query.values()}
            counts={s['school_id']:set() for s in public[d]['schools']}
            for r in verified[d]:counts[r['school']].add(r['group'])
            for r in support:
                assert set(r['schools'])==gold[d,r['site_id']] and r['complete']
                for s in r['schools']:counts[s].add(r['group'])
            assert {s:len(v) for s,v in counts.items()}==plan['counts']
            assert {s:int(k)-len(v) for s,v in counts.items() if len(v)<int(k)}==plan['deficits']
            pred=read(OUT/'predictions'/k/f'{d}.json')
            for arm,rows in pred.items():
                assert len(rows)==len(query) and {r['site_id'] for r in rows}==set(query)
                for r in rows:
                    ok=set(r['after'])==gold[d,r['site_id']]
                    count[k,'natural',arm]+=ok
                    if r['site_id'] in common:count[k,'common',arm]+=ok
            nn={r['site_id']:set(r['after']) for r in pred['nearest_raw']}
            for r in read(OUT/'nearest_provenance'/k/f'{d}.json'):
                q=query[r['query']];legal=[s for s in support if s['group']!=q['group']]
                best=min(legal,key=lambda s:np.linalg.norm(np.array(q['xy'])-s['xy']))
                assert best['site_id']==r['reference'] and nn[r['query']]==set(best['schools']);replayed+=1
            audit[d][k]=dict(new_unique_verifications=len(support),schools_with_gap=len(plan['deficits']),query_natural=len(query),query_common=len(common))
    for (k,pop,a),n in count.items():assert n==results[k][pop]['metrics'][a]['all']['correct'],(k,pop,a)
    out=dict(passed=True,budgets=audit,nearest_predictions_replayed=replayed,all_methods_independently_scored=True,
        all_reference_project_groups_excluded_from_queries=True,nested_budget_lists=True,partial_policy_facts_not_promoted_to_complete=True,
        independent_geography_validation=False)
    (OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
