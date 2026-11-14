"""Independent result and nearest-neighbor replay, without importing experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/modern_complete_reference_20261114'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    public=read(OUT/'public_plan.json');support=read(OUT/'released_support.json')
    for p,h in read(OUT/'prediction_freeze.json').items():assert sha(OUT/p)==h,p
    release=read(OUT/'release_receipt.json');assert sha(OUT/'released_support.json')==release['support_sha']
    gold={(r['district'],r['site_id']):set(r['gold']) for r in read(ROOT/'outputs/shenzhen_aligned_model_20261111/scored_cases.json') if r['has_map_gold']}
    counts={};replayed=0
    for d,p in public.items():
        rows={r['site_id']:r for r in p['rows']};selected={r['site_id'] for r in support[d]};unknown=set(rows)-selected
        assert len(selected)==256
        assert not {r['group'] for r in support[d] if r['validation']}&{r['group'] for r in support[d] if not r['validation']}
        for r in support[d]:assert set(r['schools'])==gold[d,r['site_id']]
        predictions=read(OUT/'predictions'/f'{d}.json')
        for arm,pred in predictions.items():
            assert {r['site_id'] for r in pred}==unknown and len(pred)==len(unknown)
            counts[arm]=counts.get(arm,0)+sum(set(r['after'])==gold[d,r['site_id']] for r in pred)
        nearest={r['site_id']:set(r['after']) for r in predictions['nearest_raw']}
        for r in read(OUT/'nearest_provenance'/f'{d}.json'):
            q=rows[r['query']];ref=rows[r['reference']]
            assert r['reference'] in selected and q['group']!=ref['group']
            legal=[s for s in support[d] if s['group']!=q['group']]
            best=min(legal,key=lambda s:np.linalg.norm(np.array(q['xy'])-s['xy']))
            assert best['site_id']==r['reference']
            assert nearest[q['site_id']]==set(best['schools']);replayed+=1
    results=read(OUT/'results.json')
    for arm,n in counts.items():assert n==results['metrics'][arm]['all']['correct'],arm
    assert replayed==1249
    audit=dict(passed=True,independently_scored=counts,nearest_predictions_replayed=replayed,
               selected_label_points=512,query_points=1249,group_exclusion=True,scope='Independent counting and nearest replay, not independent geography validation')
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
