"""Post-primary, no-tuning zero-new-verification control on common queries."""
import argparse
import numpy as np
import experiment_school_gap_reference as G
E=G.E;OUT=G.OUT/'zero_budget_control'
def infer():
    E.guard();G.verify();plans=G.read(G.OUT/'released_plans.json');public=G.read(G.OUT/'public_population.json');partial=G.read(G.OUT/'verified_partial_references.json')
    G.write(OUT/'protocol.json',dict(post_primary_diagnostic=True,new_labels=0,new_training=0,new_selection=False,
        same_common_queries=True,baseline='Same pretrained Modern ensemble and all existing partial references; no newly verified full sets.'))
    for d in E.DISTRICTS:
        common=set(plans[d]['3']['common_ids']);rows=[r for r in public[d]['rows'] if r['site_id'] in common];q=np.array([r['q'] for r in rows],int)
        data=G.build(d,[],partial[d],public[d]['all_groups']);pp=[E.infer(E.model(seed),data,q)[0] for seed in E.SEEDS]
        p=np.mean(pp,0);ss=[(int(i),) for i in p.argmax(1)]
        G.write(OUT/'predictions'/f'{d}.json',E.decode(d,p,ss,data,q))
    G.write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):G.sha(p) for p in (OUT/'predictions').glob('*.json')})
def score():
    for p,h in G.read(OUT/'prediction_freeze.json').items():assert G.sha(OUT/p)==h
    gold={(r['district'],r['site_id']):r for r in G.read(E.GOLD) if r['has_map_gold']};rows=[]
    for d in E.DISTRICTS:
        for r in G.read(OUT/'predictions'/f'{d}.json'):
            g=gold[d,r['site_id']];truth=set(g['gold']);pred=set(r['after'])
            rows.append(dict(district=d,site_id=r['site_id'],name=r['building_name'],truth=sorted(truth),prediction=sorted(pred),correct=truth==pred,tp=len(truth&pred),fp=len(pred-truth),fn=len(truth-pred)))
    G.write(OUT/'scored_rows.json',rows)
    result={d:E.metric([r for r in rows if d=='all' or r['district']==d]) for d in ['all']+E.DISTRICTS}
    G.write(OUT/'results.json',result);print(result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['infer','score']);args=p.parse_args();globals()[args.phase]()
