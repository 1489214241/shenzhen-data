"""Post-freeze initialization sensitivity; never selects a seed on target Gold."""
import numpy as np
import experiment_low_reference_training as R


def main():
    c=R.verify()
    for rel,h in R.read(R.OUT/'prediction_freeze.json').items():assert R.sha(R.OUT/rel)==h
    gold={(r['district'],r['site_id']):set(r['gold']) for r in R.read(R.E.GOLD) if r['has_map_gold']}
    thresholds=R.read(R.OUT/'thresholds.json');out={}
    for d in c['targets']:
        data=R.E.pkl(R.OUT/'target_inputs'/f'{d}.pkl');q=data['query_ids']
        for arm in c['arms']:
            pp=np.load(R.OUT/'target_probabilities'/f'{d}_{arm}.npz')['seeds']
            for seed,p in zip(c['seeds'],pp):
                sets=R.decoded_sets(p,thresholds[arm]['threshold'])
                rows=R.E.decode(d,p,sets,data,q)
                correct=sum(set(r['after'])==gold[d,r['site_id']] for r in rows)
                out.setdefault(arm,{}).setdefault(str(seed),{})[d]=dict(n=len(q),correct=correct,exact=100*correct/len(q))
    for arm,v in out.items():
        for seed,ds in v.items():
            n=sum(m['n'] for m in ds.values());correct=sum(m['correct'] for m in ds.values())
            ds['all']=dict(n=n,correct=correct,exact=100*correct/n)
    R.write(R.OUT/'seed_metrics.json',dict(threshold='Same source-ensemble threshold for every seed; seeds never selected using target results',metrics=out))
    print('SEED METRICS', {a:[round(v['all']['exact'],2) for v in seeds.values()] for a,seeds in out.items()})


if __name__=='__main__':main()
