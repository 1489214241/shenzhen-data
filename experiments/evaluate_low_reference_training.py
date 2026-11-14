"""Standard-library verification against public district Gold; does not train."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/modern_low_reference_training_20261114'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def main():
    release=read(RUN/'k1_release.json');results=read(RUN/'results.json')
    public=read(ROOT/'experiments/modern_school_gap_reference_20261114/public_population.json')
    gold={}
    for d in release:
        for r in read(ROOT/d/'Gold/小区_学校Gold.json'):
            gold[d,r['coordinate_group_id']]=set(r['point_school_ids'])
    keys=set();cost=0;combos={}
    for d,p in release.items():
        cost+=len(p['support']);assert len(p['support'])==len({r['group'] for r in p['support']})
        combos[d]={tuple(sorted(r['schools'])) for r in p['support']}
        for r in p['support']:assert set(r['schools'])==gold[d,r['site_id']]
        common=set(p['common_ids']);rows=[r for r in p['query'] if r['site_id'] in common]
        assert {r['site_id'] for r in rows}==common
        excluded=set(public[d]['base_reference_groups'])|{r['group'] for r in p['support']}
        assert not excluded&{r['group'] for r in rows}
        keys.update((d,r['site_id']) for r in rows)
    assert cost==84 and len(keys)==1095
    correct={};output={}
    for arm,metrics in results['metrics'].items():
        rows=read(RUN/'scored_rows'/f'{arm}.json');by={(r['district'],r['site_id']):r for r in rows}
        assert len(by)==len(rows)==1095 and set(by)==keys
        filters={'all':lambda r:True,'single':lambda r:len(gold[r['district'],r['site_id']])==1,
            'multi':lambda r:len(gold[r['district'],r['site_id']])>1,
            'seen':lambda r:tuple(sorted(gold[r['district'],r['site_id']])) in combos[r['district']],
            'unseen':lambda r:tuple(sorted(gold[r['district'],r['site_id']])) not in combos[r['district']]}
        filters.update({d:(lambda r,d=d:r['district']==d) for d in release})
        for key,r in by.items():assert set(r['truth'])==gold[key]
        for group,f in filters.items():
            rr=[r for r in rows if f(r)];count=fp=fn=0
            for r in rr:
                truth=gold[r['district'],r['site_id']];pred=set(r['prediction'])
                count+=pred==truth;fp+=len(pred-truth);fn+=len(truth-pred)
            m=metrics[group]
            assert (len(rr),count,fp,fn)==tuple(m[k] for k in ['n','correct','fp','fn'])
        correct[arm]={key:set(r['prediction'])==gold[key] for key,r in by.items()}
        output[arm]=dict(n=1095,correct=metrics['all']['correct'],exact=100*metrics['all']['correct']/1095)
    for pair,v in results['comparisons'].items():
        a,b=pair.split(' minus ');fixed=sum(correct[a][k] and not correct[b][k] for k in keys)
        harmed=sum(correct[b][k] and not correct[a][k] for k in keys)
        assert (fixed,harmed)==(v['corrected'],v['harmed'])
        assert abs(100*(fixed-harmed)/1095-v['delta_pp'])<1e-10
    result=dict(passed=True,scope='Gold-based scoring, subgroup counts, paired corrected/harmed and project exclusion; not neural retraining or independent geographic validation',
        existing_k1_verifications=cost,additional_verifications_this_trial=0,results=output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
