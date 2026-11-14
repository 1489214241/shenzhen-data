"""Re-score published predictions against the two district Gold files."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/shenzhen_membership_trees_20261114'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    gold={}
    for d in ['龙华区','龙岗区']:
        for r in read(ROOT/d/'Gold/小区_学校Gold.json'):
            key=(d,r['coordinate_group_id']);assert key not in gold
            gold[key]=set(r['point_school_ids'])
    assert len(gold)==1761
    results=read(RUN/'results.json');summary={};base=None
    for arm,expected in results['metrics'].items():
        rows=read(RUN/'scored_rows'/f'{arm}.json')
        by={(r['d'],r['site_id']):r for r in rows}
        assert len(rows)==len(by)==1761 and set(by)==set(gold)
        correctness={}
        for key,r in by.items():
            assert set(r['truth'])==gold[key],('Stored truth disagrees with district Gold',key)
            correctness[key]=set(r['prediction'])==gold[key]
        correct=sum(correctness.values());assert correct==expected['all']['correct']
        fp=sum(len(set(r['prediction'])-gold[k]) for k,r in by.items())
        fn=sum(len(gold[k]-set(r['prediction'])) for k,r in by.items())
        assert (fp,fn)==(expected['all']['fp'],expected['all']['fn'])
        item=dict(n=1761,correct=correct,exact_percent=100*correct/1761)
        for d in ['龙华区','龙岗区']:
            count=sum(v for k,v in correctness.items() if k[0]==d)
            assert count==expected[d]['correct'];item[d]=count
        if arm=='baseline_current':base=correctness
        else:
            changes=[int(v)-int(base[k]) for k,v in correctness.items()]
            cmp=results['comparisons'][arm]['all']
            assert changes.count(1)==cmp['corrected'] and changes.count(-1)==cmp['harmed']
        summary[arm]=item
    print(json.dumps(dict(passed=True,scope='Frozen prediction scoring, not retraining',results=summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
