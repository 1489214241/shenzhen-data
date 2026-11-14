"""Independent set-scoring audit; no neural fitting or model selection.
Run with --repo pointing to an unpacked 87fc8c0 snapshot.
"""
import argparse
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run=a.repo/'experiments/modern_low_reference_training_20261114'
    released=read(run/'k1_release.json')
    known={};query=set()
    for d,p in released.items():
        for r in read(a.repo/d/'Gold/小区_学校Gold.json'):
            known[(d,r['coordinate_group_id'])]=frozenset(r['point_school_ids'])
        query|={(d,s) for s in p['common_ids']}
    assert len(query)==1095
    ordered=sorted(query);arms={};summary={}
    for path in sorted((run/'scored_rows').glob('*.json')):
        records=read(path);rows={(r['district'],r['site_id']):r for r in records}
        assert set(rows)==query and len(rows)==len(records)
        ok={};fp=fn=0
        for k in ordered:
            p=frozenset(rows[k]['prediction']);g=known[k]
            assert frozenset(rows[k]['truth'])==g
            ok[k]=(p==g);fp+=len(p-g);fn+=len(g-p)
        arms[path.stem]=ok
        summary[path.stem]={'n':len(query),'correct':sum(ok.values()),'fp':fp,'fn':fn,
            'exact_percent':100*sum(ok.values())/len(query)}
    pairs={}
    for aa,bb in [('global_sparse_policy','global_full_policy'),('local_sparse_policy','local_full_policy'),
        ('local_full_policy','global_full_policy'),('local_sparse_policy','global_sparse_policy')]:
        corr=sum(arms[aa][k] and not arms[bb][k] for k in ordered)
        harm=sum(arms[bb][k] and not arms[aa][k] for k in ordered)
        pairs[aa+' minus '+bb]={'corrected':corr,'harmed':harm,'delta_pp':100*(corr-harm)/len(query)}
    archived=read(run/'results.json')
    for name,s in summary.items():
        for key in ('n','correct','fp','fn'):
            assert s[key]==archived['metrics'][name]['all'][key],(name,key)
    for name,s in pairs.items():
        for key in ('corrected','harmed','delta_pp'):
            assert abs(s[key]-archived['comparisons'][name][key])<1e-9,(name,key)
    result={'passed':True,'scope':'Independent public-Gold set rescoring; NOT retraining',
        'source_commit':'87fc8c0a6fcf6c1350e68b88b896c138634ee43b',
        'existing_simulated_verifications':sum(len(p['support']) for p in released.values()),
        'new_verifications':0,'metrics':summary,'comparisons':pairs}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
