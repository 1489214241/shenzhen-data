"""Independent full-output check for the community/road conjunction trial."""
from collections import defaultdict
import pickle
import numpy as np
import experiment_shenzhen_joint_geography as D
from zoning.evaluate import bootstrap

F=D.F;R=D.R


def main():
    dest=R.ROOT/'round25_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    R.check(D.OUT,'prediction_freeze.json')
    for p,h in F.read(D.OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    c=F.read(D.OUT/'protocol.json');report=F.read(D.OUT/'results.json')
    for d,h in report['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};facts={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
        facts[d]=defaultdict(set)
        original=F.read(D.SOURCE/d/'estimates.json')['facts']
        assert original==F.read(D.OUT/'facts'/f'{d}.json')['facts']
        assert F.sha(D.OUT/'source_replay'/d/'scope_audit.json')==F.sha(D.SOURCE/d/'scope_audit.json')
        for f in original:
            assert f['is_geographic_estimate'] and not f['verified_enrollment_eligibility']
            facts[d][f['site_id']].add(f['school_id'])
    checks={};files=0;change_ledger=[]
    for arm in c['arms']:
        for obj in c['objectives']:
            for label in c['labels']:
                scored=[];unscored=[];keys=[]
                for d in R.TARGETS:
                    before=F.read(D.PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    after=F.read(D.OUT/'predictions'/arm/obj/label/f'{d}.json');files+=1
                    assert [r['site_id'] for r in before]==[r['site_id'] for r in after]
                    for b,a in zip(before,after):
                        key=(d,a['site_id']);keys.append(key)
                        assert set(a['before'])==set(b['after'])
                        assert set(a['after'])==set(b['after'])|facts[d][a['site_id']]
                        assert len(a['after'])==len(set(a['after']))
                        assert set(a['added_school_ids'])==set(a['after'])-set(a['before'])
                        g=gold[key]
                        if a['added_school_ids']:
                            change_ledger.append(dict(arm=arm,objective=obj,label=label,district=d,site_id=a['site_id'],
                                added=a['added_school_ids'],has_map_gold=g['has_map_gold'],
                                added_in_gold=sorted(set(a['added_school_ids'])&set(g['gold'])),
                                remaining_missing=sorted(set(g['gold'])-set(a['after'])),
                                remaining_extra=sorted(set(a['after'])-set(g['gold']))))
                        if not g['has_map_gold']:unscored.append(a);continue
                        scored.append(dict(a,gold=set(g['gold']),d=d,block=str(tuple(int(v//1000) for v in xy[key]))))
                assert len(keys)==len(set(keys))==1802 and len(scored)==1761 and len(unscored)==41
                groups={}
                for name,rows in [('all',scored)]+[(d,[r for r in scored if r['d']==d]) for d in R.TARGETS]+[
                    ('single',[r for r in scored if len(r['gold'])==1]),('multi',[r for r in scored if len(r['gold'])>1])]:
                    summary={}
                    for field in ['before','after']:
                        pairs=[(set(r[field]),r['gold']) for r in rows]
                        stats=dict(n=len(rows),correct=sum(a==b for a,b in pairs),fp=sum(len(a-b) for a,b in pairs),fn=sum(len(b-a) for a,b in pairs))
                        stats['exact_accuracy']=stats['correct']/stats['n'];summary[field]=stats
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rows]
                    summary.update(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),
                        delta_95_pp=bootstrap(rows,delta,seed=20261104,draws=3000))
                    assert summary==report['reports'][arm][obj][label][name];groups[name]=summary
                checks[f'{arm}/{obj}/{label}']=groups
    assert files==88 and len(change_ledger)==44
    F.write(D.OUT/'diagnostics_not_training/changed_relation_audit.json',dict(rows=change_ledger,not_for_training=True))
    F.write(dest,dict(results=checks,prediction_files=files,metric_groups=220,source_relations=1,
        source_scope_ledger_fully_replayed=True,prediction_set_unions_verified=True,original_gold_unchanged=True,
        independent_validation=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('ALL88PREDICTION_FILES_AND220METRIC_GROUPS_VERIFIED',flush=True)


if __name__=='__main__':main()
