"""Check every original root, direct graph addition and fixed-population metric."""
import json
import pickle
import numpy as np
import experiment_shenzhen_scope_graph_refresh as D
from zoning.evaluate import bootstrap

F=D.F;R=D.R


def main():
    dest=R.ROOT/'round27_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    R.check(D.OUT,'prediction_freeze.json')
    for p,h in F.read(D.OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    c=F.read(D.OUT/'protocol.json');report=F.read(D.OUT/'results.json')
    for d,h in report['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};old={};new={}
    graph_checks={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as h:dist=pickle.load(h)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
        old[d]=F.read(D.G.OLD/f'{d}.json');new[d]=F.read(D.OUT/'graphs'/f'{d}.json')
        assert new[d]==F.read(D.G.OUT/f'{d}.json')
        ledger=F.read(D.G.OUT/f'{d}_changes.json')
        before={(e['trigger_school_id'],e['implied_school_id']) for e in old[d]['edges']}
        after={(e['trigger_school_id'],e['implied_school_id']) for e in new[d]['edges']}
        assert before<=after and after-before=={(e['trigger_school_id'],e['implied_school_id']) for e in ledger['added']}
        for edge in ledger['added']:
            proof=D.G.exhaustive_implication(edge['trigger_expression'],edge['implied_expression'])
            assert json.loads(json.dumps(proof))==edge['independent_proof']
        graph_checks[d]=dict(old_edges=len(before),new_edges=len(after),added=len(after-before),no_removed_edges=True)
    checks={};files=0;changed=[]
    for arm in c['arms']:
        for obj in c['objectives']:
            for label in c['labels']:
                scored=[];unscored=[];keys=[]
                for d in R.TARGETS:
                    before=F.read(D.PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    after=F.read(D.OUT/'predictions'/arm/obj/label/f'{d}.json')
                    origin=F.read(D.root_path(arm,obj,label,d));files+=1
                    assert [r['site_id'] for r in before]==[r['site_id'] for r in after]==[r['site_id'] for r in origin]
                    for b,a,o in zip(before,after,origin):
                        key=(d,a['site_id']);keys.append(key)
                        root=set(o['decision']['raw_school_ids'])|set(o['decision']['positive_lower_bound'])
                        # Independent comprehension, not the production expansion helper.
                        old_implied={e['implied_school_id'] for e in old[d]['edges'] if e['trigger_school_id'] in root}
                        new_implied={e['implied_school_id'] for e in new[d]['edges'] if e['trigger_school_id'] in root}
                        assert set(a['original_trigger_school_ids'])==root
                        assert set(a['new_direct_graph_relations'])==(root|new_implied)-(root|old_implied)
                        expected=set(b['after'])|root|new_implied
                        assert set(a['before'])==set(b['after']) and set(a['after'])==expected and len(a['after'])==len(expected)
                        assert set(a['added_school_ids'])==expected-set(b['after'])
                        g=gold[key]
                        if a['added_school_ids']:
                            changed.append(dict(arm=arm,objective=obj,label=label,district=d,site_id=a['site_id'],building_name=a['building_name'],
                                added=a['added_school_ids'],has_map_gold=g['has_map_gold'],before=a['before'],after=a['after'],gold=g['gold']))
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
    assert files==88
    F.write(D.OUT/'diagnostics_not_training/changed_relation_audit.json',dict(rows=changed,not_for_training=True))
    F.write(dest,dict(results=checks,prediction_files=files,metric_groups=220,changed_output_rows=len(changed),graph_checks=graph_checks,
        all_original_root_triggers_verified=True,no_transitive_additions=True,original_gold_unchanged=True,
        independent_validation=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('ALL88PREDICTION_FILES_AND220METRIC_GROUPS_VERIFIED',len(changed),'changed output rows',flush=True)


if __name__=='__main__':main()
