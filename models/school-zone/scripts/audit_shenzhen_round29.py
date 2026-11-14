"""Audit every named-region witness, source fact, output union and metric."""
from collections import defaultdict
import pickle
import numpy as np
import experiment_shenzhen_named_region as E
from shenzhen_scope_implication import canonical,implies
from shenzhen_partial_scope import partial,sat_proof
from zoning.evaluate import bootstrap

F=E.F;R=E.R


def main():
    dest=R.ROOT/'round29_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    R.check(E.OUT,'prediction_freeze.json')
    for p,h in F.read(E.OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    for rel,h in F.read(E.G.OUT/'manifest.json')['files'].items():
        assert F.sha(R.ROOT/'named_region_source_replay'/rel)==h,rel
    cfg=F.read(E.OUT/'protocol.json');report=F.read(E.OUT/'results.json')
    for d,h in report['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};facts={};source_counts={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
        source=F.read(E.OUT/'facts'/f'{d}.json');assert source==F.read(E.G.OUT/d/'facts.json')
        expressions={r['school_id']:partial(r['ast'],r['school_id']) for r in source['school_asts']}
        graph=F.read(E.G.SOURCE/f'{d}.json')
        blocked={r['school_id'] for r in graph['inventory'] if r['skip_reason']=='university_eligibility_or_pending_member'}
        expected=defaultdict(set)
        for i,w in enumerate(source['witnesses']):
            assert w['source_school_id'] not in blocked
            region=canonical(w['region_ast'])
            assert implies(region,expressions[w['source_school_id']])
            for target in w['targets']:
                sid=target['school_id'];assert sid not in blocked
                proof=sat_proof(region,expressions[sid]);assert proof==target['proof']
                for s in w['matched_sites']:expected[(s['site_id'],sid)].add(i)
        actual={(f['site_id'],f['school_id']):set(f['witness_indices']) for f in source['facts']}
        assert len(actual)==len(source['facts']) and actual==expected
        facts[d]=defaultdict(set)
        for f in source['facts']:
            assert not f['prediction_trigger_used'] and f['witness_indices']
            facts[d][f['site_id']].add(f['school_id'])
        source_counts[d]=dict(witnesses=len(source['witnesses']),relations=len(source['facts']),sites=len(facts[d]))
    checks={};files=0;changed=[]
    for arm in cfg['arms']:
        for obj in cfg['objectives']:
            for label in cfg['labels']:
                scored=[];unscored=[];keys=[]
                for d in R.TARGETS:
                    prior=F.read(E.PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    after=F.read(E.OUT/'predictions'/arm/obj/label/f'{d}.json');files+=1
                    assert [r['site_id'] for r in prior]==[r['site_id'] for r in after]
                    for p,a in zip(prior,after):
                        key=(d,a['site_id']);keys.append(key);extra=facts[d][a['site_id']]-set(p['after'])
                        assert a['before']==sorted(p['after']) and a['after']==sorted(set(p['after'])|extra)
                        assert a['added_school_ids']==sorted(extra) and not a['prediction_trigger_used']
                        g=gold[key]
                        if extra:changed.append(dict(arm=arm,objective=obj,label=label,**a,gold=g['gold'],has_map_gold=g['has_map_gold']))
                        if not g['has_map_gold']:unscored.append(a);continue
                        scored.append(dict(a,gold=set(g['gold']),d=d,block=str(tuple(int(v//1000) for v in xy[key]))))
                assert len(keys)==len(set(keys))==1802 and len(scored)==1761 and len(unscored)==41
                groups={}
                for name,rows in [('all',scored)]+[(d,[r for r in scored if r['d']==d]) for d in R.TARGETS]+[
                    ('single',[r for r in scored if len(r['gold'])==1]),('multi',[r for r in scored if len(r['gold'])>1])]:
                    summary={}
                    for field in ['before','after']:
                        pairs=[(set(r[field]),r['gold']) for r in rows]
                        s=dict(n=len(rows),correct=sum(a==b for a,b in pairs),fp=sum(len(a-b) for a,b in pairs),fn=sum(len(b-a) for a,b in pairs))
                        s['exact_accuracy']=s['correct']/s['n'];summary[field]=s
                    delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rows]
                    summary.update(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rows,delta,seed=20261104,draws=3000))
                    assert summary==report['reports'][arm][obj][label][name]
                    groups[name]=summary
                checks[f'{arm}/{obj}/{label}']=groups
    assert files==88 and len(changed)==F.read(E.OUT/'source_only_changes.json')['total_changed']
    F.write(E.OUT/'diagnostics_not_training/changed_relation_audit.json',dict(rows=changed,not_for_training=True))
    F.write(dest,dict(results=checks,source_counts=source_counts,prediction_files=files,metric_groups=220,
        changed_output_rows=len(changed),source_facts_rebuilt_bitwise=True,original_gold_unchanged=True,
        prediction_triggers_used=False,independent_validation=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('ALL88FILES_220METRICGROUPS_AND_SOURCE_WITNESSES_VERIFIED',len(changed),flush=True)


if __name__=='__main__':main()
