"""Independent root/union replay and original-population metric recomputation."""
import json
import pickle
import numpy as np
from sympy import Symbol, And, Or, Not
from sympy.logic.inference import satisfiable
import experiment_shenzhen_partial_scope as E
from zoning.evaluate import bootstrap

F=E.F;R=E.R


def main():
    dest=R.ROOT/'round28_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    R.check(E.OUT,'prediction_freeze.json')
    for p,h in F.read(E.OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    cfg=F.read(E.OUT/'protocol.json');report=F.read(E.OUT/'results.json')
    for d,h in report['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={};graphs={};proofs=[]
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
        g=F.read(E.OUT/'graphs'/f'{d}.json');graphs[d]=g
        old=F.read(E.D.G.OUT/f'{d}.json')
        assert g['edges']==old['edges']+g['added']
        blocked=set(g['blocked_school_ids'])
        for edge in g['added']:
            assert edge['trigger_school_id'] not in blocked and edge['implied_school_id'] not in blocked
            symbols={}
            def conv(x):
                if x[0]=='and':return And(*(conv(c) for c in x[1:]))
                if x[0]=='or':return Or(*(conv(c) for c in x[1:]))
                if x[0]=='minus':return And(conv(x[1]),Not(conv(x[2])))
                k=tuple(x)
                if k not in symbols:symbols[k]=Symbol('s'+str(len(symbols)))
                return symbols[k]
            a,b=conv(edge['trigger_expression']),conv(edge['implied_expression'])
            assert satisfiable(a) is not False and satisfiable(And(a,Not(b))) is False
            proofs.append(dict(district=d,trigger=edge['trigger_school_name'],implied=edge['implied_school_name'],atoms=len(symbols)))
    checks={};files=0;changed=[]
    for arm in cfg['arms']:
        for obj in cfg['objectives']:
            for label in cfg['labels']:
                scored=[];unscored=[];keys=[]
                for d in R.TARGETS:
                    prior=F.read(E.PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    after=F.read(E.OUT/'predictions'/arm/obj/label/f'{d}.json')
                    roots=F.read(E.D.root_path(arm,obj,label,d));files+=1
                    assert [r['site_id'] for r in prior]==[r['site_id'] for r in after]==[r['site_id'] for r in roots]
                    for p,a,q in zip(prior,after,roots):
                        key=(d,a['site_id']);keys.append(key)
                        seed=set(q['decision']['raw_school_ids'])|set(q['decision']['positive_lower_bound'])
                        addition={e['implied_school_id'] for e in graphs[d]['added'] if e['trigger_school_id'] in seed}
                        assert a['before']==sorted(p['after']) and a['after']==sorted(set(p['after'])|addition)
                        assert a['added_school_ids']==sorted(addition-set(p['after']))
                        assert a['original_trigger_school_ids']==sorted(seed)
                        if a['added_school_ids']:changed.append(dict(arm=arm,objective=obj,label=label,**a))
                        g=gold[key]
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
    F.write(dest,dict(results=checks,proofs=proofs,prediction_files=files,metric_groups=220,changed_output_rows=len(changed),
        original_roots_verified=True,original_gold_unchanged=True,no_transitive_additions=True,
        independent_validation=False,adopted=False,audit_code_sha256=F.sha(__file__)))
    print('ALL88FILES_220METRICGROUPS_VERIFIED',len(changed),flush=True)


if __name__=='__main__':main()
