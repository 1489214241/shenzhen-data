"""Audit full-source replay, logical bounds, every edit and all set metrics."""
import pickle
from collections import defaultdict
import numpy as np
import experiment_shenzhen_explicit_scope as E
from zoning.evaluate import bootstrap

F,R=E.F,E.R


def bounds(ast,true_terms):
    op=ast['op']
    if op=='named' and ast['text'] in true_terms:return True,True
    if op not in ('and','or','minus'):return False,True
    v=[bounds(c,true_terms) for c in ast['children']]
    if op=='and':return all(x[0] for x in v),all(x[1] for x in v)
    if op=='or':return any(x[0] for x in v),any(x[1] for x in v)
    return v[0][0] and not v[1][1],v[0][1] and not v[1][0]


def main():
    dest=R.ROOT/'round31_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    for rel,h in F.read(E.G.OUT/'manifest.json')['files'].items():
        assert F.sha(R.ROOT/'explicit_scope_source_replay'/rel)==h,rel
    facts={};source_counts={}
    for d in R.TARGETS:
        src=F.read(E.G.OUT/d/'facts.json');byid={r['school_id']:r for r in src['school_inventory']}
        plus=defaultdict(set);minus=defaultdict(set)
        for key,wanted in [('positive_repairs',1),('negative_facts',0)]:
            for f in src[key]:
                row=byid[f['school_id']];assert row['eligible']
                terms=f['proof']['true_named_terms'];q=f['site_id']
                for term in terms:
                    receipt=src['name_receipts'][term]
                    assert q in {r['site_id'] for r in receipt['hits']} and q not in receipt['ambiguous_site_ids']
                low,high=bounds(row['after_ast'],set(terms))
                assert low if wanted else not high
                if wanted:
                    change=next(c for c in src['parser_changes'] if c['school_id']==f['school_id'])
                    assert change['after']==row['after_ast']
                    plus[q].add(f['school_id'])
                else:
                    assert not f['conflicts_with_prior_positive'];minus[q].add(f['school_id'])
        for f in src['pending_local_exclusions']:
            assert f['school_id'] not in minus[f['site_id']]
        facts[d]=(plus,minus)
        source_counts[d]=dict(schools=len(byid),positive_relations=sum(map(len,plus.values())),
                             negative_relations=sum(map(len,minus.values())),pending=len(src['pending_local_exclusions']))
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};xy={}
    for d in R.TARGETS:
        with (R.NEW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    checked=0;files=0;changes=[]
    for variant in E.VARIANTS:
        folder=E.OUT/variant;R.check(folder,'prediction_freeze.json')
        for p,h in F.read(folder/'input_links.json')['files'].items():assert F.sha(p)==h,p
        cfg=F.read(folder/'protocol.json');report=F.read(folder/'results.json')
        for d,h in report['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
        counts={}
        for arm in cfg['arms']:
            for obj in cfg['objectives']:
                for label in cfg['labels']:
                    scored=[];unscored=[];keys=[]
                    for d in R.TARGETS:
                        plus,minus=facts[d];prior=F.read(E.PARENT/'predictions'/arm/obj/label/f'{d}.json')
                        after=F.read(folder/'predictions'/arm/obj/label/f'{d}.json');files+=1
                        assert [r['site_id'] for r in prior]==[r['site_id'] for r in after]
                        for p,a in zip(prior,after):
                            key=(d,a['site_id']);keys.append(key);expected=set(p['after'])
                            if variant!='negative_only':expected|=plus[a['site_id']]
                            if variant!='positive_only':expected-=minus[a['site_id']]
                            assert expected and a['before']==sorted(p['after']) and a['after']==sorted(expected)
                            assert a['added_school_ids']==sorted(expected-set(p['after']))
                            assert a['removed_school_ids']==sorted(set(p['after'])-expected)
                            g=gold[key]
                            if a['before']!=a['after']:changes.append(dict(variant=variant,arm=arm,objective=obj,label=label,**a,gold=g['gold'],has_map_gold=g['has_map_gold']))
                            if not g['has_map_gold']:unscored.append(a);continue
                            scored.append(dict(a,gold=set(g['gold']),d=d,block=str(tuple(int(v//1000) for v in xy[key]))))
                        counts[f'{arm}/{obj}/{label}/{d}']=dict(changed=sum(r['before']!=r['after'] for r in after),
                            additions=sum(len(r['added_school_ids']) for r in after),removals=sum(len(r['removed_school_ids']) for r in after))
                    assert len(keys)==len(set(keys))==1802 and len(scored)==1761 and len(unscored)==41
                    for name,rows in [('all',scored)]+[(d,[r for r in scored if r['d']==d]) for d in R.TARGETS]+[
                            ('single',[r for r in scored if len(r['gold'])==1]),('multi',[r for r in scored if len(r['gold'])>1])]:
                        summary={}
                        for field in ['before','after']:
                            pairs=[(set(r[field]),r['gold']) for r in rows]
                            v=dict(n=len(rows),correct=sum(a==b for a,b in pairs),fp=sum(len(a-b) for a,b in pairs),fn=sum(len(b-a) for a,b in pairs))
                            v['exact_accuracy']=v['correct']/v['n'];summary[field]=v
                        delta=[int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rows]
                        summary.update(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rows,delta,seed=20261104,draws=3000))
                        assert summary==report['reports'][arm][obj][label][name];checked+=1
        assert counts==F.read(folder/'source_only_changes.json')['counts']
        print('AUDITED',variant,flush=True)
    assert files==264 and checked==660
    F.write(E.OUT/'diagnostics_not_training/changed_relation_audit.json',dict(rows=changes,not_for_training=True))
    F.write(dest,dict(source_counts=source_counts,prediction_files=files,metric_groups=checked,changed_output_rows=len(changes),
        full_source_facts_rebuilt_bitwise=True,independent_boolean_bounds_verified=True,original_gold_unchanged=True,
        adopted=False,independent_validation=False,audit_code_sha256=F.sha(__file__)))


if __name__=='__main__':main()
