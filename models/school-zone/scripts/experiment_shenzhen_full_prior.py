"""All Shenzhen active districts/sites/schools: frozen Modern policy-set test.

Full population, never select queries by Gold cardinality. Preserve prior runs.
"""
import argparse
import collections
from pathlib import Path
import numpy as np
import experiment_shared_school_prior as P
from zoning.full_policy_prior import full_prior,candidate_pool,compile_pool,choose_compiled
from zoning.conditional_school_sets import catalogue
from zoning.conditional_scope_cells import build_cells
from shenzhen_scope_implication import build as build_graph

read,write,sha=P.read,P.write,P.sha
M=P.M;REPO=P.REPO;INPUT=P.INPUT;BASE=P.OUT;H=P.H
OUT=REPO/'outputs/modern_shenzhen_full_prior_20261114'
DOC=REPO.parent/'科研推进/深圳全量多校提前输入_20261114'
ARMS={a:[] for a in ['full_text_minimum','full_scope_minimum','full_prior_single_condition','full_prior_composed']}


def verify():
    for p,h in read(OUT/'inputs_sha256.json').items():assert sha(p)==h,p
    return read(OUT/'protocol.json')


def check():
    for p,h in read(OUT/'prediction_freeze.json').items():assert sha(p)==h,p


def prepare():
    M.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen run')
    old=P.verify();OUT.mkdir(parents=True)
    config=dict(date='2026-11-14',targets=old['targets'],seeds=old['seeds'],sources=old['sources'],
        arms=list(ARMS),primary='full_prior_composed minus baseline_current',training_steps=0,
        population='Every1802coordinate group and every202school across Longhua/Longgang; no target-Gold subset selection; Futian/Nanshan remain historical.',
        inputs='Same frozen Modern/native/admission scores and full202official scope inventory. New encoding uses direct scope graph plus sufficient conditional cells and independently sourced admin identities.',
        hypotheses='Every single-root scope and shared/conditional group. Composition considers union of any two independently documented conditions; applicability is estimated, never transitive implication. Also retain original output, model-only output and whole-scope-positive hypothesis.',
        constraints='Text-proved positive lower bounds required; explicit exclusions retained. No complete-set assumption. University membership never expands; co-members allowed only by independent shared/base scopes.',
        exact_selection='Maximize sum of source-threshold-centered membership log odds over final set. Frozen routed_BCE source threshold. No new parameter selection, fitting, or Shenzhen calibration. Ties fewer schools then sorted IDs.',
        ablation='full_prior_single_condition has identical objective and inputs without pairwise composition. Text/scope-minimum arms retain prior baseline and add respective evidence.',
        weak_evidence='Whole-scope states from geocoded points/roads are estimates, not administrative membership facts. Never add unknown or negative state as positive.',
        complete_syntax=False,new_external_documents=False,prior_target_results_seen=True,
        target_gold_read=False,target_polygon_read=False,adopted=False,independent_validation=False,
        limits='Frozen encoder; two-condition composition not arbitrary high-order exhaustive search; unsupported scope syntax and unknown applicability remain unresolved. No claim to know true school count for every site.')
    write(OUT/'protocol.json',config)
    links=dict(read(BASE/'inputs_sha256.json'))
    # Parent freeze is verified without reading its scored diagnostics.
    parent_freeze=read(BASE/'prediction_freeze.json')
    for p,h in parent_freeze.items():assert sha(p)==h,p
    links.update(parent_freeze)
    links[str(BASE/'prediction_freeze.json')]=sha(BASE/'prediction_freeze.json')
    for p in [OUT/'protocol.json',Path(__file__),REPO/'zoning/full_policy_prior.py',
              REPO/'tests/test_full_policy_prior.py',REPO/'zoning/conditional_scope_cells.py',
              REPO/'scripts/shenzhen_scope_implication.py',H.OUT/'source_calibration.json']:
        links[str(p)]=sha(p)
    snapshot=read(INPUT/'manifest.json');inventory={}
    for d in config['targets']:
        for f in ['full_scope_inventory.json','whole_scope.npz','base_branches.json']:
            p=INPUT/d/f;assert sha(p)==snapshot['files'][str(p.relative_to(INPUT))];links[str(p)]=sha(p)
        obs=read(INPUT/d/'full_scope_inventory.json');policy=read(INPUT/d/'reviewed_policy.json')
        graph=build_graph(obs,policy);cells,audit=build_cells(obs,policy)
        records=catalogue(policy,[graph])+cells
        write(OUT/'inventory'/f'{d}.json',dict(records=records,graph=graph,cells=cells,branch_audit=audit))
        inventory[d]=dict(schools=len(policy['school_ids']),scope_rows=len(obs),direct_edges=len(graph['edges']),
            conditional_multi_cells=len(cells),candidates=len(records),
            unsupported_scope_schools=sum(not r['admitted_branches'] for r in audit))
        links[str(OUT/'inventory'/f'{d}.json')]=sha(OUT/'inventory'/f'{d}.json')
    write(OUT/'inventory_summary.json',inventory)
    links[str(OUT/'inventory_summary.json')]=sha(OUT/'inventory_summary.json')
    write(OUT/'inputs_sha256.json',links)
    M.log('FULL PRIOR PROTOCOL SEALED',inventory)


def infer():
    M.guard();c=verify()
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve sealed predictions')
    payload=[dict(np.load(H.OUT/'probabilities/routed_bce'/f'{s}.npz')) for s in c['seeds']]
    means={d:np.stack([p[d] for p in payload]).mean(0) for d in c['targets']}
    threshold=read(H.OUT/'source_calibration.json')['routed_bce']['threshold']
    labels=['ensemble']+list(map(str,c['seeds']))
    allprob=[means]+payload;coverage={}
    for d in c['targets']:
        meta=read(INPUT/d/'metadata.json');policy=read(INPUT/d/'reviewed_policy.json')
        cons=read(INPUT/d/'constraints.json');inv=read(OUT/'inventory'/f'{d}.json')
        ledger=collections.defaultdict(list)
        for r in read(INPUT/d/'evidence_ledger.json'):ledger[r['site_id']].append(r)
        states=np.load(INPUT/d/'whole_scope.npz')['states']
        ids=[s['school_id'] for s in meta['schools']];assert states.shape==(len(meta['sites']),len(ids))
        universities=[r['school_ids'] for r in policy['relations'] if r['kind']=='university']
        shared=[r['school_ids'] for r in inv['records'] if len(r['school_ids'])>1]
        baselines={label:read(BASE/'predictions/baseline_current'/label/f'{d}.json') for label in labels}
        raws={label:read(H.OUT/'predictions/routed_bce_raw'/label/f'{d}.json') for label in labels}
        predictions={arm:{label:[] for label in labels} for arm in ARMS}
        priors=[];empty=collections.Counter();pool_sizes=[]
        # Reuse pure-source candidate unions; no scoring labels or query identity.
        static={compose:candidate_pool(inv['records'],{'verified_members':[]},[],[],[],[],compose)
                for compose in [False,True]}
        cache={}
        for q,site in enumerate(meta['sites']):
            sid=site['coordinate_group_id'];con=cons[sid]
            scope=[ids[j] for j in np.flatnonzero(states[q]==1)]
            prior=full_prior(site,policy,[inv['graph']],inv['cells'],con['positive'],con['negative'],ledger[sid],scope)
            priors.append(dict(prior,site_id=sid,building_name=site['building_name']))
            cachekey=(tuple(prior['verified_members']),tuple(con['negative']))
            if cachekey not in cache:
                compiled={}
                for compose in [False,True]:
                    lower=set(prior['verified_members']);excluded=set(con['negative'])
                    merged={frozenset(set(s)|lower):basis for s,basis in static[compose] if not (set(s)|lower)&excluded}
                    pool=sorted(merged.items(),key=lambda kv:(len(kv[0]),sorted(kv[0])))
                    compiled[compose]=compile_pool(pool,ids,universities,shared)
                if len(cache)>12:cache.clear()  # Bound memory; same empty-lower pool is cheap to retain.
                cache[cachekey]=compiled
            compiled=cache[cachekey]
            for label,pp in zip(labels,allprob):
                old=baselines[label][q];raw=raws[label][q]
                assert old['site_id']==raw['site_id']==sid
                base=dict(district=d,site_id=sid,building_name=site['building_name'],complete_school_set=False,enrollment_eligibility=False)
                for arm,key in [('full_text_minimum','verified_members'),('full_scope_minimum','estimated_members')]:
                    after=sorted((set(old['after'])|set(prior[key]))-set(con['negative']))
                    predictions[arm][label].append(dict(base,after=after,decision=dict(basis=key)))
                extra=candidate_pool([],prior,old['after'],raw['after'],prior['estimated_members'],con['negative'],False)
                extras=compile_pool(extra,ids,universities,shared)
                for arm,compose in [('full_prior_single_condition',False),('full_prior_composed',True)]:
                    # Compare source-template winner and per-query hypotheses under identical score.
                    options=[choose_compiled(cp,pp[d][q],threshold) for cp in [compiled[compose],extras]]
                    legal=[v for v in options if v[0]]
                    if legal:
                        after,decision=min(legal,key=lambda v:(-v[1]['score'],len(v[0]),v[0]))
                    else:
                        after=old['after'];decision=dict(status='fallback_conflicting_evidence',candidate_count=0)
                        empty[arm]+=1
                    predictions[arm][label].append(dict(base,after=after,decision=decision))
            pool_sizes.append(len(compiled[True][0]))
            if q%150==0:M.log('FULL INFER',d,q,'/',len(meta['sites']),'candidates',pool_sizes[-1])
        for arm,values in predictions.items():
            for label,rows in values.items():write(OUT/'predictions'/arm/label/f'{d}.json',rows)
        for label,rows in baselines.items():write(OUT/'predictions/baseline_current'/label/f'{d}.json',rows)
        write(OUT/'predictions/baseline_native/ensemble'/f'{d}.json',read(BASE/'predictions/baseline_native/ensemble'/f'{d}.json'))
        write(OUT/'priors'/f'{d}.json',priors)
        write(OUT/'rankings'/f'{d}.json',read(BASE/'rankings'/f'{d}.json'))
        coverage[d]=dict(sites=len(priors),verified_multi=sum(len(p['verified_members'])>1 for p in priors),
            scope_estimated_multi=sum(len(p['estimated_members'])>1 for p in priors),
            matched_scope_cell_sites=sum(bool(p['matched_cells']) for p in priors),
            mean_composed_candidates=float(np.mean(pool_sizes)),fallbacks=dict(empty))
    write(OUT/'input_audit.json',dict(passed=True,coverage=coverage,target_gold_read=False,source_threshold=threshold))
    files=[p for folder in ['priors','predictions','rankings'] for p in (OUT/folder).rglob('*.json')]+[OUT/'input_audit.json']
    write(OUT/'prediction_freeze.json',{str(p):sha(p) for p in files})
    M.log('FULL PRIOR PREDICTIONS SEALED',coverage)


def score():
    # Same evaluator, denominator, exact-set metric and spatial bootstrap as prior run.
    verify();check();P.OUT=OUT;P.ARMS=ARMS;P.score()


def audit():
    c=verify();check();r=read(OUT/'results.json');count=0
    for d,h in r['gold_sha256'].items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    baseline={(r['d'],r['site_id']):r for r in read(OUT/'scored_rows/baseline_current.json')}
    for arm,metrics in r['metrics'].items():
        rows=read(OUT/'scored_rows'/f'{arm}.json');assert len(rows)==1761
        assert sum(set(x['prediction'])==set(x['truth']) for x in rows)==metrics['all']['correct']
        for row in rows:
            p=set(row['prediction']);t=set(row['truth'])
            assert row['correct']==(p==t) and row['fp']==len(p-t) and row['fn']==len(t-p)
            count+=1
        if arm in ARMS:
            delta=[int(x['correct'])-int(baseline[x['d'],x['site_id']]['correct']) for x in rows]
            assert r['comparisons'][arm]['all']['corrected']==delta.count(1)
            assert r['comparisons'][arm]['all']['harmed']==delta.count(-1)
    for d in c['targets']:
        priors=read(OUT/'priors'/f'{d}.json');cons=read(INPUT/d/'constraints.json')
        for label in ['ensemble']+list(map(str,c['seeds'])):
            base=read(OUT/'predictions/baseline_current'/label/f'{d}.json')
            assert base==read(BASE/'predictions/baseline_current'/label/f'{d}.json')
            for arm in ARMS:
                rows=read(OUT/'predictions'/arm/label/f'{d}.json')
                assert len(rows)==len(base)
                for old,new,prior in zip(base,rows,priors):
                    assert old['site_id']==new['site_id']==prior['site_id']
                    p=set(new['after']);negative=set(cons[new['site_id']]['negative'])
                    assert not p&negative
                    if new['decision'].get('status')!='fallback_conflicting_evidence':
                        assert set(prior['verified_members'])<=p
                    if arm.endswith('minimum'):
                        key='verified_members' if arm=='full_text_minimum' else 'estimated_members'
                        assert p==(set(old['after'])|set(prior[key]))-negative
                    count+=1
    write(OUT/'output_audit.json',dict(passed=True,checked_rows=count,gold_unchanged=True,
        baseline_identical=True,full_population=True,prior_inclusion_and_exclusions_checked=True,
        note='Vectorized candidate selector separately tested against scalar implementation; no independent full reimplementation.'))
    M.log('FULL PRIOR AUDIT PASSED',count)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','infer','score','audit'])
    globals()[ap.parse_args().phase]()
