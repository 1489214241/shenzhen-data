"""Frozen ModernNCA-LN -> conditional set decoder, with a separate evidence arm.

prepare/infer forbid target scoring files. score only runs after prediction seal.
This is a decoder experiment, not a claim of a trained complete-set encoder.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import experiment_modernnca_shenzhen_multilabel as M
from zoning.conditional_school_sets import catalogue,site_candidates,select,named_sufficient,exact_identity,scores
from zoning.shenzhen_policy import apply as policy_apply
from shenzhen_scope_implication import expand_original
from shenzhen_reviewed_supplements import ReviewedSupplements

OUT=REPO/'outputs/modernnca_conditional_sets_20261113'
DOC=REPO.parent/'科研推进/ModernNCA条件学校集合_20261113'
PARENT=M.OUT
ARMS=['categorical_policy','set_same_evidence','set_plus_named_evidence','categorical_plus_named_evidence']
read=M.read;write=M.write;sha=M.F.sha


def verify():
    for p,h in read(OUT/'inputs_sha256.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'code_sha256.json').items():assert sha(REPO/p)==h,p
    return read(OUT/'protocol.json')


def probabilities():
    return [dict(np.load(PARENT/'probabilities/categorical'/f'{s}.npz')) for s in M.SEEDS]


def complete_source_candidates(y):
    # Training supervision only. Validation labels cannot manufacture candidates.
    labels=y['labels'];known=labels[y['train_ids']]
    complete=known[(known>=0).all(1)]
    sets={tuple(np.flatnonzero(row==1).tolist()) for row in complete}
    sets.update((s,) for s in range(labels.shape[1]))
    return [dict(school_ids=list(s),complete_school_set=False) for s in sorted(sets,key=lambda s:(len(s),s)) if s]


def make_named_facts(d,policy,meta):
    path=M.ROOT/'scope_graph_refresh_review/graphs'/f'{d}_changes.json'
    observations=read(path)['school_inventory'];facts=[]
    for r in observations:
        if r['scope_status']!='individual_scope_requires_site_applicability':continue
        names=named_sufficient(r['ast'])
        for site in meta['sites']:
            if exact_identity(site,names):
                facts.append(dict(site_id=site['coordinate_group_id'],school_id=r['school_id'],
                    site_name=site['building_name'],school_name=r['school_name'],
                    basis='exact_named_leaf_sufficient_in_official_scope_AST',
                    source_text=r['source_text'],evidence=r['evidence'],
                    year=policy['year'],stage=policy['stage'],
                    complete_school_set=False,enrollment_eligibility=False))
    return facts,path


def prepare():
    M.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    old=M.verify();M.check_manifest('prediction_freeze.json');M.check_manifest('weights_freeze.json')
    c=dict(date='2026-11-13',backbone='Frozen three-seed ModernNCA-LN categorical control from previous multi-label experiment',
        seeds=M.SEEDS,sources=old['sources'],targets=M.TARGETS,arms=ARMS,
        training_steps=0,weights_and_features_and_reference_pool_unchanged=True,
        primary='set_same_evidence minus categorical_policy',
        evidence_intervention='set_plus_named_evidence minus set_same_evidence; categorical_plus_named_evidence is a separate fixed-baseline factual supplementation control',
        candidates='Single base hypotheses, official shared clauses, directly implied root sets from existing frozen graphs; no transitive closure. Same-scope distinct campuses stay distinct. Known input positives unioned as lower bounds; no partial set is a complete label.',
        applicability='exact named identity -> satisfied; explicit exclusion -> not_satisfied; otherwise unknown. Unknown candidates may be model estimates but never verified eligibility. Points are not administrative membership proof.',
        university='No university or eligibility-member expansion. Existing positive conflicts retained and flagged for review.',
        score='mean(log frozen school probability in set) + beta*log(set size). Geometric mean support with source-calibrated cardinality correction.',
        beta_grid=[0.,.25,.5,.75,1.],
        selection='Source-only validation mean(single exact,multi exact), then overall correct, fewer FP, smaller beta. One ensemble beta reused for all three seeds and both evidence arms; no Shenzhen tuning.',
        source_candidate_boundary='All singletons and complete training-label sets only, excluding validation project groups. Source candidate formation differs from target official-policy candidates; transfer is a limitation.',
        evidence='Separate new exact-name applicability derived only from sufficient named OR leaves in existing reviewed official ASTs. No dropping AND, exclusions, household conditions, unknown branches. This is newly encoded source evidence, not newly acquired documents.',
        scoring='All original1802 outputs;1761map-hit scoring;41outside-map remain output.242semantic-pending rows retained separately. Full-set exact, FP/FN, corrected/harmed,1km district-stratified bootstrap3000.',
        coverage='After freeze only: Gold exact candidate presence and absent-versus-ranking decomposition. Never use target candidate coverage for tuning or new candidates in this run.',
        adoption='Development only; no default changes, no independent validation, no adoption based on this run.',
        target_gold_read=False,target_polygons_read=False,adopted=False,independent_validation=False)
    write(OUT/'protocol.json',c)
    links={str(PARENT/p):sha(PARENT/p) for p in ['protocol.json','prediction_freeze.json','weights_freeze.json','input_links.json']}
    # Pin every inherited input and prediction/weight file, even when not read.
    links.update(read(PARENT/'input_links.json'))
    for manifest in ['prediction_freeze.json','weights_freeze.json']:
        links.update({str(PARENT/p):h for p,h in read(PARENT/manifest).items()})
    summary={}
    for d in M.TARGETS:
        mp=M.CURRENT/'metadata'/f'{d}.json';pp=M.ROOT/'round08_reference_retraining/policies'/f'{d}.json'
        gp=M.OLD/'source_graphs'/f'{d}.json'
        meta=read(mp);policy=read(pp);supp=ReviewedSupplements(d)
        graphs=[read(gp),supp.graph];cat=catalogue(policy,graphs)
        facts,fp=make_named_facts(d,policy,meta)
        for p in [mp,pp,gp,fp]:links[str(p)]=sha(p)
        links.update(supp.hashes)
        for relative,h in policy['input_hashes'].items():
            p=(M.F.SZ.DATA/d/relative).resolve();assert sha(p)==h,p;links[str(p)]=h
        write(OUT/'catalogues'/f'{d}.json',dict(candidates=cat,all_official_school_scopes=policy['school_scope_inventory'],
            all_policy_relations=policy['relations'],school_names={s['school_id']:s['school_name'] for s in meta['schools']}))
        write(OUT/'new_named_evidence'/f'{d}.json',facts)
        bysite=collections.defaultdict(set)
        for f in facts:bysite[f['site_id']].add(f['school_id'])
        site_rows=[];extra=conflicts=0
        ids=[s['school_id'] for s in meta['schools']]
        for q,site in enumerate(meta['sites']):
            sid=site['coordinate_group_id'];known={ids[j] for j in meta['known'].get(str(q),[])}
            # Preserve exactly the old non-model positives and one-pass direct implications.
            lower=set(expand_original(known,graphs[0]))|set(expand_original(known,graphs[1]))
            lower.update(policy_apply(policy,site,sorted(known))['policy_supported_school_ids'])
            existing={k:sorted(v) for k,v in supp.positive[sid].items()}
            for values in existing.values():lower.update(values)
            excluded=set(supp.negative[sid]);lower-=excluded
            new=bysite[sid]-lower;conflict=new&excluded
            # Explicit contradictions are audited and not promoted to facts.
            usable=new-excluded;extra+=len(usable);conflicts+=len(conflict)
            current,audit=site_candidates(cat,site,lower,excluded)
            more,audit_more=site_candidates(cat,site,lower|usable,excluded)
            university=[r['rule_id'] for r in policy['relations'] if r['kind']=='university' and len(lower&set(r['school_ids']))>1]
            site_rows.append(dict(site_id=sid,building_name=site['building_name'],
                original_positive_ids=sorted(known),positive_lower_bound=sorted(lower),
                positive_evidence_kinds=existing,known_kind=meta['known_kind'].get(str(q)),
                exclusions=sorted(excluded),new_named_ids=sorted(usable),new_named_conflicts=sorted(conflict),
                university_positive_conflicts=university,
                same_evidence=current,plus_named=more,applicability=audit,plus_named_applicability=audit_more))
        write(OUT/'site_candidates'/f'{d}.json',site_rows)
        summary[d]=dict(schools=len(ids),sites=len(site_rows),policy_relations=dict(collections.Counter(r['kind'] for r in policy['relations'])),
            catalogue_records=len(cat),unique_multi_templates=len({tuple(r['school_ids']) for r in cat if len(r['school_ids'])>1}),
            sites_with_named_applicable_multi=sum(any(r['applicability']=='satisfied' and len(r['school_ids'])>1 for r in s['same_evidence']) for s in site_rows),
            exact_named_source_relations=len(facts),additional_named_relations=extra,additional_named_sites=sum(bool(s['new_named_ids']) for s in site_rows),
            named_conflicts=conflicts,university_positive_conflicts=sum(bool(s['university_positive_conflicts']) for s in site_rows),
            candidate_applicability=dict(collections.Counter(a['applicability'] for s in site_rows for a in s['applicability'])))
    for d in c['sources']:
        yp=M.OLD/'inputs'/f'{d}_labels.npz';xp=M.OLD/'inputs'/f'{d}.pkl'
        y=dict(np.load(yp));x=M.pkl(xp)
        assert not(set(x.groups[y['train_ids']])&set(x.groups[y['valid_ids']]))
        write(OUT/'source_candidates'/f'{d}.json',complete_source_candidates(y))
    code=[Path(__file__),REPO/'zoning/conditional_school_sets.py',REPO/'tests/test_conditional_school_sets.py',
          REPO/'scripts/audit_modernnca_conditional_sets.py']
    write(OUT/'code_sha256.json',{str(p.relative_to(REPO)):sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'inputs_sha256.json',links);write(OUT/'input_audit.json',summary)
    write(OUT/'candidate_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['catalogues','new_named_evidence','site_candidates','source_candidates'] for p in (OUT/folder).glob('*.json')})
    write(OUT/'protocol_receipt.json',dict(protocol_sha256=sha(OUT/'protocol.json'),target_gold_read=False))
    M.log('PREPARED',summary)


def check(name):
    for p,h in read(OUT/name).items():assert sha(OUT/p)==h,p


def infer():
    M.guard();c=verify();check('candidate_freeze.json')
    pp=probabilities();mean={d:np.stack([p[d] for p in pp]).mean(0) for d in pp[0]}
    trials=[];source_coverage={}
    for beta in c['beta_grid']:
        ns=nm=cs=cm=fp=0
        for d in c['sources']:
            y=dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'))['valid_labels']
            candidates=read(OUT/'source_candidates'/f'{d}.json');school_ids=list(range(y.shape[1]))
            if d not in source_coverage:
                templates={tuple(x['school_ids']) for x in candidates}
                source_coverage[d]=dict(n=len(y),exact_candidate_present=sum(tuple(np.flatnonzero(row)) in templates for row in y),
                    multi=int((y.sum(1)>1).sum()),multi_templates=sum(len(s)>1 for s in templates))
            for row,p in zip(y,mean[d]):
                pred=set(select(p,candidates,school_ids,beta)['school_ids']);gold=set(np.flatnonzero(row));ok=pred==gold
                if len(gold)>1:nm+=1;cm+=int(ok)
                else:ns+=1;cs+=int(ok)
                fp+=len(pred-gold)
        trials.append(dict(beta=beta,n=ns+nm,correct=cs+cm,single_correct=cs,multi_correct=cm,single_n=ns,multi_n=nm,
            balanced=.5*(cs/ns+cm/nm),fp=fp))
    chosen=max(trials,key=lambda r:(r['balanced'],r['correct'],-r['fp'],-r['beta']))
    beta=chosen['beta'];write(OUT/'source_selection.json',dict(selected=chosen,trials=trials,source_candidate_coverage=source_coverage,target_gold_read=False))
    for label,prob in [('ensemble',mean)]+list(zip(map(str,c['seeds']),pp)):
        for d in M.TARGETS:
            meta=read(M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
            inventory=read(OUT/'site_candidates'/f'{d}.json')
            baseline=read(PARENT/'predictions/categorical_policy'/label/f'{d}.json')
            assert [s['site_id'] for s in inventory]==[s['site_id'] for s in baseline]
            arms={a:[] for a in ARMS}
            for q,(site,old) in enumerate(zip(inventory,baseline)):
                base={k:old[k] for k in ['district','site_id','building_name']}
                arms['categorical_policy'].append(dict(base,after=old['after']))
                arms['categorical_plus_named_evidence'].append(dict(base,after=sorted(set(old['after'])|set(site['new_named_ids']))))
                for arm,key in [('set_same_evidence','same_evidence'),('set_plus_named_evidence','plus_named')]:
                    decision=select(prob[d][q],site[key],ids,beta)
                    arms[arm].append(dict(base,after=decision['school_ids'],decision=decision))
            for arm,rows in arms.items():write(OUT/'predictions'/arm/label/f'{d}.json',rows)
    files=[p for p in (OUT/'predictions').rglob('*.json')]+[OUT/'source_selection.json']
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    M.log('PREDICTIONS FROZEN beta',beta,'source',chosen)


def score():
    c=verify();check('candidate_freeze.json');check('prediction_freeze.json')
    base=REPO/'outputs/shenzhen_aligned_model_20261111'
    gold={(r['district'],r['site_id']):r for r in read(base/'scored_cases.json')}
    gold_hash=read(base/'results.json')['gold_sha256']
    for d,h in gold_hash.items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    xy={};inventories={}
    for d in M.TARGETS:
        x=M.pkl(REPO/'data/深圳市/修复实验输入_20261112'/d/'district.pkl')
        xy.update({(d,s):p for s,p in zip(x.site_ids,x.coords)})
        inventories.update({(d,s['site_id']):s for s in read(OUT/'site_candidates'/f'{d}.json')})
    groups={'全部':lambda r:True,**{d:(lambda r,d=d:r['d']==d) for d in M.TARGETS},
        '单校':lambda r:r['size']==1,'多校':lambda r:r['size']>1,
        '语义待核':lambda r:r['semantic_pending'],'其余':lambda r:not r['semantic_pending']}
    metrics={};all_rows={};seeds={}
    for arm in ARMS:
        seeds[arm]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            rows=[];unscored=0
            for d in M.TARGETS:
                pred=read(OUT/'predictions'/arm/label/f'{d}.json')
                assert len({r['site_id'] for r in pred})==len(pred)
                assert {r['site_id'] for r in pred}=={k[1] for k in gold if k[0]==d}
                for r in pred:
                    key=(d,r['site_id']);g=gold[key]
                    if not g['has_map_gold']:unscored+=1;continue
                    p=set(r['after']);truth=set(g['gold']);pt=xy[key];site=inventories[key]
                    cat=site['plus_named' if arm in ['set_plus_named_evidence','categorical_plus_named_evidence'] else 'same_evidence']
                    match=[a for a in cat if set(a['school_ids'])==truth]
                    rows.append(dict(d=d,site_id=key[1],name=r['building_name'],prediction=sorted(p),truth=sorted(truth),
                        correct=p==truth,size=len(truth),tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),semantic_pending=g['semantic_pending'],
                        block=str((int(pt[0]//1000),int(pt[1]//1000))),candidate_present=bool(match),
                        exact_candidate_has_named_scope=any(m['applicability']=='satisfied' for m in match),
                        lower_bound_conflicts=sorted(set(site['positive_lower_bound'])-truth),
                        new_named_conflicts=sorted(set(site['new_named_ids'])-truth)))
            assert len(rows)==1761 and unscored==41
            seeds[arm][label]={k:M.metric([r for r in rows if fn(r)]) for k,fn in groups.items()}
            if label=='ensemble':
                metrics[arm]=seeds[arm][label];all_rows[arm]=rows;write(OUT/'scored_rows'/f'{arm}.json',rows)
    comparisons={}
    for a,b in [('set_same_evidence','categorical_policy'),('set_plus_named_evidence','set_same_evidence'),
                ('categorical_plus_named_evidence','categorical_policy')]:
        comparisons[f'{a} minus {b}']={}
        for g,fn in groups.items():
            pairs=[(r,s) for r,s in zip(all_rows[a],all_rows[b]) if fn(r)]
            delta=[int(r['correct'])-int(s['correct']) for r,s in pairs]
            comparisons[f'{a} minus {b}'][g]=dict(delta_pp=100*float(np.mean(delta)),corrected=delta.count(1),harmed=delta.count(-1),
                ci95=M.evaluate.bootstrap([r for r,s in pairs],delta,20261104,3000))
    coverage={}
    for arm in ['set_same_evidence','set_plus_named_evidence']:
        coverage[arm]={}
        for g,fn in groups.items():
            rows=[r for r in all_rows[arm] if fn(r)]
            coverage[arm][g]=dict(n=len(rows),exact_candidate_present=sum(r['candidate_present'] for r in rows),
                candidate_absent=sum(not r['candidate_present'] for r in rows),
                present_but_selected_wrong=sum(r['candidate_present'] and not r['correct'] for r in rows),
                selected_correct=sum(r['correct'] for r in rows),
                exact_candidate_has_named_scope=sum(r['exact_candidate_has_named_scope'] for r in rows),
                existing_lower_bound_conflicts=sum(bool(r['lower_bound_conflicts']) for r in rows),
                new_named_vs_map_conflicts=sum(bool(r['new_named_conflicts']) for r in rows))
    result=dict(metrics=metrics,comparisons=comparisons,seeds=seeds,candidate_coverage=coverage,gold_sha256=gold_hash,
                adopted=False,independent_validation=False,training_steps=0)
    write(OUT/'results.json',result);write(DOC/'结果.json',result)
    M.log('RESULTS',metrics,'COMPARISONS',comparisons['set_same_evidence minus categorical_policy']['全部'],'COVERAGE',coverage)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','infer','score']);globals()[ap.parse_args().phase]()
