"""Second development candidate: scope-branch candidates, frozen decoder/inputs.

The preceding target results have been inspected. This is explicitly not an
independent confirmation. No target-label-defined sets or per-case patches.
"""
import argparse
import collections
from pathlib import Path
import numpy as np
import experiment_modernnca_conditional_sets as E
from zoning.conditional_scope_cells import build_cells
from zoning.conditional_school_sets import site_candidates,select

OUT=E.REPO/'outputs/modernnca_scope_cells_20261113'
DOC=E.REPO.parent/'科研推进/ModernNCA条件学校集合_20261113'
ARMS=['categorical_policy','set_same_evidence','scope_cell_sets']
read=E.read;write=E.write;sha=E.sha


def verify():
    E.verify();E.check('prediction_freeze.json');E.check('candidate_freeze.json')
    for p,h in read(OUT/'inputs_sha256.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'code_sha256.json').items():assert sha(E.REPO/p)==h,p
    return read(OUT/'protocol.json')


def check(name):
    for p,h in read(OUT/name).items():assert sha(OUT/p)==h,p


def prepare():
    E.M.guard();E.verify();E.check('prediction_freeze.json');E.check('candidate_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve scope-cell experiment')
    previous=read(E.OUT/'protocol.json');beta=read(E.OUT/'source_selection.json')['selected']['beta']
    protocol=dict(previous,arms=ARMS,beta_fixed=beta,primary='scope_cell_sets minus categorical_policy',
        secondary='scope_cell_sets minus set_same_evidence',
        intervention='Candidate representation only. Collect all sufficient official AST OR branches across202schools; each school in a condition-set has an independent direct condition->scope proof. Keep conjunctions/exclusions intact; no school graph closure.',
        prior_target_results_seen=True,independent_validation=False,
        calibration='Reuse frozen source-selected beta without search or retuning. Same source weights, school probabilities, features, references and target positive lower bounds as first experiment.',
        target_specific_set_patches=False,new_source_evidence_relations=0,
        source_target_candidate_mismatch='Source calibration candidates derive from complete source training labels; target cells derive from policy branches. This is a fixed decoder, not a learned conditional-set encoder.')
    write(OUT/'protocol.json',protocol);links={str(E.OUT/p):sha(E.OUT/p) for p in ['protocol.json','source_selection.json','candidate_freeze.json','prediction_freeze.json']}
    summary={}
    for d in E.M.TARGETS:
        pp=E.M.ROOT/'round08_reference_retraining/policies'/f'{d}.json'
        op=E.M.ROOT/'scope_graph_refresh_review/graphs'/f'{d}_changes.json'
        meta=read(E.M.CURRENT/'metadata'/f'{d}.json');policy=read(pp);observations=read(op)['school_inventory']
        for p in [pp,op]:links[str(p)]=sha(p)
        cells,audit=build_cells(observations,policy)
        old=read(E.OUT/'catalogues'/f'{d}.json');cat=old['candidates']+cells
        write(OUT/'catalogues'/f'{d}.json',dict(old,candidates=cat,new_scope_cells=cells,branch_inventory=audit))
        oldsites=read(E.OUT/'site_candidates'/f'{d}.json');rows=[]
        for site,oldsite in zip(meta['sites'],oldsites):
            assert site['coordinate_group_id']==oldsite['site_id']
            candidates,status=site_candidates(cat,site,oldsite['positive_lower_bound'],oldsite['exclusions'])
            rows.append(dict(oldsite,scope_cell_candidates=candidates,scope_cell_applicability=status))
        write(OUT/'site_candidates'/f'{d}.json',rows)
        oldmulti={tuple(r['school_ids']) for r in old['candidates'] if len(r['school_ids'])>1}
        newmulti={tuple(r['school_ids']) for r in cat if len(r['school_ids'])>1}
        summary[d]=dict(schools=len(audit),supported_scope_schools=sum(bool(r['admitted_branches']) for r in audit),
            scope_conditions=len(cells),old_unique_multi_templates=len(oldmulti),new_unique_multi_templates=len(newmulti),
            additional_unique_multi_templates=len(newmulti-oldmulti),target_gold_read=False)
    code=[Path(__file__),E.REPO/'zoning/conditional_scope_cells.py',E.REPO/'tests/test_conditional_scope_cells.py',E.REPO/'scripts/audit_modernnca_scope_cells.py']
    write(OUT/'code_sha256.json',{str(p.relative_to(E.REPO)):sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(E.REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'inputs_sha256.json',links);write(OUT/'input_audit.json',summary)
    write(OUT/'candidate_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['catalogues','site_candidates'] for p in (OUT/folder).glob('*.json')})
    write(OUT/'protocol_receipt.json',dict(protocol_sha256=sha(OUT/'protocol.json'),target_gold_read_in_this_phase=False,prior_target_results_seen=True))
    print('PREPARED SCOPE CELLS',summary,flush=True)


def infer():
    E.M.guard();c=verify();check('candidate_freeze.json');pp=E.probabilities()
    mean={d:np.stack([p[d] for p in pp]).mean(0) for d in pp[0]}
    for label,probs in [('ensemble',mean)]+list(zip(map(str,c['seeds']),pp)):
        for d in E.M.TARGETS:
            meta=read(E.M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
            sites=read(OUT/'site_candidates'/f'{d}.json');rows=[]
            for q,site in enumerate(sites):
                decision=select(probs[d][q],site['scope_cell_candidates'],ids,c['beta_fixed'])
                rows.append(dict(district=d,site_id=site['site_id'],building_name=site['building_name'],after=decision['school_ids'],decision=decision))
            write(OUT/'predictions/scope_cell_sets'/label/f'{d}.json',rows)
            for arm in ['categorical_policy','set_same_evidence']:
                write(OUT/'predictions'/arm/label/f'{d}.json',read(E.OUT/'predictions'/arm/label/f'{d}.json'))
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'predictions').rglob('*.json')})
    print('SCOPE CELL PREDICTIONS SEALED',flush=True)


def score():
    c=verify();check('prediction_freeze.json');check('candidate_freeze.json')
    old=read(E.OUT/'results.json');rows=read(E.OUT/'scored_rows/categorical_policy.json')
    # These are frozen original scoring rows, not inputs to prepare/infer.
    bykey={(r['d'],r['site_id']):r for r in rows};metrics={};seeds={};allrows={}
    groups={'全部':lambda r:True,**{d:(lambda r,d=d:r['d']==d) for d in E.M.TARGETS},
        '单校':lambda r:r['size']==1,'多校':lambda r:r['size']>1,
        '语义待核':lambda r:r['semantic_pending'],'其余':lambda r:not r['semantic_pending']}
    for arm in ARMS:
        seeds[arm]={}
        for label in ['ensemble']+list(map(str,c['seeds'])):
            scored=[];outside=0
            for d in E.M.TARGETS:
                sites={s['site_id']:s for s in read(OUT/'site_candidates'/f'{d}.json')}
                predictions=read(OUT/'predictions'/arm/label/f'{d}.json')
                assert len(predictions)==len(sites) and {r['site_id'] for r in predictions}==set(sites)
                for r in predictions:
                    key=(d,r['site_id'])
                    if key not in bykey:outside+=1;continue
                    base=bykey[key];g=set(base['truth']);p=set(r['after']);site=sites[r['site_id']]
                    cat=site['scope_cell_candidates'] if arm=='scope_cell_sets' else site['same_evidence']
                    matches=[v for v in cat if set(v['school_ids'])==g]
                    scored.append(dict(base,prediction=sorted(p),correct=p==g,tp=len(p&g),fp=len(p-g),fn=len(g-p),
                        candidate_present=bool(matches),exact_candidate_has_named_scope=any(m['applicability']=='satisfied' for m in matches)))
            assert len(scored)==1761 and outside==41
            seeds[arm][label]={g:E.M.metric([r for r in scored if f(r)]) for g,f in groups.items()}
            if label=='ensemble':
                metrics[arm]=seeds[arm][label];allrows[arm]=scored;write(OUT/'scored_rows'/f'{arm}.json',scored)
    comparisons={}
    for b in ['categorical_policy','set_same_evidence']:
        a='scope_cell_sets';comparisons[f'{a} minus {b}']={}
        for g,f in groups.items():
            paired=[(r,s) for r,s in zip(allrows[a],allrows[b]) if f(r)]
            delta=[int(r['correct'])-int(s['correct']) for r,s in paired]
            comparisons[f'{a} minus {b}'][g]=dict(delta_pp=100*float(np.mean(delta)),corrected=delta.count(1),harmed=delta.count(-1),
                ci95=E.M.evaluate.bootstrap([r for r,s in paired],delta,20261104,3000))
    coverage={}
    for g,f in groups.items():
        rr=[r for r in allrows['scope_cell_sets'] if f(r)]
        coverage[g]=dict(n=len(rr),candidate_present=sum(r['candidate_present'] for r in rr),candidate_absent=sum(not r['candidate_present'] for r in rr),
            present_but_selected_wrong=sum(r['candidate_present'] and not r['correct'] for r in rr),selected_correct=sum(r['correct'] for r in rr),
            lower_bound_conflicts=sum(bool(r['lower_bound_conflicts']) for r in rr))
    result=dict(metrics=metrics,seeds=seeds,comparisons=comparisons,candidate_coverage=coverage,gold_sha256=old['gold_sha256'],
        adopted=False,independent_validation=False,prior_target_results_seen=True,training_steps=0)
    for d,h in result['gold_sha256'].items():assert sha(E.M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    write(OUT/'results.json',result);write(DOC/'子范围集合结果.json',result)
    print('RESULTS',metrics,'COMPARISON',comparisons['scope_cell_sets minus categorical_policy']['全部'],'COVERAGE',coverage,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','infer','score']);globals()[ap.parse_args().phase]()
