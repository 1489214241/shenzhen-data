"""Frozen Modern + advance shared-school prior. No production changes.

Run prepare, infer, score, audit as separate processes. Test labels are blocked
until all policy priors, ranks and predictions have been sealed.
"""
import argparse
import collections
from pathlib import Path
import numpy as np
import experiment_conditional_admission_hybrid as H
from zoning.shared_school_prior import build_prior, decode_prior

A = H.A
M = A.M
read, write, sha = A.read, A.write, A.sha
REPO = M.REPO
INPUT = A.E.INPUT
OUT = REPO / 'outputs/modern_shared_school_prior_20261114'
DOC = REPO.parent / '科研推进/共享学区提前输入实验_20261114'
ARMS = {
    'text_minimum': ('minimum', False, 'admission'),
    'text_group_exact': ('group_exact', False, 'admission'),
    'text_count_admission': ('count_exact', False, 'admission'),
    'text_count_native': ('count_exact', False, 'native'),
    'estimated_minimum': ('minimum', True, 'admission'),
    'estimated_group_exact': ('group_exact', True, 'admission'),
    'estimated_count_admission': ('count_exact', True, 'admission'),
    'estimated_count_native': ('count_exact', True, 'native'),
}


def check(manifest):
    for p, h in read(OUT / manifest).items():
        assert sha(p) == h, p


def verify():
    check('inputs_sha256.json')
    return read(OUT / 'protocol.json')


def prepare():
    M.guard()
    if OUT.exists() and not ((OUT/'preparation_fix_receipt.json').exists()
                             and not (OUT/'inputs_sha256.json').exists()):
        raise FileExistsError('Do not overwrite a sealed experiment')
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read(H.OUT / 'protocol.json')
    cfg = dict(date='2026-11-14', sources=previous['sources'], targets=previous['targets'],
        seeds=previous['seeds'], arms=ARMS, baseline='routed_bce_legacy from frozen hybrid run',
        primary='text_minimum minus baseline_current', training_steps=0,
        intervention='Policy prior is assembled before final set selection; frozen Modern native/admission scores are reused. Not a retrained network or prompt experiment.',
        verified_prior='Independent textual positive relations plus named or directly triggered shared clauses. Never use model scope estimates as confirmed input.',
        estimated_prior='Secondary arm only: if verified multi prior unavailable, original native top1 may trigger its directly implied shared group. Unknown applicability remains estimated.',
        cardinality='Policy members are positive lower bounds, NOT exhaustive sets. Exact-group and exact-count arms explicitly TEST completeness; they never relabel unknown schools as training negatives.',
        count_ablation='Give only k, rank all nonexcluded schools; do not force supplied member IDs. The university exception preserves independently supported sharing only.',
        minimum='Union prior members with current prediction. Group-exact replaces active output by prior group. Count-exact replaces active output by top k. Uncovered outputs unchanged.',
        university='No university membership expansion; at most one per university group unless explicit shared/base evidence supports more.',
        scope='Shared partial ranges preserve names/direction. No closure across shared groups. All school counts >=2 supported, not just pairs.',
        scoring='Same1761map-hit rows;41output-only;242semantic-pending retained. Exact full sets, precision/recall, FP/FN, corrected/harmed,1km district bootstrap3000.',
        oracle='AFTER prediction freeze only: read answer cardinality to diagnose frozen raw rankings; diagnostic only, never a deployable gain. Both rankings fixed beforehand.',
        selection='No fitting, thresholds, grid, or target selection. All declared arms reported; no adoption.',
        prior_target_results_seen=True, independent_validation=False, adopted=False,
        target_gold_read=False, target_polygon_read=False)
    write(OUT / 'protocol.json', cfg)
    links = {str(OUT/'protocol.json'): sha(OUT/'protocol.json')}
    for p in [OUT/'preparation_fix_receipt.json', OUT/'prepare_before_path_fix.py',
              REPO/'data/深圳市/当前数据集.json']:
        if p.exists(): links[str(p)] = sha(p)
    for p in [Path(__file__), REPO/'zoning/shared_school_prior.py',
              REPO/'tests/test_shared_school_prior.py', REPO/'zoning/conditional_school_sets.py']:
        links[str(p)] = sha(p)
        dest = OUT/'code'/p.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    pred_manifest = read(H.OUT/'prediction_freeze.json')
    old_freeze = read(H.OUT/'freeze.json')
    snapshot = read(INPUT/'manifest.json')
    for p in [H.OUT/'protocol.json', H.OUT/'prediction_freeze.json', H.OUT/'freeze.json', INPUT/'manifest.json']:
        links[str(p)] = sha(p)
    for d in cfg['targets']:
        for f in ['metadata.json', 'constraints.json', 'reviewed_policy.json', 'evidence_ledger.json']:
            p = INPUT/d/f
            assert sha(p) == snapshot['files'][str(p.relative_to(INPUT))]
            links[str(p)] = sha(p)
        policy = read(INPUT/d/'reviewed_policy.json')
        for rel, h in policy['input_hashes'].items():
            p = (M.F.SZ.DATA/d/rel).resolve()
            assert sha(p) == h
            links[str(p)] = h
        p = H.OUT/'cache/control'/f'{d}.npz'
        assert sha(p) == old_freeze[str(p)]
        links[str(p)] = sha(p)
        for label in ['ensemble'] + list(map(str, cfg['seeds'])):
            for arm in ['routed_bce_legacy', 'routed_bce_raw']:
                p = H.OUT/'predictions'/arm/label/f'{d}.json'
                assert sha(p) == pred_manifest[str(p.relative_to(H.OUT))]
                links[str(p)] = sha(p)
        p = H.OUT/'predictions/control_base_legacy/ensemble'/f'{d}.json'
        assert sha(p) == pred_manifest[str(p.relative_to(H.OUT))]
        links[str(p)] = sha(p)
    for seed in cfg['seeds']:
        p = H.OUT/'probabilities/routed_bce'/f'{seed}.npz'
        assert sha(p) == pred_manifest[str(p.relative_to(H.OUT))]
        links[str(p)] = sha(p)
    write(OUT/'inputs_sha256.json', links)
    M.log('SHARED PRIOR PROTOCOL SEALED', cfg['primary'])


def infer():
    M.guard(); cfg = verify()
    if (OUT/'prediction_freeze.json').exists():
        raise FileExistsError('Do not overwrite frozen predictions')
    payload = [dict(np.load(H.OUT/'probabilities/routed_bce'/f'{s}.npz')) for s in cfg['seeds']]
    means = {d: np.stack([p[d] for p in payload]).mean(0) for d in cfg['targets']}
    audit = {}
    for d in cfg['targets']:
        meta = read(INPUT/d/'metadata.json')
        cons = read(INPUT/d/'constraints.json')
        policy = read(INPUT/d/'reviewed_policy.json')
        ledger = read(INPUT/d/'evidence_ledger.json')
        # Independently check that constraint positives are all text-supported.
        proofs = {(r['site_id'], r['school_id']) for r in ledger
                  if r['positive'] and r['routes']['hard_constraint']
                  and r['routes']['evidence_type'] == 'textual_proof'}
        assert all((site, s) in proofs for site, con in cons.items() for s in con['positive'])
        ids = [s['school_id'] for s in meta['schools']]
        native = np.load(H.OUT/'cache/control'/f'{d}.npz')['prob']
        universities = [r['school_ids'] for r in policy['relations'] if r['kind']=='university']
        priors = []
        for q, site in enumerate(meta['sites']):
            sid = site['coordinate_group_id']; con = cons[sid]
            prior = build_prior(policy, site, con['positive'], con['negative'], ids[int(native[q].argmax())])
            priors.append(dict(prior, site_id=sid, building_name=site['building_name']))
        write(OUT/'priors'/f'{d}.json', priors)
        write(OUT/'policies'/f'{d}.json', policy)
        audit[d] = dict(sites=len(priors), school_count=len(ids),
            shared_clauses=sum(r['kind']=='shared' and len(r['school_ids'])>1 for r in policy['relations']),
            verified_multi_sites=sum(len(r['verified_members'])>=2 for r in priors),
            estimated_only_multi_sites=sum(len(r['estimated_members'])>=2 and len(r['verified_members'])<2 for r in priors),
            verified_counts=dict(collections.Counter(len(r['verified_members']) for r in priors)),
            blocked=sum(bool(r['blocked_rules']) for r in priors))
        for label, pp in [('ensemble', means)] + list(zip(map(str, cfg['seeds']), payload)):
            baseline = read(H.OUT/'predictions/routed_bce_legacy'/label/f'{d}.json')
            assert [r['site_id'] for r in baseline] == [r['site_id'] for r in priors]
            arms = {a: [] for a in ARMS}
            for q, (old, prior) in enumerate(zip(baseline, priors)):
                con = cons[old['site_id']]
                for arm, (mode, estimated, ranking) in ARMS.items():
                    scores = native[q] if ranking=='native' else pp[d][q]
                    after, decision = decode_prior(old['after'], prior, scores, ids, mode, estimated,
                                                   con['negative'], universities)
                    arms[arm].append(dict(district=d, site_id=old['site_id'], building_name=old['building_name'],
                        after=after, decision=decision, complete_school_set=False, enrollment_eligibility=False))
            for arm, rows in arms.items():
                write(OUT/'predictions'/arm/label/f'{d}.json', rows)
            write(OUT/'predictions/baseline_current'/label/f'{d}.json', baseline)
        old_native = read(H.OUT/'predictions/control_base_legacy/ensemble'/f'{d}.json')
        write(OUT/'predictions/baseline_native/ensemble'/f'{d}.json', old_native)
        # Freeze full rankings now, before oracle diagnostic can see a count.
        write(OUT/'rankings'/f'{d}.json', [dict(site_id=s['coordinate_group_id'],
              native=sorted(ids, key=lambda sid: (-float(native[q, ids.index(sid)]), sid)),
              admission=sorted(ids, key=lambda sid: (-float(means[d][q, ids.index(sid)]), sid)))
              for q, s in enumerate(meta['sites'])])
    write(OUT/'input_audit.json', dict(passed=True, coverage=audit, target_gold_read=False))
    files = [p for folder in ['priors', 'policies', 'predictions', 'rankings'] for p in (OUT/folder).rglob('*.json')]
    files.append(OUT/'input_audit.json')
    write(OUT/'prediction_freeze.json', {str(p):sha(p) for p in files})
    M.log('SHARED PRIOR PREDICTIONS SEALED', audit)


def score():
    cfg = verify(); check('prediction_freeze.json')
    if (OUT/'results.json').exists():
        raise FileExistsError('Do not overwrite scored results')
    goldpath = REPO/'outputs/modernnca_partial_sets_20261113/scored_rows/categorical_policy.json'
    gold = read(goldpath)
    truth = {(r['d'], r['site_id']): r for r in gold}
    old_results = read(REPO/'outputs/modernnca_partial_sets_20261113/results.json')
    hashes = old_results['gold_sha256']
    for d, h in hashes.items():
        assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json') == h
    priors = {(d,r['site_id']):r for d in cfg['targets'] for r in read(OUT/'priors'/f'{d}.json')}
    groups = {'all':lambda r:True, 'single':lambda r:r['size']==1,
              'multi':lambda r:r['size']>1, 'three_plus':lambda r:r['size']>=3,
              'semantic_pending':lambda r:r['semantic_pending'],
              'verified_prior':lambda r:r['verified_prior'], 'estimated_only':lambda r:r['estimated_only'],
              **{d:lambda r,d=d:r['d']==d for d in cfg['targets']}}
    metrics, seed_metrics, rows_by = {}, {}, {}
    for arm in ['baseline_current', 'baseline_native'] + list(ARMS):
        seed_metrics[arm] = {}
        labels = ['ensemble'] if arm=='baseline_native' else ['ensemble']+list(map(str,cfg['seeds']))
        for label in labels:
            rows=[]; outside=0
            for d in cfg['targets']:
                predictions=read(OUT/'predictions'/arm/label/f'{d}.json')
                meta=read(INPUT/d/'metadata.json')
                assert len(predictions)==len(meta['sites'])
                assert {r['site_id'] for r in predictions}=={s['coordinate_group_id'] for s in meta['sites']}
                for r in predictions:
                    key=(d,r['site_id'])
                    if key not in truth:outside+=1;continue
                    g=truth[key]; p=set(r['after']); t=set(g['truth']); prior=priors[key]
                    rows.append(dict(g, prediction=sorted(p), correct=p==t, tp=len(p&t),fp=len(p-t),fn=len(t-p),
                                     verified_prior=len(prior['verified_members'])>=2,
                                     estimated_only=len(prior['verified_members'])<2 and len(prior['estimated_members'])>=2))
            assert len(rows)==1761 and outside==41 and sum(r['semantic_pending'] for r in rows)==242
            mm={k:M.metric([r for r in rows if f(r)]) for k,f in groups.items()}
            seed_metrics[arm][label]=mm
            if label=='ensemble':
                rows_by[arm]=rows; metrics[arm]=mm;write(OUT/'scored_rows'/f'{arm}.json',rows)
    comparisons={}
    before={(r['d'],r['site_id']):r for r in rows_by['baseline_current']}
    for arm in ARMS:
        comparisons[arm]={}
        for name,f in groups.items():
            rows=[r for r in rows_by[arm] if f(r)]
            delta=[int(r['correct'])-int(before[r['d'],r['site_id']]['correct']) for r in rows]
            comparisons[arm][name]=dict(corrected=delta.count(1),harmed=delta.count(-1),
                delta_pp=100*float(np.mean(delta)),ci95=M.evaluate.bootstrap(rows,delta,20261104,3000))
    diagnosis={}
    for kind in ['verified_members','estimated_members']:
        rr=[(truth[k],p[kind]) for k,p in priors.items() if k in truth and len(p[kind])>=2]
        diagnosis[kind]=dict(n=len(rr), lower_bound_supported=sum(set(s)<=set(g['truth']) for g,s in rr),
            cardinality_equal=sum(len(s)==len(g['truth']) for g,s in rr),
            exact_group=sum(set(s)==set(g['truth']) for g,s in rr),
            input_school_conflict=sum(not set(s)<=set(g['truth']) for g,s in rr),
            incomplete_positive_set=sum(set(s)<set(g['truth']) for g,s in rr))
    # Isolated oracle: answers only supply k after all rankings/predictions freeze.
    oracle={}
    for kind in ['native','admission']:
        rows=[]
        for d in cfg['targets']:
            for r in read(OUT/'rankings'/f'{d}.json'):
                key=d,r['site_id']
                if key not in truth:continue
                g=truth[key]; t=set(g['truth']); p=set(r[kind][:len(t)])
                rows.append(dict(g,prediction=sorted(p),correct=p==t,tp=len(p&t),fp=len(p-t),fn=len(t-p)))
        oracle[kind]={k:M.metric([r for r in rows if f(r)]) for k,f in groups.items()
                      if k not in ['verified_prior','estimated_only']}
        write(OUT/'diagnostics_not_training'/f'oracle_{kind}.json',rows)
    result=dict(metrics=metrics,seeds=seed_metrics,comparisons=comparisons,prior_diagnostics=diagnosis,
                oracle_count_diagnostic_only=oracle,primary=cfg['primary'],gold_sha256=hashes,
                scored_gold_file_sha256=sha(goldpath),adopted=False,independent_validation=False)
    write(OUT/'results.json',result)
    M.log('SHARED PRIOR RESULTS',{a:m['all']['correct'] for a,m in metrics.items()},diagnosis)


def audit():
    cfg=verify();check('prediction_freeze.json');result=read(OUT/'results.json')
    for d,h in result['gold_sha256'].items():
        assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    checked=0
    for arm,metrics in result['metrics'].items():
        rows=read(OUT/'scored_rows'/f'{arm}.json')
        assert len(rows)==1761
        for r in rows:
            p=set(r['prediction']);t=set(r['truth'])
            assert r['correct']==(p==t) and r['tp']==len(p&t) and r['fp']==len(p-t) and r['fn']==len(t-p)
            checked+=1
        assert metrics['all']['correct']==sum(set(r['prediction'])==set(r['truth']) for r in rows)
    for d in cfg['targets']:
        priors=read(OUT/'priors'/f'{d}.json');cons=read(INPUT/d/'constraints.json')
        ranks=read(OUT/'rankings'/f'{d}.json');meta=read(INPUT/d/'metadata.json')
        policy=read(INPUT/d/'reviewed_policy.json')
        uni=[set(r['school_ids']) for r in policy['relations'] if r['kind']=='university']
        for label in ['ensemble']+list(map(str,cfg['seeds'])):
            before=read(OUT/'predictions/baseline_current'/label/f'{d}.json')
            original=read(H.OUT/'predictions/routed_bce_legacy'/label/f'{d}.json')
            assert before==original
            for arm,(mode,estimated,ranking) in ARMS.items():
                after=read(OUT/'predictions'/arm/label/f'{d}.json')
                for q,(b,a,prior) in enumerate(zip(before,after,priors)):
                    members=set(prior['estimated_members' if estimated else 'verified_members'])
                    p=set(a['after']);neg=set(cons[b['site_id']]['negative'])
                    if len(members)<2:assert a['after']==b['after'];continue
                    assert not p&neg
                    if mode=='minimum':assert p==(set(b['after'])|members)-neg
                    elif mode=='group_exact':assert p==members-neg
                    elif label=='ensemble':
                        expected=[]
                        for sid in ranks[q][ranking]:
                            if sid in neg:continue
                            if any(sid in g and set(expected)&g and not ({sid}|(set(expected)&g))<=members for g in uni):continue
                            expected.append(sid)
                            if len(expected)==len(members):break
                        assert p==set(expected)
                    checked+=1
    write(OUT/'output_audit.json',dict(passed=True,checked_rows=checked,
        original_baseline_identical=True,gold_unchanged=True,
        ensemble_count_decoding_independently_reconstructed=True))
    M.log('SHARED PRIOR AUDIT PASSED',checked)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','infer','score','audit'])
    globals()[parser.parse_args().phase]()
