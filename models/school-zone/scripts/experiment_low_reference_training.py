"""Frozen K1 Shenzhen trial: 2 training views x 2 Modern query constructions.

Run prepare, fit, score, audit in separate processes. Source training labels are
legal; target labels are restricted to the existing K1 release and policy facts.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import experiment_school_gap_reference as G
import experiment_modernnca_shenzhen_multilabel as M
from zoning.low_reference_training import choose_relations, rebuild_view, SchoolLocalModernNCA
from zoning.modernnca_multilabel import MultiLabelModernNCA, observed_bce_loss
from shenzhen_balanced_ranking import balanced_weights
from shenzhen_exact_calibration import exact_thresholds

E = G.E
OUT = REPO / 'outputs/modern_low_reference_training_20261114'
ARMS = ['global_full', 'global_sparse', 'local_full', 'local_sparse']
read, write, sha, log = E.read, E.write, E.sha, E.log


def guard():
    E.guard()
    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0]).replace('\\', '/')
            if ('modern_school_gap_reference_20261114' in p and
                any(s in p for s in ['released_plans.json', '/weights/', '/predictions/'])):
                raise PermissionError('Use only isolated K1 release: ' + p)
            if 'modern_complete_reference_20261114' in p and any(s in p for s in ['/weights/', 'released_support.json']):
                raise PermissionError('512-target information forbidden: ' + p)
    sys.addaudithook(hook)


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve frozen experiment')
    # Preparation isolates only K1. Training never opens the joint K1/K2/K3 file.
    plans = read(G.OUT / 'released_plans.json')
    receipt = read(G.OUT / 'selection_freeze.json')
    assert sha(G.OUT / 'released_plans.json') == receipt['plans']
    release = {d: plans[d]['1'] for d in E.DISTRICTS}
    write(OUT / 'k1_release.json', release)
    write(OUT / 'release_receipt.json', dict(parent_sha256=receipt['plans'],
        budget=1, selected=sum(len(v['support']) for v in release.values()),
        oracle='Existing school-aware Gold simulation; no new acquisition'))
    guard()
    c = dict(date='2026-11-14', arms=ARMS, seeds=E.SEEDS,
        sources=read(M.OLD / 'protocol.json')['sources'], targets=E.DISTRICTS,
        lr=.001, reg=.001, steps=320, checkpoints=[40,80,160,320], batch=32,
        dim=32, clip=5., episodes=8, episode_seed=20261114,
        primary='global_sparse_policy minus global_full_policy',
        secondary=['local_sparse_policy minus local_full_policy',
                   'local_full_policy minus global_full_policy',
                   'local_sparse_policy minus global_sparse_policy'],
        hypothesis='Training on scarce reference relations improves open-set school combination prediction without new target labels.',
        initialization='Fresh identical random initializations for paired arms; no target-adapted or pretrained weights.',
        source_support='Original independent policy references restricted to source training-fold groups. Full keeps all; sparse selects one relation per represented school, uniformly by project then coordinate. Zero-support schools remain candidates.',
        source_episodes='Eight fixed sparse source views, cycled at training steps. Same query minibatches across arms; no target support resampling. Query own project excluded; it can therefore have zero remaining references for its school.',
        feature_rebuild='Recompute learned boundaries from visible ORIGINAL singleton observations, group multiplicities, membership, known-row name suppression, all six context features. Static policy/road/community observations remain independent input.',
        scope='Sparse SOURCE relation masking is not strict one-acquisition-per-school deployment. Target keeps all original weak refs, verified partial positives and fixed84 K1 complete supports.',
        labels='Same source complete/partial labels and target old/verified partial positives plus K1 complete supports in all arms. Unknown is -1, not a negative. Complete support overrides partial estimate at same coordinate.',
        validation='Same1246 source spatial holdout rows. Source support excludes ALL validation-fold groups. One separate fixed sparse view shared across arms. Source balanced single/multi exact selects checkpoint and global threshold, then fewer FP, then earlier step. Three-seed source ensemble recalibrates threshold. No Shenzhen validation or query labels used.',
        model='Independent sigmoid masked BCE for all arms; local changes only global mean query into per-school prototype+anchor query, identical parameter count.',
        weights='Equal property weighting with source complete single/multi mass balanced50/50; identical weights and training row draws across arms.',
        policy='Identical existing native policy and reviewed supplements; university zones base only. Raw outputs also reported.',
        scoring='Fixed common1095 queries; district/single/multi, unseen-combination diagnostics after freeze, corrected/harmed; paired district-stratified1km block bootstrap2000.',
        limitations=['Repeatedly explored development geography; no independent confirmation.',
            'Fixed learning rate and320step cap; no exhaustive optimization or claim of convergence.',
            'Old46.94percent uses different initialization/training/validation; report as contextual benchmark only.',
            'School-aware K1 acquisition is an existing Gold simulation, not fresh official verification.'],
        adopted=False, independent_validation=False)
    write(OUT / 'protocol.json', c)
    paths = [G.OUT/'public_population.json', G.OUT/'verified_partial_references.json', OUT/'k1_release.json']
    sources = {}; ys = {}; audits = {}; contexts = {}
    for d in c['sources']:
        x = E.pkl(M.OLD/'inputs'/f'{d}.pkl')
        y = dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'))
        sources[d], ys[d] = x, y
        arr = np.load(REPO/'data'/d/'arrays.npz')
        ext = np.load(REPO/'data'/d/'extra_evidence.npz')
        meta = read(REPO/'data'/d/'district.json')
        roadkey = meta.get('lines') or meta['city']
        contexts[d] = (arr['coords'], E.EV.lines(roadkey), ext['name_match'])
        eligible = np.flatnonzero(y['fold'][x.ref_sites] != 0)
        assert not set(x.groups[x.ref_sites[eligible]]) & set(x.groups[y['valid_ids']])
        audits[d] = dict(schools=x.school_n, full_reference_rows=len(eligible),
            represented_schools=len(set(x.ref_schools[eligible])), training_rows=len(y['train_ids']),
            validation_rows=len(y['valid_ids']), source_views=[])
        paths += [M.OLD/'inputs'/f'{d}.pkl', M.OLD/'inputs'/f'{d}_labels.npz',
                  REPO/'data'/d/'arrays.npz', REPO/'data'/d/'extra_evidence.npz',
                  REPO/'data'/d/'district.json', REPO/'data/lines'/f'{roadkey}.pkl']
    public = read(G.OUT/'public_population.json')
    partial = read(G.OUT/'verified_partial_references.json')
    targets = {}
    for d in c['targets']:
        data = G.build(d, release[d]['support'], partial[d], public[d]['all_groups'])
        x = data['x']; targets[d] = data
        labels = np.full((x.n, x.school_n), -1, np.float32)
        old = dict(np.load(M.OLD/'inputs'/f'{d}_labels.npz'))
        labels[old['labels'] == 1] = 1
        by = {s:i for i,s in enumerate(data['school_ids'])}
        for row in partial[d]: labels[row['q'], by[row['school']]] = 1
        for row in release[d]['support']: labels[row['q']] = data['labels'][row['q']]
        ids = np.flatnonzero((labels >= 0).any(1))
        ys[d] = dict(labels=labels, train_ids=ids)
        common = set(release[d]['common_ids'])
        q = np.array([r['q'] for r in release[d]['query'] if r['site_id'] in common],int)
        assert not set(x.groups[q]) & set(x.groups[ids])
        assert not set(x.groups[q]) & set(x.groups[x.ref_sites])
        data['query_ids'] = q
        E.savep(OUT/'target_inputs'/f'{d}.pkl',data)
        audits[d] = dict(new_verifications=len(release[d]['support']),query=len(q),
            complete_training_rows=int((labels[ids]>=0).all(1).sum()),
            partial_training_rows=int((labels[ids]<0).any(1).sum()),
            reference_rows=len(x.ref_sites), references_query_disjoint=True)
        paths += E.input_paths(d) + [M.OLD/'inputs'/f'{d}_labels.npz']
        paths += [Path(p) for p in E.ReviewedSupplements(d).hashes]
        paths += [E.ROOT/'round08_reference_retraining/policies'/f'{d}.json',M.OLD/'source_graphs'/f'{d}.json']
    assert sum(audits[d]['new_verifications'] for d in c['targets'])==84
    assert sum(audits[d]['query'] for d in c['targets'])==1095
    allx = {**sources, **{d:v['x'] for d,v in targets.items()}}
    total = sum(len(set(allx[d].groups[y['train_ids']])) for d,y in ys.items())
    weights = {}
    for d,y in ys.items():
        ids = y['train_ids']; _,inv,cnt = np.unique(allx[d].groups[ids],return_inverse=True,return_counts=True)
        w = np.zeros(allx[d].n); w[ids] = len(ys)/total/cnt[inv]; weights[d]=w
    weights, balance = balanced_weights({d:y['labels'] for d,y in ys.items()}, weights, c['sources'])
    for d,x in sources.items():
        y=ys[d]; eligible=np.flatnonzero(y['fold'][x.ref_sites]!=0)
        for view in ['full','validation']+[f'episode{i}' for i in range(c['episodes'])]:
            seed=c['episode_seed']+(10000 if view=='validation' else int(view[7:]) if view.startswith('episode') else 0)
            ix=choose_relations(x,eligible,seed,view!='full')
            v=rebuild_view(x,ix,*contexts[d])
            if view!='full': assert np.bincount(v.ref_schools,minlength=x.school_n).max(initial=0)<=1
            q=y['valid_ids'] if view=='validation' else y['train_ids']
            t=M.task(v,q,None if view=='validation' else y['labels'],None if view=='validation' else weights[d])
            if view=='validation': t['truth']=y['valid_labels']
            E.savep(OUT/'tasks'/d/f'{view}.pkl',t)
            audits[d]['source_views'].append(dict(view=view,relations=ix.tolist(),
                reference_rows=len(ix),zero_reference_schools=x.school_n-len(set(v.ref_schools))))
            log('BUILT',d,view,len(ix))
        del contexts[d]
    for d,data in targets.items():
        y=ys[d]; E.savep(OUT/'tasks'/d/'full.pkl',M.task(data['x'],y['train_ids'],y['labels'],weights[d]))
    write(OUT/'input_audit.json',dict(districts=audits,balance=balance,target_gold_read=False))
    code=[Path(__file__),REPO/'zoning/low_reference_training.py',REPO/'tests/test_low_reference_training.py',
          Path(G.__file__),Path(E.__file__),Path(M.__file__),REPO/'scripts/shenzhen_exact_calibration.py',
          REPO/'scripts/shenzhen_balanced_ranking.py']
    code += [REPO/'zoning'/f for f in ['modernnca_multilabel.py','modernnca_normalization.py','full_evidence.py','model.py','evidence.py']]
    code += [REPO/'scripts'/f for f in ['shenzhen_partial_model.py','shenzhen_reviewed_supplements.py','shenzhen_current_supplements.py','shenzhen_scope_implication.py']]
    code += [REPO/'third_party/TALENT/model/models/modernNCA.py']
    write(OUT/'code_hashes.json',{str(p.relative_to(REPO)):sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'input_hashes.json',{str(p):sha(p) for p in sorted(set(paths))})
    write(OUT/'freeze.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    log('PREPARED')


def verify():
    for p,h in read(OUT/'freeze.json').items(): assert sha(OUT/p)==h,p
    for p,h in read(OUT/'input_hashes.json').items(): assert sha(p)==h,p
    for p,h in read(OUT/'code_hashes.json').items(): assert sha(REPO/p)==h,p
    return read(OUT/'protocol.json')


def calibrate(probabilities, truths):
    result=exact_thresholds(probabilities,truths)
    v=result['selections']['balanced_exact']
    v['balanced']=.5*(v['single_correct']/result['source_single']+v['multi_correct']/result['source_multi'])
    return v


def rank(v): return (v['balanced'],v['correct'],-v['fp'],-v['step'])


def new_model(arm,seed,c):
    return (SchoolLocalModernNCA if arm.startswith('local') else MultiLabelModernNCA)(41,27,seed,c['dim']).to(E.DEVICE)


def decoded_sets(p,t):
    return [tuple(np.flatnonzero(row>=t)) if (row>=t).any() else (int(row.argmax()),) for row in p.astype(float)]


def fit():
    guard(); c=verify(); train={}; val={}
    for d in c['sources']+c['targets']:
        views=['full']+[f'episode{i}' for i in range(c['episodes'])] if d in c['sources'] else ['full']
        train[d]={v:E.pkl(OUT/'tasks'/d/f'{v}.pkl') for v in views}
        if d in c['sources']: val[d]=E.pkl(OUT/'tasks'/d/'validation.pkl')
    truths=[v['truth'] for v in val.values()]
    selection={}; allsource={}
    for arm in c['arms']:
        allsource[arm]=[]
        for seed in c['seeds']:
            dest=OUT/'weights'/arm/f'{seed}.pt'; report=OUT/'selection'/arm/f'{seed}.json'
            if dest.exists() and report.exists():
                selection[f'{arm}/{seed}']=read(report)
                allsource[arm].append(dict(np.load(OUT/'source_probabilities'/arm/f'{seed}.npz')))
                continue
            m=new_model(arm,seed,c);opt=torch.optim.Adam(m.parameters(),lr=c['lr'])
            rng=np.random.default_rng(seed);history=[];best=None;beststate=None
            for step in range(1,c['steps']+1):
                m.train();opt.zero_grad();reg=m.regularization(c['reg']);reg.backward();total=float(reg.detach())
                for d,views in train.items():
                    view=f'episode{(step-1)%c["episodes"]}' if arm.endswith('sparse') and d in c['sources'] else 'full'
                    t=views[view];ix=rng.choice(t['n'],min(c['batch'],t['n']),replace=False)
                    b=M.device_batch(t,ix);z=m(b)
                    loss=observed_bce_loss(z,b['labels'],b['weight'])*t['n']/len(ix)
                    assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c['clip']));opt.step()
                if step%20==0: log('TRAIN',arm,seed,step,round(total,4))
                if step in c['checkpoints']:
                    pp=[M.predict(m,t,'multilabel') for t in val.values()]
                    v=calibrate(pp,truths);v.update(step=step,loss=total,gradient_norm=norm);history.append(v)
                    if best is None or rank(v)>rank(best):
                        best=v;beststate={k:z.detach().cpu().clone() for k,z in m.state_dict().items()}
                    write(OUT/'progress.json',dict(arm=arm,seed=seed,history=history,best=best));log('VALID',arm,seed,v)
            dest.parent.mkdir(parents=True,exist_ok=True);torch.save(beststate,dest);m.load_state_dict(beststate)
            pp={d:M.predict(m,t,'multilabel') for d,t in val.items()}
            path=OUT/'source_probabilities'/arm/f'{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**pp)
            result=dict(best=best,history=history,weight_sha256=sha(dest),at_step_limit=best['step']==c['steps'])
            write(report,result);selection[f'{arm}/{seed}']=result;allsource[arm].append(pp)
    thresholds={}
    for arm in c['arms']:
        pp=[np.mean([v[d] for v in allsource[arm]],axis=0) for d in c['sources']]
        thresholds[arm]=calibrate(pp,truths)
    write(OUT/'training_selection.json',selection);write(OUT/'thresholds.json',thresholds)
    # Freeze every selection before any target prediction or evaluation.
    write(OUT/'selection_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['weights','selection','source_probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()} | {'thresholds.json':sha(OUT/'thresholds.json')})
    del train
    for d in c['targets']:
        data=E.pkl(OUT/'target_inputs'/f'{d}.pkl');q=data['query_ids'];t=M.task(data['x'],q)
        meta=read(E.input_paths(d)[3]);outputs={}
        for arm in c['arms']:
            pp=[]
            for seed in c['seeds']:
                m=new_model(arm,seed,c);m.load_state_dict(torch.load(OUT/'weights'/arm/f'{seed}.pt',weights_only=True,map_location=E.DEVICE))
                pp.append(M.predict(m,t,'multilabel'))
            p=np.mean(pp,axis=0);ss=decoded_sets(p,thresholds[arm]['threshold'])
            outputs[arm+'_raw']=[dict(district=d,site_id=meta['sites'][i]['coordinate_group_id'],building_name=meta['sites'][i]['building_name'],after=sorted(data['school_ids'][s] for s in schools)) for i,schools in zip(q,ss)]
            outputs[arm+'_policy']=E.decode(d,p,ss,data,q)
            path=OUT/'target_probabilities'/f'{d}_{arm}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,mean=p,seeds=np.stack(pp),query_ids=q)
        write(OUT/'predictions'/f'{d}.json',outputs)
    write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):sha(p) for folder in ['predictions','target_probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()})
    log('PREDICTIONS SEALED')


def score():
    c=verify()
    assert sha(E.GOLD)==read(G.OUT/'selection_freeze.json')['gold'], 'Frozen Gold changed'
    for name in ['selection_freeze.json','prediction_freeze.json']:
        for p,h in read(OUT/name).items(): assert sha(OUT/p)==h,p
    gold={(r['district'],r['site_id']):r for r in read(E.GOLD) if r['has_map_gold']}
    release=read(OUT/'k1_release.json');cases={}
    for d in c['targets']:
        plan=release[d];queries={r['site_id']:r for r in plan['query'] if r['site_id'] in set(plan['common_ids'])}
        combos={tuple(sorted(r['schools'])) for r in plan['support']}
        outputs=read(OUT/'predictions'/f'{d}.json')
        assert sha(G.OUT/'predictions/1'/f'{d}.json')==read(G.OUT/'prediction_freeze.json')[str(Path('predictions')/'1'/f'{d}.json')]
        old=read(G.OUT/'predictions/1'/f'{d}.json')
        outputs['archived_modern_tuned_policy']=[r for r in old['modern_tuned_policy'] if r['site_id'] in queries]
        outputs['archived_modern_frozen_policy']=[r for r in old['modern_frozen_policy'] if r['site_id'] in queries]
        for arm,pred in outputs.items():
            assert len(pred)==len(queries) and {r['site_id'] for r in pred}==set(queries)
            for r in sorted(pred,key=lambda v:v['site_id']):
                sid=r['site_id'];truth=set(gold[d,sid]['gold']);p=set(r['after']);xy=queries[sid]['xy']
                cases.setdefault(arm,[]).append(dict(district=d,site_id=sid,name=r['building_name'],truth=sorted(truth),prediction=sorted(p),correct=p==truth,
                    tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),size=len(truth),unseen=tuple(sorted(truth)) not in combos,
                    block=f'{d}/{int(xy[0]//1000)}/{int(xy[1]//1000)}'))
    filters={'all':lambda r:True,**{d:(lambda r,d=d:r['district']==d) for d in c['targets']},
             'single':lambda r:r['size']==1,'multi':lambda r:r['size']>1,'unseen':lambda r:r['unseen'],'seen':lambda r:not r['unseen']}
    metrics={a:{g:E.metric([r for r in rows if f(r)]) for g,f in filters.items()} for a,rows in cases.items()}
    names=list(cases);base=cases[names[0]];assert len(base)==1095
    for rows in cases.values(): assert [(r['district'],r['site_id']) for r in rows]==[(r['district'],r['site_id']) for r in base]
    mat=np.array([[r['correct'] for r in cases[a]] for a in names],int)
    blocks={d:sorted({r['block'] for r in base if r['district']==d}) for d in c['targets']}
    indexes={b:np.array([i for i,r in enumerate(base) if r['block']==b]) for bb in blocks.values() for b in bb}
    rng=np.random.default_rng(20261114);samples=[]
    for _ in range(2000):
        ix=np.concatenate([indexes[b] for d in c['targets'] for b in rng.choice(blocks[d],len(blocks[d]),replace=True)])
        samples.append(mat[:,ix].mean(1)*100)
    samples=np.array(samples);comparisons={}
    pairs=[('global_sparse_policy','global_full_policy'),('local_sparse_policy','local_full_policy'),
           ('local_full_policy','global_full_policy'),('local_sparse_policy','global_sparse_policy')]
    pairs += [(a+'_policy','archived_modern_tuned_policy') for a in c['arms']]
    for a,b in pairs:
        ia,ib=names.index(a),names.index(b);diff=mat[ia]-mat[ib]
        comparisons[a+' minus '+b]=dict(delta_pp=float(diff.mean()*100),corrected=int((diff==1).sum()),harmed=int((diff==-1).sum()),
            ci95=np.quantile(samples[:,ia]-samples[:,ib],[.025,.975]).tolist(),
            ci98_75=np.quantile(samples[:,ia]-samples[:,ib],[.00625,.99375]).tolist(),
            matched_training=not b.startswith('archived'),
            single_harmed=int(sum(r['size']==1 and diff[i]==-1 for i,r in enumerate(base))))
    for a,rows in cases.items(): write(OUT/'scored_rows'/f'{a}.json',rows)
    write(OUT/'results.json',dict(metrics=metrics,comparisons=comparisons,adopted=False,independent_validation=False))
    log('SCORED',{a:round(v['all']['exact'],2) for a,v in metrics.items()})


def audit():
    c=verify();res=read(OUT/'results.json');release=read(OUT/'k1_release.json')
    for file in ['selection_freeze.json','prediction_freeze.json']:
        for p,h in read(OUT/file).items(): assert sha(OUT/p)==h
    for arm,metrics in res['metrics'].items():
        rows=read(OUT/'scored_rows'/f'{arm}.json')
        assert len(rows)==1095 and len({(r['district'],r['site_id']) for r in rows})==1095
        assert sum(set(r['prediction'])==set(r['truth']) for r in rows)==metrics['all']['correct']
        assert sum(len(set(r['prediction'])-set(r['truth'])) for r in rows)==metrics['all']['fp']
        assert sum(len(set(r['truth'])-set(r['prediction'])) for r in rows)==metrics['all']['fn']
    for d in c['targets']:
        data=E.pkl(OUT/'target_inputs'/f'{d}.pkl');x=data['x'];q=data['query_ids']
        assert not set(x.groups[q]) & set(x.groups[x.ref_sites])
        assert len(release[d]['support']) in [33,51]
        pred=read(OUT/'predictions'/f'{d}.json')
        for arm in c['arms']:
            probs=dict(np.load(OUT/'target_probabilities'/f'{d}_{arm}.npz'))
            np.testing.assert_array_equal(probs['mean'],probs['seeds'].mean(0))
            threshold=read(OUT/'thresholds.json')[arm]['threshold']
            # Independent decoder and original policy replay from sealed probabilities.
            sets=[]
            for row in probs['mean']:
                ss=[i for i,v in enumerate(row) if float(v)>=threshold]
                sets.append(tuple(ss or [int(np.argmax(row))]))
            expected=[sorted(data['school_ids'][s] for s in ss) for ss in sets]
            assert expected==[r['after'] for r in pred[arm+'_raw']]
            replay=E.decode(d,probs['mean'],sets,data,q)
            assert replay==pred[arm+'_policy']
    write(OUT/'independent_audit.json',dict(passed=True,metrics_recounted=True,raw_and_policy_replayed=4*1095,
        target_new_labels=84,query_count=1095,reference_query_disjoint=True,prediction_and_selection_hashes_verified=True))
    log('AUDIT PASSED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','fit','score','audit']);a=p.parse_args();globals()[a.phase]()
