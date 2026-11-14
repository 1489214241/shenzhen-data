"""All-school membership pilot: reuse Modern evidence, test tree fusion.

No candidate truncation, mandatory Modern top1, school identity feature, new
target labels, or production change. Source-selected thresholds only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import collections
import copy
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from contextlib import nullcontext
import experiment_conditional_admission_hybrid as H
from shenzhen_exact_calibration import exact_thresholds, threshold_metrics

A=H.A; E=A.E; M=A.M
read,write,sha=A.read,A.write,A.sha
REPO=M.REPO; INPUT=E.INPUT
OUT=REPO/'outputs/shenzhen_membership_trees_20261114'
FEATURES=['mean','local']
MODES=['balanced','protected']


def check(name):
    for p,h in read(OUT/name).items():assert sha(p)==h,p


def local_features(x, chunk=32):
    """Two nearest LEGAL references per school, all-school distance/count info.

Keeps original features; does not infer administrative identity from a point.
Every source/target school receives the same transformation, including no-ref.
"""
    out=[];nr=len(x.ref_sites);width=x.pair.shape[-1]
    for lo in range(0,x.n,chunk):
        hi=min(lo+chunk,x.n);b=hi-lo
        nearest=np.zeros((b,x.school_n,2*width+8),np.float32)
        nearest[:,:,2*width:2*width+2]=15.
        d1=np.full((b,x.school_n),15000.,np.float32)
        for s in range(x.school_n):
            cols=np.flatnonzero(x.ref_schools==s)
            if not len(cols):continue
            distances=np.where(~x.self_mask[lo:hi,cols],x.pair[lo:hi,cols,0]*300,np.inf)
            order=np.argsort(distances,axis=1,kind='stable')
            sorted_d=np.take_along_axis(distances,order,axis=1)
            for k in range(min(2,len(cols))):
                valid=np.isfinite(sorted_d[:,k]);q=np.arange(b)
                value=x.pair[lo+q,cols[order[:,k]]]
                nearest[:,s,k*width:(k+1)*width]=np.where(valid[:,None],value,0)
            d1[:,s]=np.minimum(sorted_d[:,0],15000.)
            d2=np.minimum(sorted_d[:,1],15000.) if len(cols)>1 else np.full(b,15000.)
            nearest[:,s,2*width:2*width+7]=np.stack([
                np.minimum(sorted_d[:,0],15000.)/1000,d2/1000,
                np.log1p((distances<=200).sum(1)),np.log1p((distances<=500).sum(1)),
                np.log1p((distances<=1000).sum(1)),np.isfinite(sorted_d[:,0]),
                np.isfinite(sorted_d[:,1]) if len(cols)>1 else np.zeros(b)],-1)
        nearest[:,:,-1]=np.argsort(np.argsort(d1,axis=1,kind='stable'),axis=1,kind='stable')/max(x.school_n-1,1)
        assert nearest.shape==(b,x.school_n,2*width+8) and np.isfinite(nearest).all()
        out.append(nearest)
    return np.concatenate(out)


def observed_training(feature, labels, weights, ids):
    y=labels[ids];valid=(y>=0)&(weights[ids,None]>0)
    w=np.broadcast_to(weights[ids,None]/np.maximum((y>=0).sum(1,keepdims=True),1),y.shape)
    return feature[ids][valid],y[valid].astype(np.uint8),w[valid]


def prepare():
    M.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    OUT.mkdir(parents=True)
    H.install();c=A.verify()
    protocol=dict(sources=c['sources'],targets=c['targets'],features=FEATURES,modes=MODES,
        primary='local_balanced_legacy minus baseline_current',
        mechanism='All-school independent membership; model can correct original top1. Compare mean-reference summary vs added nearest two per-school legal reference evidence.',
        model='sklearn HistGradientBoostingClassifier; engineering pilot using existing project GBT family, not an XGBoost reproduction or novel architecture claim.',
        hp=dict(learning_rate=.05,max_leaf_nodes=31,max_iter=300,l2_regularization=1.,early_stopping=False,random_state=20261114),
        candidate_scope='Every school; no top10/topk school filtering or known answer count; no school or site ID feature.',
        supervision='Same original full/partial source labels and verified Shenzhen partial labels as routed snapshot. Restore native top1 as an ordinary supervised candidate. Unknown -1 excluded, not negative. Same teacher per-property weights, mean over observed labels.',
        reference_scope='All existing legal references retained; mean arm reuses139 features; local adds closest two reference feature vectors per school, distances/counts/availability/rank, same query-property exclusion.',
        threshold='Source1246 rows only: balanced exact averages singleton and multi exact; protected retains native source singleton accuracy within0.5pp where possible. Threshold and mode fixed before target scores.',
        protected_failure='If no protected candidate exists, retain source-best micro threshold for diagnosis, flag failure; never relabel fallback as passing.',
        decode='Same legacy policy postprocessing as Modern applied to newly selected roots; verified-only ablation also reported. University member-list expansion forbidden. Hard text exclusions/positives enforced.',
        modern='Frozen three-teacher ensemble features; one deterministic tree fit per feature arm. No new Modern training or independent model-seed validation.',
        source_scores='Training-source Modern scores can be in-sample; source validation remains held out from tree fit. This is a pilot, not out-of-fold stacking confirmation.',
        evaluation='All1802groups output;1761scored;41output-only;242semantic-pending retained. Exact sets,FP,FN,precision,recall,corrected/harmed,1km bootstrap3000.',
        target_gold_read=False,target_polygon_read=False,prior_target_results_seen=True,
        new_verified_reference_budget=0,adopted=False,independent_validation=False)
    write(OUT/'protocol.json',protocol)
    paths=[Path(__file__).resolve(),OUT/'protocol.json',REPO/'scripts/shenzhen_exact_calibration.py',
        H.OUT/'freeze.json',H.OUT/'prediction_freeze.json',INPUT/'manifest.json']
    inv={}
    for d in c['sources']+c['targets']:
        p=H.OUT/'cache/routed'/f'{d}.npz';xpath=INPUT/d/'routed.pkl'
        cache=dict(np.load(p));x=M.pkl(xpath)
        original=dict(np.load(INPUT/d/'routed_labels.npz'))
        np.testing.assert_array_equal(cache['original_labels'],original['labels'])
        assert not set(original['train_ids'])&set(original['valid_ids'])
        w=np.load(E.TEACHER/f'{d}_weights.npy').copy()
        if d in c['targets']:
            unit=w.max();w[:]=0;ids=original['train_ids']
            _,inverse,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True)
            w[ids]=unit/count[inverse]
        assert (x.groups[:,None]==x.groups[x.ref_sites][None]).astype(bool).shape==x.self_mask.shape
        np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        local=local_features(x)
        dest=OUT/'cache'/f'{d}.npz';dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,feature=cache['feature'],local=local,prob=cache['prob'],
            labels=original['labels'],train_ids=original['train_ids'],valid_ids=original['valid_ids'],
            valid_labels=original['valid_labels'],weight=w)
        y=original['labels'][original['train_ids']]
        inv[d]=dict(sites=x.n,schools=x.school_n,references=len(x.ref_sites),
            reference_sites=len(set(x.ref_sites)),schools_with_reference=len(set(x.ref_schools)),
            train_rows=len(y),observed_positive=int((y==1).sum()),observed_negative=int((y==0).sum()),
            complete_training_set_sizes=dict(collections.Counter(map(int,(y[(y>=0).all(1)]==1).sum(1)))),
            validation_set_sizes=dict(collections.Counter(map(int,original['valid_labels'].sum(1)))),
            mean_dim=cache['feature'].shape[-1],local_extra_dim=local.shape[-1])
        paths += [p,xpath,INPUT/d/'routed_labels.npz',E.TEACHER/f'{d}_weights.npy',dest]
        if d in c['targets']:
            paths += [INPUT/d/f for f in ['metadata.json','reviewed_policy.json','constraints.json']]
            paths += [M.ROOT/'round08_reference_retraining/policies'/f'{d}.json',M.OLD/'source_graphs'/f'{d}.json',
                H.OUT/'predictions/routed_bce_legacy/ensemble'/f'{d}.json']
        print('PREPARED',d,inv[d],flush=True)
        del x,cache,local
    write(OUT/'input_inventory.json',inv);paths.append(OUT/'input_inventory.json')
    write(OUT/'inputs_sha256.json',{str(p):sha(p) for p in paths})
    print('INPUT AND PROTOCOL SEALED',flush=True)


def load(c,arm):
    xs={}
    for d in c['sources']+c['targets']:
        x=dict(np.load(OUT/'cache'/f'{d}.npz'))
        if arm=='local':x['feature']=np.concatenate([x['feature'],x['local']],-1)
        del x['local'];xs[d]=x
    return xs


def calibrate(c,pp,xs):
    probs=[pp[d][xs[d]['valid_ids']] for d in c['sources']]
    labels=[xs[d]['valid_labels'] for d in c['sources']]
    cal=exact_thresholds(probs,labels)
    native=sum(int((xs[d]['prob'][xs[d]['valid_ids']].argmax(1)[y.sum(1)==1]==y.argmax(1)[y.sum(1)==1]).sum())
               for d,y in zip(c['sources'],labels))
    floor=native-.005*cal['source_single']
    eligible=[r for r in cal['candidates'] if r['single_correct']>=floor]
    best=max(eligible or cal['candidates'],key=lambda r:(r['single_correct']+r['multi_correct'],r['multi_correct'],r['threshold']))
    result={mode:dict(threshold_metrics(probs,labels,t),threshold=t) for mode,t in
        [('balanced',cal['selections']['balanced_exact']['threshold']),('protected',best['threshold'])]}
    result['protected'].update(passed=bool(eligible),native_single=native,floor=floor)
    result['balanced']['passed_single_protection']=result['balanced']['single_correct']>=floor
    return result


def fit():
    M.guard();check('inputs_sha256.json');c=read(OUT/'protocol.json')
    if (OUT/'weights_freeze.json').exists():raise FileExistsError('Preserve fits')
    calibration={}
    for arm in FEATURES:
        xs=load(c,arm);rows=[observed_training(x['feature'],x['labels'],x['weight'],x['train_ids']) for x in xs.values()]
        xx=np.concatenate([a for a,b,w in rows]);yy=np.concatenate([b for a,b,w in rows]);ww=np.concatenate([w for a,b,w in rows])
        ww=ww/ww.mean();assert set(np.unique(yy))=={0,1}
        print('FIT',arm,xx.shape,dict(collections.Counter(yy.tolist())),flush=True)
        model=HistGradientBoostingClassifier(**c['hp'])
        with nullcontext():model.fit(xx,yy,sample_weight=ww)
        del rows,xx,yy,ww
        dest=OUT/'weights'/f'{arm}.pkl';dest.parent.mkdir(exist_ok=True);dest.write_bytes(pickle.dumps(model))
        pp={}
        with nullcontext():
            for d in c['sources']:
                x=xs[d];f=x['feature'];pp[d]=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2])
        calibration[arm]=calibrate(c,pp,xs)
        dest=OUT/'source_probabilities'/f'{arm}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**pp)
        print('SOURCE',arm,calibration[arm],flush=True)
        del xs,model
    write(OUT/'source_calibration.json',calibration)
    files=[OUT/'source_calibration.json']+list((OUT/'weights').glob('*'))+list((OUT/'source_probabilities').glob('*'))
    write(OUT/'weights_freeze.json',{str(p):sha(p) for p in files})


def selected(scores,threshold):
    result=scores>=threshold;empty=~result.any(1)
    result[empty,scores[empty].argmax(1)]=True
    return result


def infer():
    M.guard();check('inputs_sha256.json');check('weights_freeze.json');c=read(OUT/'protocol.json')
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve predictions')
    calibration=read(OUT/'source_calibration.json')
    for arm in FEATURES:
        model=pickle.loads((OUT/'weights'/f'{arm}.pkl').read_bytes());payload={}
        for d in c['targets']:
            x=dict(np.load(OUT/'cache'/f'{d}.npz'));f=x['feature']
            if arm=='local':f=np.concatenate([f,x['local']],-1)
            with nullcontext():q=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2])
            payload[d]=q
            meta=read(INPUT/d/'metadata.json');cons=read(INPUT/d/'constraints.json');ids=[s['school_id'] for s in meta['schools']]
            policy=read(M.ROOT/'round08_reference_retraining/policies'/f'{d}.json');graph=read(M.OLD/'source_graphs'/f'{d}.json')
            for mode in MODES:
                yes=selected(q,calibration[arm][mode]['threshold']);rr={k:[] for k in ['raw','verified','legacy']}
                for i,site in enumerate(meta['sites']):
                    raw=sorted(ids[j] for j in np.flatnonzero(yes[i]));base=dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],complete_school_set=False)
                    rr['raw'].append(dict(base,after=raw));con=cons[base['site_id']]
                    rr['verified'].append(dict(base,after=E.apply_verified_constraints(raw,con['positive'],con['negative'])))
                    legal=np.where(yes[i],1+q[i],q[i]);known=[ids[j] for j in meta['known'].get(str(i),[])]
                    result=E.predict_sets(legal,1.,ids,policy,graph,site,known)
                    rr['legacy'].append(dict(base,after=result['predicted_school_ids'],decision=result))
                rr['legacy']=E.ReviewedSupplements(d).apply(rr['legacy'])
                for decode,rows in rr.items():write(OUT/'predictions'/f'{arm}_{mode}_{decode}'/f'{d}.json',rows)
        dest=OUT/'target_probabilities'/f'{arm}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**payload)
    files=list((OUT/'predictions').rglob('*.json'))+list((OUT/'target_probabilities').glob('*'))
    write(OUT/'prediction_freeze.json',{str(p):sha(p) for p in files})
    print('ALL TARGET PREDICTIONS SEALED',flush=True)


def score():
    check('inputs_sha256.json');check('weights_freeze.json');check('prediction_freeze.json')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scores')
    c=read(OUT/'protocol.json');goldpath=REPO/'outputs/modernnca_partial_sets_20261113/scored_rows/categorical_policy.json'
    truth={(r['d'],r['site_id']):r for r in read(goldpath)}
    hashes=read(REPO/'outputs/modernnca_partial_sets_20261113/results.json')['gold_sha256']
    for d,h in hashes.items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    filters={'all':lambda r:True,'single':lambda r:r['size']==1,'multi':lambda r:r['size']>1,
        'three_plus':lambda r:r['size']>=3,'semantic_pending':lambda r:r['semantic_pending'],
        **{d:lambda r,d=d:r['d']==d for d in c['targets']}}
    metrics={};allrows={};changes={}
    for arm in ['baseline_current']+sorted(p.name for p in (OUT/'predictions').iterdir()):
        rows=[];outside=0
        for d in c['targets']:
            path=H.OUT/'predictions/routed_bce_legacy/ensemble'/f'{d}.json' if arm=='baseline_current' else OUT/'predictions'/arm/f'{d}.json'
            preds=read(path);meta=read(INPUT/d/'metadata.json')
            assert len(preds)==len(meta['sites']) and {p['site_id'] for p in preds}=={s['coordinate_group_id'] for s in meta['sites']}
            for p in preds:
                key=d,p['site_id']
                if key not in truth:outside+=1;continue
                g=truth[key];a=set(p['after']);t=set(g['truth'])
                rows.append(dict(g,prediction=sorted(a),correct=a==t,tp=len(a&t),fp=len(a-t),fn=len(t-a)))
        assert len(rows)==1761 and outside==41 and sum(r['semantic_pending'] for r in rows)==242
        metrics[arm]={k:M.metric([r for r in rows if f(r)]) for k,f in filters.items()};allrows[arm]=rows
        write(OUT/'scored_rows'/f'{arm}.json',rows)
    comparisons={};before={(r['d'],r['site_id']):r for r in allrows['baseline_current']}
    for arm,rows in allrows.items():
        if arm=='baseline_current':continue
        comparisons[arm]={}
        for k,f in filters.items():
            rr=[r for r in rows if f(r)];delta=[int(r['correct'])-int(before[r['d'],r['site_id']]['correct']) for r in rr]
            comparisons[arm][k]=dict(corrected=delta.count(1),harmed=delta.count(-1),delta_pp=100*float(np.mean(delta)),ci95=M.evaluate.bootstrap(rr,delta,20261104,3000))
        changes[arm]=[dict(r,before=before[r['d'],r['site_id']]['prediction'],before_correct=before[r['d'],r['site_id']]['correct']) for r in rows if r['prediction']!=before[r['d'],r['site_id']]['prediction']]
    write(OUT/'changes_diagnostic_only.json',changes)
    write(OUT/'results.json',dict(metrics=metrics,comparisons=comparisons,source=read(OUT/'source_calibration.json'),
        gold_sha256=hashes,scoring_rows_sha256=sha(goldpath),primary=c['primary'],adopted=False,independent_validation=False))
    print('RESULTS',{a:m['all']['correct'] for a,m in metrics.items()},flush=True)


def audit():
    check('inputs_sha256.json');check('weights_freeze.json');check('prediction_freeze.json');c=read(OUT/'protocol.json')
    checks=0
    for arm in FEATURES:
        model=pickle.loads((OUT/'weights'/f'{arm}.pkl').read_bytes());xs=load(c,arm);pp={}
        for d,x in xs.items():
            f=x['feature']
            with nullcontext():p=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2])
            parent='source_probabilities' if d in c['sources'] else 'target_probabilities'
            np.testing.assert_array_equal(p,np.load(OUT/parent/f'{arm}.npz')[d]);pp[d]=p;checks+=1
        assert calibrate(c,pp,xs)==read(OUT/'source_calibration.json')[arm]
    r=read(OUT/'results.json');baseline={(x['d'],x['site_id']):x for x in read(OUT/'scored_rows/baseline_current.json')}
    for arm,metrics in r['metrics'].items():
        rows=read(OUT/'scored_rows'/f'{arm}.json');assert len(rows)==1761
        assert sum(set(x['prediction'])==set(x['truth']) for x in rows)==metrics['all']['correct']
        if arm!='baseline_current':
            delta=[int(x['correct'])-int(baseline[x['d'],x['site_id']]['correct']) for x in rows]
            assert delta.count(1)==r['comparisons'][arm]['all']['corrected']
            assert delta.count(-1)==r['comparisons'][arm]['all']['harmed']
    for d,h in r['gold_sha256'].items():assert sha(M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    write(OUT/'audit.json',dict(passed=True,probability_arrays_replayed=checks,source_calibration_replayed=True,gold_unchanged=True,adopted=False))
    print('AUDIT PASSED',checks,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','fit','infer','score','audit'])
    globals()[ap.parse_args().phase]()
