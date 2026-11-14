"""Development diagnostic: can a different classifier recover source multi sets?

Two fixed classifiers, exact same frozen observed labels and row-weight ratios.
Source calibration only. No Shenzhen evaluation or target predictions in this script.
This changes representation, architecture AND objective versus the neural parent;
it is a capacity probe, not an isolated attribution or a matched compute claim.
"""
import os
os.environ['OMP_NUM_THREADS']='6'
os.environ['MKL_NUM_THREADS']='6'
os.environ['OPENBLAS_NUM_THREADS']='6'
import argparse
from pathlib import Path
import pickle
import shutil
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
import experiment_shenzhen_reference_retraining as R
from shenzhen_relation_probe import school_vectors
from shenzhen_exact_calibration import exact_thresholds, calibration_intervals, threshold_metrics

F=R.F
OUT=R.ROOT/'round11_relation_capacity_probe'
NEW=F.REPO/'data/深圳市/居民小组修复模型输入_20261112'
ARM='new_references_labels'


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen probe')
    c=R.verify();R.check(NEW,'manifest.json');R.check(R.OUT,'probability_freeze.json')
    xs,ys=R.load(c,ARM)
    for d in R.TARGETS:
        with (NEW/d/'inputs.pkl').open('rb') as handle:xs[d]=pickle.load(handle)
    protocol=dict(sources=c['sources'],targets=R.TARGETS,arms=['anchor_context','pooled_relations'],
        parameters=dict(loss='log_loss',learning_rate=.05,max_iter=200,max_leaf_nodes=15,
            min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20261131),
        sklearn_version=sklearn.__version__,features={'anchor_context':33,'pooled_relations':159},
        source_split='Identical frozen round03 spatial validation; no new validation claim.',
        supervision='Identical round08 observed complete source and partial target cells; unknown cells omitted.',
        weights='Exact round08 per-property weights copied per observed school cell, globally rescaled to mean1.',
        missing_references='NaN pooled values plus zero mass; never interpreted as observed negatives.',
        objective_change='Binary log loss, without neural LSEP term; not a compute-matched neural comparison.',
        source_gate=dict(min_multi_correct=8,min_single_correct=901,min_attainable_multi=18,
            objective='balanced_exact',purpose='Only determines whether to proceed to a separately frozen target test.'),
        target_gold_read=False,target_polygon_read=False,target_predictions_generated=False,
        independent_validation=False,adopted=False,
        research=[dict(url='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html',
            reviewed='Public estimator documentation: missing-value routing, sample weights, fixed iterations without early stopping.'),
            dict(url='https://arxiv.org/abs/1703.06114',reviewed='Abstract only: permutation-invariant set functions. This is deterministic pooling plus trees, not a Deep Sets reproduction.')])
    F.write(OUT/'protocol.json',protocol)
    train=[];labels=[];weights=[];audit={};offset=0
    for d,x in xs.items():
        y=ys[d];ids=y['train_ids'];feature=school_vectors(x,ids);mask=y['labels'][ids]>=0
        base=np.load(R.OUT/'weights'/f'{ARM}_{d}.npz')['weights'][ids]
        assert np.all(base>0)
        matrix=feature[mask];truth=y['labels'][ids][mask].astype(np.int8)
        weight=np.broadcast_to(base[:,None],mask.shape)[mask]
        train.append(matrix);labels.append(truth);weights.append(weight)
        info=dict(rows=len(ids),observed_cells=len(truth),positive=int((truth==1).sum()),negative=int((truth==0).sum()),
            cell_start=offset,cell_end=offset+len(truth),weight_sum=float(weight.sum()),
            query_self_references_masked=bool(np.array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])))
        assert info['query_self_references_masked'];offset+=len(truth)
        if d in R.TARGETS:assert info['negative']==0 and len(y['valid_ids'])==0
        else:
            vi=y['valid_ids'];vf=school_vectors(x,vi)
            assert not set(x.groups[ids])&set(x.groups[vi])
            dest=OUT/'valid'/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,features=vf,labels=y['valid_labels'],ids=vi)
            multi_ids=ids[(y['labels'][ids]>=0).all(1)&((y['labels'][ids]==1).sum(1)>1)]
            dest=OUT/'train_multi'/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,features=school_vectors(x,multi_ids),labels=y['labels'][multi_ids].astype(bool),ids=multi_ids)
            info['valid_rows']=len(vi);info['train_complete_multi']=len(multi_ids)
        audit[d]=info;print('PREPARE',d,info,flush=True)
    matrix=np.concatenate(train);truth=np.concatenate(labels);weight=np.concatenate(weights)
    normalizer=float(weight.mean());weight/=normalizer
    np.save(OUT/'train_features.npy',matrix);np.savez_compressed(OUT/'train_labels_weights.npz',labels=truth,weights=weight)
    F.write(OUT/'input_audit.json',dict(districts=audit,features=matrix.shape[1],observed_cells=len(truth),
        weight_normalizer=normalizer,unknown_cells_used_as_negative=0,school_identifiers_used_as_features=False))
    links=dict(F.read(R.OUT/'input_links.json')['files'])
    for folder,manifest in [(NEW,'manifest.json'),(R.OUT,'input_freeze.json'),(R.OUT,'probability_freeze.json')]:
        links[str(folder/manifest)]=F.sha(folder/manifest)
        for rel,h in F.read(folder/manifest)['files'].items():links[str(folder/rel)]=h
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_relation_probe.py',F.REPO/'tests/test_shenzhen_relation_probe.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():raise FileExistsError('Preserve fits')
    x=np.load(OUT/'train_features.npy',mmap_mode='r');data=dict(np.load(OUT/'train_labels_weights.npz'))
    history={}
    for arm in c['arms']:
        width=c['features'][arm];model=HistGradientBoostingClassifier(**c['parameters'])
        model.fit(x[:,:width],data['labels'],sample_weight=data['weights'])
        assert model.n_iter_==200
        dest=OUT/'models'/f'{arm}.pkl';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('wb') as handle:pickle.dump(model,handle,protocol=5)
        payload={}
        for d in c['sources']:
            for split in ['valid','train_multi']:
                v=dict(np.load(OUT/split/f'{d}.npz'));a=v['features']
                p=model.predict_proba(a.reshape(-1,a.shape[-1])[:,:width])[:,1].reshape(a.shape[:2]) if len(a) else np.zeros(a.shape[:2])
                assert np.isfinite(p).all();payload[d+'_'+split]=p
        dest=OUT/'probabilities'/f'{arm}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**payload)
        history[arm]=dict(iterations=int(model.n_iter_),features=width,model_sha256=F.sha(OUT/'models'/f'{arm}.pkl'))
        F.write(OUT/'training_history.json',history);print('FIT_COMPLETE',arm,flush=True)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for folder in ['models','probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve report')
    truth=[np.load(OUT/'valid'/f'{d}.npz')['labels'] for d in c['sources']]
    train_truth=[np.load(OUT/'train_multi'/f'{d}.npz')['labels'] for d in c['sources']]
    seeds=F.read(R.OUT/'protocol.json')['seeds']
    old=[dict(np.load(R.OUT/'probabilities'/ARM/f'{s}.npz')) for s in seeds]
    baselines=[np.stack([p[d] for p in old]).mean(0) for d in c['sources']]
    baseline_cal=exact_thresholds(baselines,truth)
    report={'baseline':dict(selections=baseline_cal['selections'],attainable_multi=sum(k>1 for lo,hi,k in calibration_intervals(baselines,truth)[0]))}
    for arm in c['arms']:
        payload=dict(np.load(OUT/'probabilities'/f'{arm}.npz'))
        probs=[payload[d+'_valid'] for d in c['sources']];cal=exact_thresholds(probs,truth)
        attainable=sum(k>1 for lo,hi,k in calibration_intervals(probs,truth)[0]);chosen=cal['selections']['balanced_exact']
        gate=c['source_gate'];passed=chosen['multi_correct']>=gate['min_multi_correct'] and chosen['single_correct']>=gate['min_single_correct'] and attainable>=gate['min_attainable_multi']
        report[arm]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
            train_multi=threshold_metrics([payload[d+'_train_multi'] for d in c['sources']],train_truth,chosen['threshold']),
            districts={d:threshold_metrics([p],[y],chosen['threshold']) for d,p,y in zip(c['sources'],probs,truth)})
        F.write(OUT/'calibration'/f'{arm}.json',cal)
        print(arm,report[arm],flush=True)
    F.write(OUT/'source_report.json',dict(reports=report,adopted=False,target_scored=False,
        next_action='Only a source-gate passing arm can enter the separately frozen target evaluation.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','report'])
    globals()[parser.parse_args().phase]()
