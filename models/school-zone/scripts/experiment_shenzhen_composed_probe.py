"""Matched relation-classifier ablation: add only three composed-scope channels.

Control is the exactly reused round11 pooled classifier. Verify all observed
training relation features remain identical under the current repaired inputs.
No Shenzhen predictions or scores until the same predeclared source gate passes.
"""
import os
os.environ['OMP_NUM_THREADS']='6';os.environ['MKL_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='6'
import argparse
from pathlib import Path
import pickle
import shutil
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import experiment_shenzhen_relation_probe as P
import build_shenzhen_composed_feature_inputs as C
from shenzhen_relation_probe import school_vectors
from shenzhen_exact_calibration import exact_thresholds,calibration_intervals,threshold_metrics

F=P.F;R=P.R
OUT=R.ROOT/'round14_composed_scope_probe'


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source probe')
    parent=P.verify();R.check(P.OUT,'probability_freeze.json');R.check(C.OUT,'manifest.json')
    for p,h in F.read(C.OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    rc=R.verify();xs,ys=R.load(rc,P.ARM)
    for d in R.TARGETS:
        with (C.TARGET/d/'inputs.pkl').open('rb') as handle:xs[d]=pickle.load(handle)
    control=np.load(P.OUT/'train_features.npy',mmap_mode='r');data=dict(np.load(P.OUT/'train_labels_weights.npz'))
    bounds=F.read(P.OUT/'input_audit.json')['districts'];extra=[];checks={}
    for d,x in xs.items():
        y=ys[d];ids=y['train_ids'];known=y['labels'][ids]>=0
        actual=school_vectors(x,ids)[known]
        info=bounds[d];sl=slice(info['cell_start'],info['cell_end'])
        np.testing.assert_array_equal(actual,control[sl])
        np.testing.assert_array_equal(y['labels'][ids][known],data['labels'][sl])
        scope=dict(np.load(C.OUT/d/'composed_scope.npz'))
        added=scope['features'][ids][known];extra.append(added)
        checks[d]=dict(old_training_features_bitwise_equal=True,labels_bitwise_equal=True,
            observed_cells=len(added),new_feature_counts=added.sum(0).astype(int).tolist())
        for split in ['valid','train_multi'] if d in rc['sources'] else []:
            prev=dict(np.load(P.OUT/split/f'{d}.npz'));ids=prev['ids']
            arr=np.concatenate([prev['features'],scope['features'][ids]],axis=-1)
            dest=OUT/split/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,features=arr,labels=prev['labels'],ids=ids)
        print('FEATURES_VERIFIED',d,checks[d],flush=True)
    F.write(OUT/'protocol.json',dict(sources=rc['sources'],targets=R.TARGETS,
        arms=['pooled_control','plus_composed_scope'],parameters=parent['parameters'],
        control='Exact round11 pooled_relations model, unchanged observed training features and labels verified.',
        intervention='Add three categorical complete-geographic-scope channels; all previous159channels unchanged.',
        native_rules='Retain K1-K8/X2; nonarea alternatives prevent negative full-scope assumptions.',
        source_gate=parent['source_gate'],features={'pooled_control':159,'plus_composed_scope':162},
        independent_validation=False,adopted=False,target_gold_read=False,target_polygon_read=False,
        source_holdout='Same1246development validation rows;60multi,1186single. No new independence claim.',
        partial_labels='Same positive-only targets, no unknown-as-negative and no new geographic training labels.'))
    np.save(OUT/'train_features.npy',np.concatenate([control,np.concatenate(extra)],axis=1))
    shutil.copyfile(P.OUT/'train_labels_weights.npz',OUT/'train_labels_weights.npz')
    F.write(OUT/'feature_audit.json',checks)
    links={}
    for folder,manifest in [(P.OUT,'input_freeze.json'),(P.OUT,'probability_freeze.json'),(C.OUT,'manifest.json')]:
        links[str(folder/manifest)]=F.sha(folder/manifest)
        for rel,h in F.read(folder/manifest)['files'].items():links[str(folder/rel)]=h
    links.update(F.read(C.OUT/'input_links.json')['files'])
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_composed_features.py',F.REPO/'tests/test_shenzhen_composed_features.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():raise FileExistsError('Preserve completed fit')
    model=HistGradientBoostingClassifier(**c['parameters'])
    X=np.load(OUT/'train_features.npy',mmap_mode='r');data=dict(np.load(OUT/'train_labels_weights.npz'))
    model.fit(X,data['labels'],sample_weight=data['weights']);assert model.n_iter_==200
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    with (OUT/'models/plus_composed_scope.pkl').open('wb') as handle:pickle.dump(model,handle,protocol=5)
    shutil.copyfile(P.OUT/'models/pooled_relations.pkl',OUT/'models/pooled_control.pkl')
    (OUT/'probabilities').mkdir(parents=True,exist_ok=True)
    payload={}
    for d in c['sources']:
        for split in ['valid','train_multi']:
            a=np.load(OUT/split/f'{d}.npz')['features']
            payload[d+'_'+split]=model.predict_proba(a.reshape(-1,162))[:,1].reshape(a.shape[:2]) if len(a) else np.zeros(a.shape[:2])
            assert np.isfinite(payload[d+'_'+split]).all()
    np.savez_compressed(OUT/'probabilities/plus_composed_scope.npz',**payload)
    shutil.copyfile(P.OUT/'probabilities/pooled_relations.npz',OUT/'probabilities/pooled_control.npz')
    F.write(OUT/'training_audit.json',dict(new_models=1,reused_control_models=1,iterations=int(model.n_iter_),
        training_rows=len(data['labels']),labels_weights_unchanged=F.sha(OUT/'train_labels_weights.npz')==F.sha(P.OUT/'train_labels_weights.npz')))
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for folder in ['models','probabilities'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    print('FIT_COMPLETE',flush=True)


def report():
    F.guard();c=verify();R.check(OUT,'probability_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve source outcomes')
    ys=[np.load(OUT/'valid'/f'{d}.npz')['labels'] for d in c['sources']]
    train_ys=[np.load(OUT/'train_multi'/f'{d}.npz')['labels'] for d in c['sources']]
    native=F.read(P.OUT/'source_report.json')['reports']['baseline'];reports={'native':native}
    for arm in c['arms']:
        payload=dict(np.load(OUT/'probabilities'/f'{arm}.npz'));ps=[payload[d+'_valid'] for d in c['sources']]
        cal=exact_thresholds(ps,ys);attainable=sum(k>1 for lo,hi,k in calibration_intervals(ps,ys)[0])
        chosen=cal['selections']['balanced_exact'];gate=c['source_gate']
        passed=chosen['multi_correct']>=gate['min_multi_correct'] and chosen['single_correct']>=gate['min_single_correct'] and attainable>=gate['min_attainable_multi']
        reports[arm]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
            train_multi=threshold_metrics([payload[d+'_train_multi'] for d in c['sources']],train_ys,chosen['threshold']),
            districts={d:threshold_metrics([p],[y],chosen['threshold']) for d,p,y in zip(c['sources'],ps,ys)})
        F.write(OUT/'calibration'/f'{arm}.json',cal);print(arm,reports[arm],flush=True)
    F.write(OUT/'source_report.json',dict(reports=reports,adopted=False,target_scored=False,
        next_action='Only source-gate passing candidate may enter separately frozen target evaluation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','report']);globals()[p.parse_args().phase]()
