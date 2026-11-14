"""Prospective next development run: retain old top1, route new evidence to gate.

Previous target scores were seen. This is explicitly not independent validation.
"""
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import experiment_conditional_admission as A

BASE=A.OUT
OUT=A.M.REPO/'outputs/modern_conditional_admission_hybrid_20261113'

def install():A.OUT=OUT

def prepare():
    A.M.guard();c=A.verify()
    if OUT.exists():raise FileExistsError('Preserve hybrid experiment')
    OUT.mkdir(parents=True)
    c.update(primary='routed_bce_legacy minus control_bce_legacy',
        rationale='Earlier target scores seen: routed evidence aids additions but changes native top1. Test old native categorical probabilities plus routed gate evidence, retaining old top1 exactly.',
        feature='Old control native ensemble probabilities/top1 for BOTH arms; routed anchor/reference features used only by admission. Recompute target mask excluding old top1.',
        prior_experiment=str(BASE),prior_target_results_seen=True,independent_validation=False,
        control_fits_reused_unchanged=True,primary_loss='bce',secondary_loss='zlpr',
        selection_unchanged=True)
    A.write(OUT/'protocol.json',c)
    links=A.read(BASE/'freeze.json')
    links[str(BASE/'freeze.json')]=A.sha(BASE/'freeze.json')
    links[str(OUT/'protocol.json')]=A.sha(OUT/'protocol.json')
    links[str(Path(__file__).resolve())]=A.sha(Path(__file__))
    for d in c['sources']+c['targets']:
        source=BASE/'cache/control'/f'{d}.npz';dest=OUT/'cache/control'/f'{d}.npz'
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
        old=dict(np.load(source));routed=dict(np.load(BASE/'cache/routed'/f'{d}.npz'))
        x=A.M.pkl(A.E.INPUT/d/'routed.pkl');p=old['prob']
        f=np.concatenate([A.features(x,p,lo,min(lo+32,x.n)) for lo in range(0,x.n,32)])
        y=dict(np.load(A.E.INPUT/d/'routed_labels.npz'))
        labels=y['labels'].copy();labels[np.arange(x.n),p.argmax(1)]=-1
        w=np.load(A.E.TEACHER/f'{d}_weights.npy').copy()
        if d in c['targets']:
            unit=w.max();w[:]=0;ids=y['train_ids']
            _,inv,count=np.unique(x.groups[ids],return_inverse=True,return_counts=True);w[ids]=unit/count[inv]
        w[~(labels>=0).any(1)]=0
        routed.update(feature=f,prob=p,labels=labels,weight=w)
        dest=OUT/'cache/routed'/f'{d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**routed)
        for arm in ['control','routed']:
            pth=OUT/'cache'/arm/f'{d}.npz';links[str(pth)]=A.sha(pth)
    for kind in ['bce','zlpr']:
        for folder in ['selection','weights']:
            source=BASE/folder/f'control_{kind}';dest=OUT/folder/f'control_{kind}'
            shutil.copytree(source,dest)
        p=BASE/'selection'/f'control_{kind}.json';dest=OUT/'selection'/p.name;shutil.copy2(p,dest)
    A.write(OUT/'freeze.json',links)
    A.M.log('HYBRID PROTOCOL AND INPUTS SEALED; NO CURRENT TARGET SCORES READ')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','fit','infer','score','audit_inputs','audit_outputs'])
    phase=ap.parse_args().phase
    if phase=='prepare':prepare()
    else:
        install()
        if phase.startswith('audit_'):
            import audit_conditional_admission as audit
            getattr(audit,phase.removeprefix('audit_'))()
        else:getattr(A,phase)()
