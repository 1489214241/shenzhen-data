"""Source-only training versus validation ranking audit of frozen round03."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pickle
import numpy as np
import torch
import shenzhen_full_inputs as F
from zoning import model as M,DEVICE
from shenzhen_partial_model import BernoulliRetrieval
from shenzhen_exact_calibration import calibration_intervals,threshold_metrics
from experiment_shenzhen_partial_labels import probabilities


def main():
    F.guard()
    root=F.REPO/'outputs/shenzhen_resolution_20261112';old=root/'round03_partial_labels';out=root/'source_training_ranking_audit'
    if out.exists():raise FileExistsError('Keep diagnostic snapshot')
    for fname in ['input_freeze.json','probability_freeze.json']:
        for rel,h in F.read(old/fname)['files'].items():assert F.sha(old/rel)==h,rel
    c=F.read(old/'protocol.json');xs={};ys={};ids={}
    for d in c['sources']:
        with (old/'inputs'/f'{d}.pkl').open('rb') as f:xs[d]=pickle.load(f)
        z=dict(np.load(old/'inputs'/f'{d}_labels.npz'));labels=z['labels']
        ids[d]=np.flatnonzero((labels>=0).all(1));ys[d]=labels[ids[d]].astype(bool)
    start=M.LinearRetrieval(next(iter(xs.values())));reports={}
    for arm in c['arms']:
        allp=[]
        for seed in c['seeds']:
            model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE)
            state=torch.load(old/'checkpoints'/f'{arm}_{seed}.pt',map_location=DEVICE,weights_only=False)
            assert state['stage']=='full' and state['step']==c['steps'];model.load_state_dict(state['model']);model.eval()
            p={d:probabilities(model,x,ids[d],c['chunk']) for d,x in xs.items()};allp.append(p)
            dest=out/'probabilities'/arm;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{seed}.npz',**p)
            print('SOURCE_INFERRED',arm,seed,flush=True)
        mean={d:np.stack([v[d] for v in allp]).mean(0) for d in xs};r={}
        for name,ds in [('全部',list(xs))]+[(d,[d]) for d in xs]:
            pp=[mean[d] for d in ds];yy=[ys[d] for d in ds];intervals,sizes=calibration_intervals(pp,yy)
            t=F.read(old/'source_calibration.json')[arm]['threshold']
            r[name]=dict(threshold=t,**threshold_metrics(pp,yy,t),multi=int((sizes>1).sum()),
                attainable_single=sum(k==1 for lo,hi,k in intervals),attainable_multi=sum(k>1 for lo,hi,k in intervals))
        reports[arm]=r
    F.write(out/'report.json',dict(training_complete_source_rows=reports,target_gold_read=False,
        model_hashes_verified=True,meaning='Training-row diagnostics only; neither independent validation nor target tuning.'))
    F.write(out/'manifest.json',dict(files={str(p.relative_to(out)):F.sha(p) for p in out.rglob('*') if p.is_file()}))
    print({a:r['全部'] for a,r in reports.items()},flush=True)


if __name__=='__main__':main()
