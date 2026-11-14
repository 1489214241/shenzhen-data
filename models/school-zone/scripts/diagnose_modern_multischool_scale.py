"""Source-only score-scale diagnostic; no target labels or predictions read."""
from pathlib import Path
import numpy as np
import torch
import continue_modernnca_evidence_v2 as C

C.install(); E=C.E
OUT=E.M.REPO/'outputs/modern_scale_diagnostic_20261113'

def main():
    E.M.guard(); c=E.verify(); OUT.mkdir(parents=True,exist_ok=True)
    y={d:dict(np.load(E.INPUT/d/'control_labels.npz')) for d in c['sources']}
    x={d:E.M.pkl(E.INPUT/d/'control.pkl') for d in c['sources']}
    logits=[]
    for seed in c['seeds']:
        model=E.teacher(c,seed).eval(); pp={}
        for d in c['sources']:
            t=E.M.task(x[d],y[d]['valid_ids']); zs=[]
            with torch.no_grad():
                for lo in range(0,t['n'],64):
                    zs.append(model(E.M.device_batch(t,slice(lo,lo+64))).cpu().numpy())
            pp[d]=np.concatenate(zs)
        logits.append(pp)
    variants={key:[] for key in ['raw_sigmoid','softmax','max_relative']}
    stats={}; attainable=0; nm=0
    for d in c['sources']:
        zz=[torch.tensor(p[d],dtype=torch.float64) for p in logits]
        p=torch.stack([z.softmax(1) for z in zz]).mean(0).numpy()
        variants['raw_sigmoid'].append(torch.stack([z.sigmoid() for z in zz]).mean(0).numpy())
        variants['softmax'].append(p)
        variants['max_relative'].append(p/p.max(1,keepdims=True))
        yy=y[d]['valid_labels'].astype(bool); multi=yy.sum(1)>1
        separated=np.array([a[g].min()>a[~g].max() for a,g in zip(p,yy)])
        attainable+=int((separated&multi).sum());nm+=int(multi.sum())
        z=zz[0]; shift=torch.linspace(-20,20,len(z))[:,None]
        stats[d]=dict(top_score_quantiles=np.quantile(z.max(1).values,[0,.25,.5,.75,1]).tolist(),
            shift_softmax_max_error=float((z.softmax(1)-(z+shift).softmax(1)).abs().max()),
            shift_sigmoid_max_error=float((z.sigmoid()-(z+shift).sigmoid()).abs().max()),
            multi=int(multi.sum()),multi_ranking_separable=int((separated&multi).sum()))
    labels=[y[d]['valid_labels'] for d in c['sources']]
    baseline=sum(int((p.argmax(1)[yy.sum(1)==1]==yy.argmax(1)[yy.sum(1)==1]).sum())
                 for p,yy in zip(variants['softmax'],labels))
    result=dict(source_only=True,target_gold_read=False,source_single_baseline=baseline,
        ranking_separable_multi=attainable,source_multi=nm,districts=stats,
        calibration={k:E.calibrate(p,labels,baseline) for k,p in variants.items()})
    E.write(OUT/'source_diagnostic.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
