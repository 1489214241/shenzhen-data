"""Development-only diagnosis of the frozen ModernNCA-LN research baseline."""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_full_evidence as base
from zoning import DEVICE, evaluate
from zoning.modernnca_normalization import LayerNormModernNCA
from diagnose_modernnca import components
from run_modernnca_relation import cpu_task,device_batch,save,digest

OLD=REPO/'outputs/modernnca_mechanism_20261112'
DOC=REPO.parent/'科研推进/ModernNCA针对性改进_20261113'


def main():
    records=[]; maximum=0.; summary={}
    for k,s in [(0,1),(1,1)]:
        c,D,R,X=base.loaded(k,s); batch=f'k{k}_s{s}'; summary[batch]={}
        for d in D:
            rows=evaluate.eval_rows(D[d],R[d].known); ids=[r['q'] for r in rows]
            task=cpu_task(d,ids,D,X); rr=[]
            for seed in base.SEEDS:
                sf=OLD/batch/'layernorm/seeds'/f'{d}_{seed}.pt'
                m=LayerNormModernNCA(X[d].np_,X[d].na,seed).to(DEVICE)
                m.load_state_dict(torch.load(sf,map_location=DEVICE,weights_only=True)); m.eval()
                expected=np.load(sf.with_suffix('.npz'))['P'][ids]
                for st in range(0,len(ids),64):
                    t=device_batch(task,slice(st,st+64)); z=components(m,t)
                    z={a:b.cpu().numpy() for a,b in z.items()}
                    maximum=max(maximum,float(np.abs(z['P']-expected[st:st+64]).max()))
                    valid=t['valid'].cpu().numpy()
                    for j,row in enumerate(rows[st:st+64]):
                        if len(row['truth'])!=1: continue
                        truth=int(row['truth'][0]); q=int(row['q']); pred=int(z['P'][j].argmax())
                        refpred=int(z['reference'][j].argmax()); anchorpred=int(z['anchor'][j].argmax())
                        rt=valid[j] & (X[d].ref_schools==truth)
                        ro=valid[j] & (X[d].ref_schools!=truth)
                        dd=np.linalg.norm(D[d].coords[X[d].ref_sites]-D[d].coords[q],axis=-1)
                        nearest_true=float(dd[rt].min()) if rt.any() else None
                        nearest_other=float(dd[ro].min()) if ro.any() else None
                        category=('correct' if pred==truth else
                            'no_true_reference' if not rt.any() else
                            'fusion_overturns_reference' if refpred==truth else
                            'fusion_overturns_anchor' if anchorpred==truth else
                            'sum_overturns_top_token' if int(z['top'][j])==truth else
                            'both_channels_wrong_with_true_reference')
                        r=dict(batch=batch,district=d,city=D[d].city,q=q,seed=seed,truth=truth,pred=pred,
                            category=category,refpred=refpred,anchorpred=anchorpred,top=int(z['top'][j]),
                            true_reference=bool(rt.any()),nearest_true_m=nearest_true,nearest_other_m=nearest_other,
                            true_reference_nearer=nearest_true is not None and (nearest_other is None or nearest_true<nearest_other),
                            predicted_policy_exclusion=bool(X[d].anchor[q,pred,X[d].anchor_blocks['policy'].stop-1]>0),
                            true_policy_exclusion=bool(X[d].anchor[q,truth,X[d].anchor_blocks['policy'].stop-1]>0),
                            anchor_mass=float(z['anchor'][j].sum()))
                        rr.append(r)
                del m
            counts=Counter(r['category'] for r in rr)
            counts['wrong_despite_nearer_true_reference']=sum(r['pred']!=r['truth'] and r['true_reference_nearer'] for r in rr)
            counts['wrong_predicted_excluded_school']=sum(r['pred']!=r['truth'] and r['predicted_policy_exclusion'] for r in rr)
            counts['true_school_excluded_in_input']=sum(r['true_policy_exclusion'] for r in rr)
            summary[batch][d]=dict(counts); records+=rr
            print(batch,d,dict(counts),flush=True)
        del c,D,R,X,task
    assert maximum<2e-5
    save(DOC/'基线诊断逐题.json',records)
    save(DOC/'基线诊断.json',dict(counts=summary,replay_max_error=maximum,
        baseline_manifest_sha256=digest(OLD/'normalization_frozen.json'),
        note='Previously used development data. Unique-truth cases only; seed-case counts are not independent samples. Diagnoses motivate hypotheses, not establish causality.'))


if __name__=='__main__': main()
