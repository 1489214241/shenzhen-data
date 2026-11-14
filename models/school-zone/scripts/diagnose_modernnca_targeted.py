"""Predeclared descriptive post-fit diagnosis. Inference ablation is not retraining."""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_targeted as run
from run_modernnca_relation import cpu_task,device_batch,digest,save
from zoning import DEVICE,evaluate


def freeze():
    plan=dict(source_sha256=digest(Path(__file__)),experiment_sha256=digest(run.OUT/'frozen.json'),
        batch='k1_s1',all_districts=True,seeds=run.base.SEEDS,methods=run.METHODS,
        analysis='For each fitted candidate, disable only its added residual at inference with all fitted encoder/query weights unchanged. Record prediction changes, paired delta and distributions of residuals. Descriptive reliance check, not independently retrained ablation, not a candidate for selection/adoption.',
        before_target_scoring=not (run.DOC/'结果.json').exists())
    path=run.OUT/'diagnosis_frozen.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf8')); plan['before_target_scoring']=old['before_target_scoring']; assert plan==old
    else:
        assert plan['before_target_scoring']; save(path,plan)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);args=p.parse_args();freeze()
    if args.phase=='freeze':return
    c,D,R,X=run.base.loaded(1,1); batch='k1_s1'; report={}; joint={m:[[],[]] for m in run.METHODS}
    for d in D:
        rows=evaluate.eval_rows(D[d],R[d].known); ids=[r['q'] for r in rows]; task=cpu_task(d,ids,D,X)
        for kind in run.METHODS:
            original=[]; removed=[]; stats=[]
            for seed in run.base.SEEDS:
                sf=run.OUT/batch/kind/'seeds'/f'{d}_{seed}.pt'
                m=run.make(kind,X[d],None,seed);m.load_state_dict(torch.load(sf,map_location=DEVICE,weights_only=True));m.eval()
                pred=[]; no=[]; delta=[]; gate=[]; gate0=[]
                with torch.no_grad():
                    for st in range(0,len(ids),64):
                        t=device_batch(task,slice(st,st+64)); P,z=m(t,details=True);pred.append(P.cpu().numpy())
                        delta.append(z['delta'].cpu().numpy())
                        nr=t['px'].shape[1]
                        gate.append(z['logits'].softmax(-1)[:,:nr].sum(-1).cpu().numpy())
                        gate0.append(z['base_logits'].softmax(-1)[:,:nr].sum(-1).cpu().numpy())
                    if kind=='constantgate':m.offset.zero_()
                    else:m.residual[-1].weight.zero_();m.residual[-1].bias.zero_()
                    for st in range(0,len(ids),64):no.append(m(device_batch(task,slice(st,st+64))).cpu().numpy())
                p1=np.concatenate(pred);p0=np.concatenate(no);de=np.concatenate(delta);g=np.concatenate(gate);g0=np.concatenate(gate0)
                expected=np.load(sf.with_suffix('.npz'))['P'][ids]
                assert np.abs(p1-expected).max()<2e-5
                stat=dict(seed=seed,delta_mean=float(de.mean()),delta_std=float(de.std()),delta_min=float(de.min()),delta_max=float(de.max()),
                    reference_mass_mean=float(g.mean()),reference_mass_before_offset_mean=float(g0.mean()),
                    changed_predictions=int((p1.argmax(-1)!=p0.argmax(-1)).sum()))
                if kind=='schoolquery':stat['mean_school_residual_norm']=float(np.linalg.norm(de,axis=-1).mean())
                stats.append(stat);original.append(p1);removed.append(p0);del m
            # evaluate.correct expects full district row indexing.
            p1=np.zeros((D[d].n,D[d].school_n));p0=p1.copy()
            p1[ids]=np.mean(original,axis=0);p0[ids]=np.mean(removed,axis=0)
            a=evaluate.correct(p1,D[d],rows);b=evaluate.correct(p0,D[d],rows)
            joint[kind][0]+=a;joint[kind][1]+=b
            out=run.OUT/'diagnostics'/f'{d}_{kind}.npz';out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out,ids=np.array(ids),original=np.stack(original),removed=np.stack(removed))
            report[f'{d}/{kind}']=dict(per_seed=stats,accuracy=evaluate.summary(a)[d],without_residual=evaluate.summary(b)[d],artifact_sha256=digest(out))
        del task;gc.collect();torch.cuda.empty_cache()
    comparisons={m:evaluate.compare(a,b,dict(c['evaluate'],draws=10000)) for m,(a,b) in joint.items()}
    save(run.DOC/'新增模块使用诊断.json',dict(note='Inference-only residual removal; no retraining. Descriptive, not causal or an additional adoption test.',per_district=report,comparisons=comparisons))
    print('Post-fit reliance diagnosis complete',comparisons,flush=True)


if __name__=='__main__':main()
