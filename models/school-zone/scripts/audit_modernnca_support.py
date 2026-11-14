"""Replay validation artifacts; interventions never change selection or gate."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_support as run
from run_modernnca_relation import digest,save,predict,device_batch
from zoning import DEVICE


def main():
    run.freeze()
    sealed=json.loads((run.OUT/'sealed_selection.json').read_text(encoding='utf8'))
    assert len(sealed)==40 and all(digest(run.OUT/p)==h for p,h in sealed.items())
    c,D,R,X=run.base.loaded(0,1)
    report={}; max_error=0.; cases=[]
    for d in D:
        t=run.val_task(d,D,R,X); y=t['y'].numpy()
        ref=np.load(run.OUT/'validation/identity'/f'{d}.npz')['P'].argmax(1)
        for kind in run.METHODS:
            folder=run.OUT/'validation'/kind
            m=run.make(kind,X[d],None,run.base.SEEDS[0])
            m.load_state_dict(torch.load(folder/f'{d}.pt',map_location=DEVICE,weights_only=True)); m.eval()
            original=np.load(folder/f'{d}.npz')['P']; replay=predict(m,t)
            err=float(np.max(np.abs(replay-original))); max_error=max(max_error,err)
            assert err<3e-6,(d,kind,err)
            choice=json.loads((run.OUT/'selection'/f'{d}_{kind}.json').read_text(encoding='utf8'))
            assert float((replay.argmax(1)==y).mean())==choice['best']['accuracy']
            # The actual learned weights are retained, only the readout changes.
            switches=['identity','scalar','diagonal'] if kind=='covariance' else ['identity'] if kind=='ridge' else []
            alt={k:[] for k in switches}
            with torch.no_grad():
                for st in range(0,len(t['ids']),128):
                    b=device_batch(t,slice(st,st+128)); enc=m.encode_support(b)
                    for k in switches: alt[k].append(m.readout(*enc,b,variant=k).cpu().numpy())
            outcomes={}; learned=replay.argmax(1)==y
            for k,ps in alt.items():
                pp=np.concatenate(ps); switched=pp.argmax(1)==y
                outcomes[k]=dict(learned_head_accuracy=float(learned.mean()*100),
                    switched_accuracy=float(switched.mean()*100),
                    learned_minus_switched=float((learned.astype(float)-switched).mean()*100),
                    fixed=int((learned&~switched).sum()),broken=int((~learned&switched).sum()))
                for i,q in enumerate(t['ids']):
                    cases.append(dict(target=d,validation=run.base.VALIDATION[d],city=D[run.base.VALIDATION[d]].city,
                        q=int(q),kind=kind,switch=k,truth=int(y[i]),base_prediction=int(ref[i]),
                        trained_prediction=int(replay[i].argmax()),switched_prediction=int(pp[i].argmax())))
            report[f'{d}_{kind}']=dict(n=len(y),replay_max_error=err,interventions=outcomes,
                state_sha256=digest(folder/f'{d}.pt'),probability_sha256=digest(folder/f'{d}.npz'))
            print('AUDIT',d,kind,err,flush=True)
    save(run.DOC/'重放审计.json',dict(max_error=max_error,models=report,
        script_sha256=digest(Path(__file__)),note='Validation only; interventions are post-training explanations, not independent retrained ablations or selection criteria.'))
    save(run.OUT/'trained_intervention_cases.json',cases)


if __name__=='__main__': main()
