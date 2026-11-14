"""Post-completion target artifact audit; selection remains sealed."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_interaction as run
from run_modernnca_relation import cpu_task,predict,digest,save
from zoning import DEVICE,evaluate


def main():
    complete=json.loads((run.OUT/'completed.json').read_text(encoding='utf8'))
    assert complete['target_evaluated']
    sealed=json.loads((run.OUT/'sealed_selection.json').read_text(encoding='utf8'))
    assert all(digest(run.OUT/p)==h for p,h in sealed.items())
    models={};perseed={};maxerr=0.;count=0
    for k,s in run.base.BATCHES:
        _,D,R,X=run.base.loaded(k,s);folder=run.OUT/f'k{k}_s{s}'
        rows={m:[[],[],[]] for m in run.METHODS};cases=[]
        for d in D:
            audit=json.loads((folder/'training_audit'/f'{d}.json').read_text(encoding='utf8'))
            er=evaluate.eval_rows(D[d],R[d].known)
            assert not(set(audit[d]['ids']) & {r['q'] for r in er})
            ps={}
            for kind in run.METHODS:
                P=np.load(folder/kind/'probs'/f'{d}.npz')['probs']
                assert P.shape==(3,D[d].n,D[d].school_n)
                assert np.isfinite(P).all() and (P>=0).all() and np.allclose(P.sum(-1),1,atol=2e-5)
                choice=json.loads((run.OUT/'selection'/f'{d}_{kind}.json').read_text(encoding='utf8'))
                for j,seed in enumerate(run.base.SEEDS):
                    p=folder/kind/'seeds'/f'{d}_{seed}.npz'
                    assert np.array_equal(P[j],np.load(p)['P'])
                    info=json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
                    assert info['steps']==choice['steps'] and info['hp']==choice['hp']
                    record=dict(probability_sha256=digest(p),weights_sha256=digest(p.with_suffix('.pt')),fit_sha256=digest(p.with_suffix('.json')))
                    # Predefined audit sample: first seed of EVERY district/method/batch.
                    if j==0:
                        m=run.make(kind,X[d],None,seed)
                        m.load_state_dict(torch.load(p.with_suffix('.pt'),map_location=DEVICE,weights_only=True))
                        replay=predict(m,cpu_task(d,np.arange(D[d].n),D,X))
                        error=float(np.abs(P[j]-replay).max());assert error<3e-6
                        assert np.array_equal(P[j].argmax(1),replay.argmax(1))
                        maxerr=max(maxerr,error);record['replay_max_error']=error;count+=1
                        del m;gc.collect();torch.cuda.empty_cache()
                    models[str(p.relative_to(run.OUT))]=record
                    rows[kind][j]+=evaluate.correct(P[j],D[d],er)
                ps[kind]=P
            means={m:P.mean(0) for m,P in ps.items()}
            for e in er:
                q=e['q'];cases.append(dict(district=d,city=D[d].city,**e,
                    predictions={m:int(P[q].argmax()) for m,P in means.items()},
                    seed_predictions={m:[int(pp[q].argmax()) for pp in P] for m,P in ps.items()}))
            print('TARGET AUDIT',k,s,d,'replayed',count,flush=True)
        save(folder/'scored_cases.json',cases)
        perseed[f'k{k}_s{s}']={m:[evaluate.summary(rr) for rr in seeds] for m,seeds in rows.items()}
        del D,R,X;gc.collect();torch.cuda.empty_cache()
    assert len(models)==384 and count==128
    save(run.DOC/'目标完整性审计.json',dict(artifacts=len(models),replayed=count,max_error=maxerr,
        models=models,per_seed=perseed,script_sha256=digest(Path(__file__)),
        note='Every seed artifact checked; first seed of every district/method/batch replayed. No target-driven setting changes.'))
    print('TARGET AUDIT COMPLETE',count,maxerr,flush=True)


if __name__=='__main__':main()
