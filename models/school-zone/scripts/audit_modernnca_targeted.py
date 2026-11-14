"""Audit frozen inputs, matched supervision, selection, and every saved seed."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_targeted as run
from zoning import DEVICE, evaluate, model as MD
from zoning.full_evidence import tokens
from run_modernnca_relation import digest,save


def main():
    run.freeze()
    sealed=json.loads((run.OUT/'sealed_predictions.json').read_text(encoding='utf8'))
    assert len(sealed)==96 and all(digest(run.OUT/p)==h for p,h in sealed.items())
    report=dict(source_sha256=digest(Path(__file__)),selection={},predictions={},input_contract={},complete=False)
    for d in run.base.VALIDATION:
        for m in run.METHODS:
            f=run.OUT/'selection'/f'{d}_{m}.json'; v=json.loads(f.read_text(encoding='utf8'))
            assert v['validation']==run.base.VALIDATION[d]
            assert [t['hp'] for t in v['trials']]==run.base.GRID and all(t['complete'] for t in v['trials'])
            chosen=max(v['trials'],key=lambda z:(z['best']['accuracy'],-z['best']['nll'],-z['best']['step']))
            assert v['hp']==chosen['hp'] and v['steps']==chosen['best']['step']
            report['selection'][f'{d}/{m}']=dict(steps=v['steps'],hard_cap=v['steps']==1280,sha256=digest(f))
    for k,s in run.base.BATCHES:
        batch=f'k{k}_s{s}'; c,D,R,X=run.base.loaded(k,s)
        cases=json.loads((run.OUT/batch/'scored_cases.json').read_text(encoding='utf8'))
        assert [(r['district'],r['q'],r['truth']) for r in cases]==[(d,r['q'],r['truth']) for d in D for r in evaluate.eval_rows(D[d],R[d].known)]
        for d in D:
            x=X[d]; ids=np.linspace(0,D[d].n-1,min(17,D[d].n),dtype=int); t=MD.batch(x,ids)
            packed,valid=tokens(t,x.np_); nr=len(x.ref_sites)
            assert torch.equal(packed[:,:nr,:-2],t['px'])
            assert torch.equal(packed[:,nr:,x.np_:-2],t['ax'])
            assert torch.equal(packed[:,:nr,-2],t['gb'][None].expand(len(ids),-1))
            assert torch.equal(valid[:,:nr],t['valid']) and valid[:,nr:].all()
            assert torch.equal(t['mem'][-D[d].school_n:],torch.eye(D[d].school_n,device=DEVICE))
            expected=sorted({(q,school) for q,ss in R[d].known.items() if q not in D[d].not_reference for school in ss})
            assert expected==list(zip(x.ref_sites.tolist(),x.ref_schools.tolist()))
            recorded=json.loads((run.OUT/batch/'training_audit'/f'{d}.json').read_text(encoding='utf8'))
            for root in [run.OLD,run.LN]:
                assert recorded==json.loads((root/batch/'training_audit'/f'{d}.json').read_text(encoding='utf8'))
            assert not set(recorded[d]['ids']) & {r['q'] for r in evaluate.eval_rows(D[d],R[d].known)}
            report['input_contract'][f'{batch}/{d}']=dict(references=nr,schools=D[d].school_n,same_training=True,all_candidates=True,lossless=True)
            for m in ['layernorm','reference']+run.METHODS:
                root=run.root_for(m); f=root/batch/m/'probs'/f'{d}.npz'; arr=np.load(f)['probs']
                assert arr.shape==(3,D[d].n,D[d].school_n)
                assert np.isfinite(arr).all() and np.allclose(arr.sum(-1),1,atol=2e-5)
                pred=arr.mean(0).argmax(1)
                assert all(r['predictions'][m]==int(pred[r['q']]) for r in cases if r['district']==d)
                if m not in run.METHODS: continue
                errors=[]; weights={}
                choice=json.loads((run.OUT/'selection'/f'{d}_{m}.json').read_text(encoding='utf8'))
                for j,seed in enumerate(run.base.SEEDS):
                    sf=root/batch/m/'seeds'/f'{d}_{seed}.npz'
                    assert np.array_equal(arr[j],np.load(sf)['P'])
                    info=json.loads(sf.with_suffix('.json').read_text(encoding='utf8'))
                    assert info['complete'] and info['steps']==choice['steps'] and info['hp']==choice['hp']
                    replay=run.make(m,x,None,seed); replay.load_state_dict(torch.load(sf.with_suffix('.pt'),map_location=DEVICE,weights_only=True)); replay.eval()
                    with torch.no_grad(): pp=replay(t).cpu().numpy()
                    error=float(np.abs(pp-arr[j,ids]).max()); assert error<2e-5
                    errors.append(error); weights[str(seed)]=digest(sf.with_suffix('.pt')); del replay
                report['predictions'][f'{batch}/{d}/{m}']=dict(replay_max_error=max(errors),weight_hashes=weights,probability_sha256=digest(f))
            del t,packed,valid
        del c,D,R,X; gc.collect(); torch.cuda.empty_cache()
    assert len(report['selection'])==24 and len(report['predictions'])==96
    results=json.loads((run.DOC/'结果.json').read_text(encoding='utf8'))
    for batch,res in results.items():
        for a,b in run.PAIRS:
            v=res['comparisons'][f'{a}-{b}']
            for city in ['两城','长沙','杭州']:
                ra=res['model'][a][city];rb=res['model'][b][city]
                assert ra['n']==rb['n']
                assert abs(ra['accuracy']-rb['accuracy']-v[city]['delta'])<1e-10
                assert abs((v[city]['fixed']-v[city]['broken'])/ra['n']*100-v[city]['delta'])<1e-10
            assert v['adjusted_ci'][0]<=v['两城']['ci'][0] and v['adjusted_ci'][1]>=v['两城']['ci'][1]
    report.update(complete=True,pass_=True,score_arithmetic_pass=True)
    save(run.DOC/'完整性审计.json',report)
    print('PASS',len(report['predictions']),'ensembles / 288 seeds; max replay error',max(v['replay_max_error'] for v in report['predictions'].values()))


if __name__=='__main__': main()
