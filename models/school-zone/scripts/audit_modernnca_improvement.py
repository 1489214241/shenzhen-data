"""Independent audit of normalization-only adaptation and its saved predictions."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
import run_modernnca_normalization as run
from zoning import DEVICE, evaluate, model as MD
from zoning.full_evidence import tokens
from run_modernnca_relation import digest,save


def audit():
    run.freeze()
    frozen=json.loads((run.OUT/'normalization_frozen.json').read_text(encoding='utf8'))
    original_audit=json.loads((run.base.DOC/'完整性审计.json').read_text(encoding='utf8'))
    sealed=json.loads((run.OUT/'sealed_predictions.json').read_text(encoding='utf8'))
    assert len(sealed)==64
    assert all(digest(run.OUT/p)==h for p,h in sealed.items())
    report=dict(source_hashes_pass=True,predictions={},selection={},input_contract={},
                audit_source_sha256=digest(Path(__file__)))
    for k,s in run.base.BATCHES:
        batch=f'k{k}_s{s}'; c,D,R,X=run.base.loaded(k,s)
        assert digest(run.base.SOURCE/batch/'inputs.pkl')==frozen['input_hashes'][batch]
        oldhash=json.loads((run.OLD/batch/'prediction_hashes.json').read_text(encoding='utf8'))
        assert all(digest(run.OLD/batch/p)==h for p,h in oldhash.items())
        for d in D:
            x=X[d]; ids=np.arange(min(13,D[d].n)); t=MD.batch(x,ids)
            packed,valid=tokens(t,x.np_); nr=len(x.ref_sites)
            assert torch.equal(packed[:,:nr,:-2],t['px'])
            assert torch.equal(packed[:,nr:,x.np_:-2],t['ax'])
            assert torch.equal(packed[:,:nr,-2],t['gb'][None].expand(len(ids),-1))
            assert torch.equal(valid[:,:nr],t['valid']) and valid[:,nr:].all()
            assert torch.equal(t['mem'][-D[d].school_n:],torch.eye(D[d].school_n,device=DEVICE))
            expected_refs=sorted({(q,school) for q,ss in R[d].known.items() if q not in D[d].not_reference for school in ss})
            assert expected_refs==list(zip(x.ref_sites.tolist(),x.ref_schools.tolist()))
            auditpath=run.OUT/batch/'training_audit'/f'{d}.json'
            recorded=json.loads(auditpath.read_text(encoding='utf8'))
            original=json.loads((run.OLD/batch/'training_audit'/f'{d}.json').read_text(encoding='utf8'))
            assert recorded==original
            assert not set(recorded[d]['ids']) & {r['q'] for r in evaluate.eval_rows(D[d],R[d].known)}
            report['input_contract'][f'{batch}/{d}']=dict(references=nr,schools=D[d].school_n,
                same_training_ids_labels_groups=True,lossless_packing=True,target_labels_excluded=True)
            for method in ['layernorm','pooled_bn']:
                f=run.OUT/batch/method/'probs'/f'{d}.npz'; arr=np.load(f)['probs']
                assert arr.shape==(3,D[d].n,D[d].school_n)
                assert np.isfinite(arr).all() and np.allclose(arr.sum(-1),1,atol=2e-5)
                weights={}
                for j,seed in enumerate(run.base.SEEDS):
                    sf=run.OUT/batch/method/'seeds'/f'{d}_{seed}.npz'
                    assert np.array_equal(arr[j],np.load(sf)['P'])
                    state=torch.load(sf.with_suffix('.pt'),map_location='cpu',weights_only=True)
                    if method=='layernorm':
                        fit_info=json.loads(sf.with_suffix('.json').read_text(encoding='utf8'))
                        choice=json.loads((run.OUT/'selection'/f'{d}_layernorm.json').read_text(encoding='utf8'))
                        assert fit_info['complete'] and fit_info['steps']==choice['steps']
                        assert fit_info['hp']==choice['hp']
                    if method=='pooled_bn':
                        log=json.loads(sf.with_suffix('.json').read_text(encoding='utf8'))
                        assert log['calibration_ids']=={dd:v['ids'] for dd,v in recorded.items()}
                        src=Path(log['source']); assert digest(src)==log['source_sha256']
                        assert digest(src)==original_audit['predictions'][batch][f'{d}/modernnca']['weight_hashes'][str(seed)]
                        orig=torch.load(src,map_location='cpu',weights_only=True)
                        assert set(state)==set(orig)
                        for name,value in state.items():
                            if not name.endswith(('running_mean','running_var','num_batches_tracked')):
                                assert torch.equal(value,orig[name]),name
                    weights[str(seed)]=digest(sf.with_suffix('.pt'))
                    if j==0:
                        replay=run.make('layernorm' if method=='layernorm' else 'modernnca',x,None,seed)
                        replay.load_state_dict(state); replay.eval()
                        with torch.no_grad(): pp=replay(t).cpu().numpy()
                        error=float(np.abs(pp-arr[j,ids]).max()); assert error<2e-5
                        del replay
                report['predictions'][f'{batch}/{d}/{method}']=dict(shape=list(arr.shape),replay_max_error=error,
                    prediction_sha256=digest(f),weight_sha256=weights)
            del t,packed,valid
        cases=json.loads((run.OUT/batch/'scored_cases.json').read_text(encoding='utf8'))
        expected=[(d,r['q'],r['truth']) for d in D for r in evaluate.eval_rows(D[d],R[d].known)]
        assert [(r['district'],r['q'],r['truth']) for r in cases]==expected
        for method in ['layernorm','pooled_bn','modernnca','reference']:
            root=run.OUT if method in ['layernorm','pooled_bn'] else run.OLD
            for d in D:
                pred=np.load(root/batch/method/'probs'/f'{d}.npz')['probs'].mean(0).argmax(1)
                assert all(r['predictions'][method]==int(pred[r['q']]) for r in cases if r['district']==d)
        del c,D,R,X; gc.collect(); torch.cuda.empty_cache()
    for f in sorted((run.OUT/'selection').glob('*_layernorm.json')):
        record=json.loads(f.read_text(encoding='utf8')); d=f.stem.rsplit('_',1)[0]
        assert record['validation']==run.base.VALIDATION[d]
        assert [t['hp'] for t in record['trials']]==run.base.GRID
        assert all(t['complete'] for t in record['trials'])
        chosen=max(record['trials'],key=lambda z:(z['best']['accuracy'],-z['best']['nll'],-z['best']['step']))
        assert record['hp']==chosen['hp'] and record['steps']==chosen['best']['step']
        report['selection'][d]=dict(steps=record['steps'],at_hard_cap=record['steps']==1280,pass_=True)
    assert len(report['selection'])==8 and len(report['predictions'])==64
    report.update(complete=True,pass_=True)
    save(run.DOC/'完整性审计.json',report)
    print('PASS',len(report['predictions']),'ensembles; max replay error',max(r['replay_max_error'] for r in report['predictions'].values()),flush=True)


if __name__=='__main__': audit()
