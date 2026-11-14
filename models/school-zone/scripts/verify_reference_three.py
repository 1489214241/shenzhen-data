"""Final artifact, coverage, probability and checkpoint replay audit."""
import sys
import json
import csv
import platform
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import torch
from zoning import reference_trials as T, evaluate
from run_reference_three import ROOT,DOC,OLD,BATCHES,METHODS,SEEDS,DISTRICTS,audit_data,filehash,source_manifest,save

def main():
    frozen=json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    assert frozen['sources']==source_manifest()
    fhash=filehash(ROOT/'freeze.json')
    checked=0; rows_checked=0; replay_max=0.; records=[]; historical={}
    for k,s in BATCHES:
        _,D,R=audit_data(k,s); batch=f'k{k}_s{s}'
        for d in DISTRICTS:
            path=OLD/batch/'answers'/(d+'.csv')
            with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            assert len(rows)==D[d].n
            assert all(int(r['小区编号'])==i and r['小区']==D[d].site_names[i] for i,r in enumerate(rows))
            known={int(r['小区编号']):r['学校'] for r in rows if r['来源'].startswith('规则判出') or r['来源']=='核验参照'}
            current={q:'、'.join(D[d].school_names[x] for x in ss) for q,ss in R[d].known.items()}
            assert known==current
            historical[batch+':'+d]=dict(sites_and_references_identical=True,old_csv_sha256=filehash(path))
        for method in METHODS:
            for d in DISTRICTS:
                task=T.DistrictTask(D[d],R[d]); qs=[e['q'] for e in evaluate.eval_rows(D[d],R[d].known)]
                has=np.zeros((len(qs),task.S),bool)
                for i,q in enumerate(qs):
                    labels=task.support(q)[1].cpu().numpy()
                    if len(labels): has[i,np.unique(labels)]=True
                    else: has[i]=True
                for seed in SEEDS:
                    folder=ROOT/batch/method/d/str(seed)
                    rec=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                    assert rec['freeze_sha256']==fhash
                    assert rec['prediction_sha256']==filehash(folder/'prediction.npz')
                    assert rec['checkpoint_sha256']==filehash(folder/'model.pt')
                    z=np.load(folder/'prediction.npz'); p=z['probs']
                    assert z['qs'].tolist()==qs and p.shape==(len(qs),task.S)
                    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
                    assert np.all(p[~has]==0), (batch,method,d,seed,'unsupported class has probability')
                    # Every seed checkpoint must restore and reproduce a fixed slice.
                    ck=torch.load(folder/'model.pt',map_location='cuda',weights_only=True)
                    m=T.ReferenceModel(method).cuda(); m.load_state_dict(ck['state_dict'])
                    ix=np.linspace(0,len(qs)-1,3).astype(int)
                    # Preserve original GEMM query batch shape for the inexpensive point models.
                    # GPU GEMV versus GEMM roundoff can be amplified by large classifier logits.
                    replay=(T.predict(m,task,qs)[ix] if method in ('cnaps','feat')
                            else T.predict(m,task,[qs[i] for i in ix]))
                    err=float(np.max(np.abs(replay-p[ix])))
                    replay_max=max(replay_max,err)
                    assert err<3e-6, (batch,method,d,seed,err)
                    checked+=1;rows_checked+=len(qs)
                    records.append(dict(batch=batch,method=method,district=d,seed=seed,
                                        train_seconds=rec['train_seconds'],inference_seconds=rec['inference_seconds']))
    assert checked==72
    result=dict(status='PASS',checkpoints=checked,prediction_rows=rows_checked,
                checkpoint_replay_max_abs_error=replay_max,freeze_sha256=fhash,
                historical_row_audit=historical,
                python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                device=torch.cuda.get_device_name(),records=records)
    save(ROOT/'verification.json',result);save(DOC/'验收.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
