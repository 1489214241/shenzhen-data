"""Prespecified checkpoint inference diagnostics; never changes training or selection."""
import sys
import json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from zoning import reference_trials as T, evaluate
from run_reference_three import ROOT,DOC,BATCHES,METHODS,SEEDS,DISTRICTS,audit_data,save,comparison,filehash

@torch.no_grad()
def infer(model,task,qs):
    model.eval()
    out=np.zeros((len(qs),task.S),np.float32)
    refgroups=set(task.groups[task.ri]);buckets={}
    for i,q in enumerate(qs):
        key=task.groups[q] if task.groups[q] in refgroups else None
        buckets.setdefault(key,[]).append((i,q))
    for pairs in buckets.values():
        sx,sy=task.support(pairs[0][1]);ix=[i for i,q in pairs]
        if not len(sy):
            out[ix]=1/task.S
            continue
        hs,hq=model.enc(sx),model.enc(task.x[[q for i,q in pairs]])
        if model.method=='egnn':
            p=(-(hq[:,None]-hs[None]).square().sum(-1)).softmax(-1) @ F.one_hot(sy,task.S).float()
        else:
            classes,proto=T.class_means(hs,sy)
            if model.method=='feat': logits=-(hq[:,None]-proto[None]).square().sum(-1)
            else:
                pars=model.classifier(dict(enumerate(proto.split(1))))
                logits=F.linear(hq,pars['weight_mean'],pars['bias_mean'])
            p=T.expand_probability(logits,classes,task.S)
        out[ix]=p.cpu().numpy()
    return out

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    result={}
    frozen=json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    for k,s in BATCHES:
        c,D,R=audit_data(k,s); batch=f'k{k}_s{s}'
        result[batch]={}
        for method in METHODS:
            original_rows=[]; diagnostic_rows=[]; concentration={}
            for d in DISTRICTS:
                task=T.DistrictTask(D[d],R[d]); er=evaluate.eval_rows(D[d],R[d].known); qs=[e['q'] for e in er]
                P=[]; Q=[]
                for seed in SEEDS:
                    folder=ROOT/batch/method/d/str(seed)
                    ck=torch.load(folder/'model.pt',map_location='cuda',weights_only=True)
                    m=T.ReferenceModel(method).cuda(); m.load_state_dict(ck['state_dict'])
                    dest=folder/'diagnostic.npz'
                    if not dest.exists(): np.savez_compressed(dest,qs=qs,probs=infer(m,task,qs))
                    p=np.ones((D[d].n,D[d].school_n),np.float32)/D[d].school_n
                    q=p.copy(); p[qs]=np.load(folder/'prediction.npz')['probs']; q[qs]=np.load(dest)['probs']
                    P.append(p); Q.append(q)
                avg=np.mean(P,0)
                picks=avg[qs].argmax(1)
                counts=np.bincount(task.rs.cpu().numpy(),minlength=task.S)
                votes=np.bincount(picks,minlength=task.S)
                concentration[d]=dict(most_supported_school=int(counts.argmax()),most_supported_references=int(counts.max()),
                    predicted_most_supported_rate=float((picks==counts.argmax()).mean()),
                    most_predicted_school=int(votes.argmax()),most_predicted_rate=float(votes.max()/len(qs)),
                    available_schools=int((counts>0).sum()),predicted_schools=int((votes>0).sum()))
                original_rows+=evaluate.correct(avg,D[d],er)
                diagnostic_rows+=evaluate.correct(np.mean(Q,0),D[d],er)
            ss=evaluate.summary(diagnostic_rows);ss.pop('两城',None)
            result[batch][method]=dict(diagnostic=ss,original_minus_diagnostic=comparison(original_rows,diagnostic_rows,c['evaluate']),
                                       prediction_concentration=concentration)
    choices={m:{d:dict(lr=ch['lr'],steps=ch['steps'],
                    hit_max_steps=ch['steps']==240,
                    trials=[dict(lr=t['lr'],history=t['history']) for t in ch['trials']],
                    training_counts={n:dict(queries=len(v['qs']),unsupported=v['unsupported']) for n,v in ch['train_pool'].items()})
                 for d,ch in frozen['choices'][m].items()} for m in METHODS}
    result['selection_diagnostics']=choices
    result['diagnostic_registration_sha256']=filehash(DOC/'诊断登记.txt')
    save(ROOT/'diagnostics.json',result);save(DOC/'附加诊断.json',result)
    print(json.dumps({k:v for k,v in result.items() if k.startswith('k')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
