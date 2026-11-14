"""Source-only support/feature and numerical audit of multi-school learning.

Reference-set matches below are diagnostics against source labels only. A
rule-derived reference set is not assumed complete or used as new supervision.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import Counter, defaultdict
import pickle
import numpy as np
import torch
import shenzhen_full_inputs as F
from zoning import model as M, DEVICE
from shenzhen_partial_model import BernoulliRetrieval, school_log_mass

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
PARENT=ROOT/'round03_partial_labels'
MODELS=ROOT/'round06_balanced_ranking'
OUT=ROOT/'multiset_input_and_numeric_audit'


def masses(model,t):
    pl=t['pair']@model.p+t['gb'][None]+t['px'][:,:,model.np:]@model.pc+model.pnet(t['px']).squeeze(-1)
    al=t['anchor']@model.a+model.offset+t['ax'][:,:,model.na:]@model.ac+model.anet(t['ax']).squeeze(-1)
    pl=pl.masked_fill(~t['valid'],-torch.inf)
    candidates=torch.cat([pl,al],1)
    current=school_log_mass(candidates,t['mem'])+model.binary_bias
    exact=torch.stack([torch.logsumexp(candidates[:,t['mem'][:,j].bool()].double(),1)
                       for j in range(t['mem'].shape[1])],1)+model.binary_bias.double()
    return current,exact


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve diagnostic snapshot')
    c=F.read(MODELS/'protocol.json')
    for frozen in ['input_freeze.json','probability_freeze.json']:
        for rel,h in F.read(MODELS/frozen)['files'].items():assert F.sha(MODELS/rel)==h,rel
    xs={};ys={};diagnostics=[];aggregate=defaultdict(Counter)
    for d in c['sources']:
        with (PARENT/'inputs'/f'{d}.pkl').open('rb') as f:x=pickle.load(f)
        z=dict(np.load(PARENT/'inputs'/f'{d}_labels.npz'));xs[d]=x;ys[d]=z
        refs=defaultdict(set)
        for q,s in zip(x.ref_sites,x.ref_schools):refs[int(q)].add(int(s))
        for split in ['train','validation']:
            if split=='train':
                ids=np.flatnonzero((z['labels']>=0).all(1));labels=z['labels'][ids]==1
            else:ids=z['valid_ids'];labels=z['valid_labels']
            for q,gold in zip(ids,labels):
                multi=gold.sum()>1;positive=np.flatnonzero(gold);available=~x.self_mask[q]
                supported=set(map(int,x.ref_schools[available]));no_ref=[int(s) for s in positive if s not in supported]
                near=set(map(int,x.ref_schools[available & (x.pair[q,:,x.pair_blocks['distance'].start]<=2000/300)]))
                sets=[ss for r,ss in refs.items() if x.groups[r]!=x.groups[q]]
                exact_ref=any(ss==set(positive) for ss in sets)
                policy=x.anchor[q,positive,x.anchor_blocks['policy']]
                known_side=(policy[:,:12].reshape(-1,4,3)[:,:,1:]!=0).any((1,2))
                named=(x.anchor[q,positive,x.anchor_blocks['name']]!=0).any(1)
                a=Counter(n=1,positive_schools=len(positive),positive_no_reference=len(no_ref),
                    rows_with_missing_positive_reference=bool(no_ref),positive_no_reference_2km=sum(s not in near for s in positive),
                    rows_with_exact_estimated_reference_set=exact_ref,positive_has_known_policy_side=int(known_side.sum()),
                    positive_has_name_feature=int(named.sum()))
                for group in ['all',d]:aggregate[(group,split,'multi' if multi else 'single')].update(a)
                if multi:
                    diagnostics.append(dict(district=d,split=split,query_index=int(q),positive_school_indices=positive.tolist(),
                        missing_reference_school_indices=no_ref,reference_set_match=exact_ref,
                        policy_side_known=known_side.tolist(),name_feature_present=named.tolist()))
    summary={'/'.join(k):dict(v) for k,v in aggregate.items()}
    F.write(OUT/'source_reference_features.json',dict(summary=summary,multi_rows=diagnostics,target_gold_read=False,
        warning='Source-label diagnosis only; reference sets may be partial and are not promoted to full labels.'))
    numeric={};start=M.LinearRetrieval(next(iter(xs.values())))
    # Fixed first protocol seed, all three arms; this tests arithmetic, not seed selection.
    seed=c['seeds'][0]
    for arm in c['arms']:
        model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE)
        state=torch.load(MODELS/'checkpoints'/f'{arm}_{seed}.pt',map_location=DEVICE,weights_only=False)
        model.load_state_dict(state['model']);model.eval();stats=Counter();maxerror=0.;poserrors=[]
        for d,x in xs.items():
            z=ys[d];y=z['labels'];train=np.flatnonzero((y>=0).all(1)&((y==1).sum(1)>1))
            valid=z['valid_ids'][z['valid_labels'].sum(1)>1]
            ids=np.r_[train,valid];labels=np.r_[y[train]==1,z['valid_labels'][z['valid_labels'].sum(1)>1]]
            with torch.no_grad():
                for lo in range(0,len(ids),64):
                    t=M.batch(x,ids[lo:lo+64]);actual,exact=masses(model,t)
                    actual=actual.cpu().numpy();exact=exact.cpu().numpy();gold=labels[lo:lo+64]
                    error=np.abs(actual-exact);stats['rows']+=len(actual);stats['positive_cells']+=int(gold.sum())
                    stats['all_cells_error_over_1e_3']+=int((error>1e-3).sum())
                    stats['positive_cells_error_over_1e_3']+=int(((error>1e-3)&gold).sum())
                    stats['positive_sigmoid_saturated_high']+=int(((actual>16.64)&gold).sum())
                    stats['positive_sigmoid_saturated_low']+=int(((actual<-88)&gold).sum())
                    maxerror=max(maxerror,float(error.max()));poserrors.extend(error[gold].tolist())
        numeric[arm]=dict(seed=seed,**stats,max_abs_logit_error=maxerror,max_positive_logit_error=max(poserrors))
        print('NUMERIC',arm,numeric[arm],flush=True)
    F.write(OUT/'numeric_comparison.json',dict(reports=numeric,reference='float64 per-school logsumexp, before sigmoid',target_gold_read=False))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        code_sha256=F.sha(__file__),input_and_models_verified=True,target_gold_read=False))
    for k,v in summary.items():
        if k.startswith('all/'):print(k,v,flush=True)


if __name__=='__main__':main()
