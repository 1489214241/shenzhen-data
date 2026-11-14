"""Source-only frozen-score decomposition, not a selected target decoder.

Counterfactual aggregations are diagnostic only. No fitting, calibration, target
inference or policy expansion is performed. Full source validation is retained.
"""
import argparse
from collections import defaultdict
import numpy as np
import torch
import experiment_shenzhen_adaptive_threshold as T
F,R,E=T.F,T.R,T.E
OUT=R.ROOT/'source_aggregation_diagnosis_20261113'
MODES=['native','max_property','weighted_mean','anchor_only','reference_only']


def group_matrix(x):
    keys={};ids=[];schools=[]
    for j,q in enumerate(x.ref_sites):
        key=(str(x.groups[q]),int(x.ref_schools[j]))
        if key not in keys:
            keys[key]=len(keys);schools.append(key[1])
        ids.append(keys[key])
    mat=np.zeros((len(ids),len(keys)),dtype=np.float32)
    mat[np.arange(len(ids)),ids]=1
    return mat,np.asarray(schools,np.int64)


def cache():
    F.guard();c=T.verify();R.check(T.OUT,'cache_freeze.json')
    if (OUT/'score_freeze.json').exists():R.check(OUT,'score_freeze.json');return
    xs,ys=E.load(c,'features_labels')
    for seed in c['seeds']:
        dest=OUT/'scores'/f'{seed}.npz'
        if dest.exists():raise FileExistsError('Inspect any unfrozen partial diagnostic before continuation')
        model=T.scorer(xs,c,seed);old=dict(np.load(T.OUT/'cache'/f'{seed}.npz'));payload={}
        for d in c['sources']:
            x=xs[d];ids=ys[d]['valid_ids'];gm,gs=group_matrix(x)
            gm=torch.from_numpy(gm).to(R.DEVICE);gs=torch.from_numpy(gs).to(R.DEVICE)
            results={mode:[] for mode in MODES}
            with torch.no_grad():
                for lo in range(0,len(ids),c['chunk']):
                    t=R.M.batch(x,ids[lo:lo+c['chunk']]);nr=len(x.ref_sites)
                    pl=t['pair']@model.p+t['gb'][None]+t['px'][:,:,model.np:]@model.pc+model.pnet(t['px']).squeeze(-1)
                    al=t['anchor']@model.a+model.offset+t['ax'][:,:,model.na:]@model.ac+model.anet(t['ax']).squeeze(-1)
                    pl=pl.masked_fill(~t['valid'],-torch.inf)
                    assert not model.cond and not getattr(model,'school_off',False)
                    candidate=torch.cat([pl,al],1);off=candidate.max(1,keepdim=True).values
                    refmass=torch.exp(pl-off)
                    reference=refmass@t['mem'][:nr]
                    native=(torch.exp(candidate-off)@t['mem']).clamp_min(1e-30).log()+off+model.binary_bias
                    torch.testing.assert_close(native,model(t),atol=0,rtol=0)
                    # Sum duplicate coordinates within each property-school before
                    # taking the best property. Thus mere row duplication cannot
                    # act as extra evidence in this counterfactual.
                    group_scores=(refmass@gm).log()+off
                    best=torch.full_like(al,-torch.inf).scatter_reduce(1,gs[None].expand(len(al),-1),group_scores,reduce='amax',include_self=True)
                    max_property=torch.maximum(best,al)+model.binary_bias
                    normalizer=(t['valid'].float()*t['gb'].exp()[None])@t['mem'][:nr]+1
                    weighted_mean=native-normalizer.log()
                    values=dict(native=native,max_property=max_property,weighted_mean=weighted_mean,
                        anchor_only=al+model.binary_bias,reference_only=reference.log()+off+model.binary_bias)
                    for k,v in values.items():results[k].append(v.cpu().numpy())
            for mode,parts in results.items():payload[d+'_'+mode]=np.concatenate(parts)
            np.testing.assert_array_equal(payload[d+'_native'],old[d+'_valid_z'])
            assert all(np.isfinite(payload[d+'_'+mode]).all() for mode in MODES if mode!='reference_only')
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
        print('AGGREGATION_COMPONENTS_FROZEN',seed,flush=True)
    F.write(OUT/'score_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in (OUT/'scores').glob('*.npz')},
        sources={str(T.OUT/'input_freeze.json'):F.sha(T.OUT/'input_freeze.json'),str(T.OUT/'cache_freeze.json'):F.sha(T.OUT/'cache_freeze.json'),str(__file__):F.sha(__file__)},
        target_predictions_generated=False,not_a_target_candidate=True))


def score():
    F.guard();c=T.verify();R.check(OUT,'score_freeze.json')
    if (OUT/'report.json').exists():raise FileExistsError('Preserve scored diagnosis')
    _,ys=E.load(c,'features_labels')
    payloads=[dict(np.load(OUT/'scores'/f'{s}.npz')) for s in c['seeds']]
    reports={};rows=[]
    for idx,seed in enumerate(c['seeds']+['ensemble']):
        result={}
        for d in c['sources']:
            y=ys[d]['valid_labels'].astype(bool);multi=y.sum(1)>1
            scores={m:payloads[idx][d+'_'+m].astype(float) if seed!='ensemble' else np.stack([p[d+'_'+m].astype(float) for p in payloads]).mean(0) for m in MODES}
            eligible={m:np.isfinite(z).any(1) for m,z in scores.items()}
            separated={m:(np.where(y,z,np.inf).min(1)>np.where(~y,z,-np.inf).max(1))&eligible[m] for m,z in scores.items()}
            result[d]={}
            for name,mask in [('all',np.ones(len(y),bool)),('single',~multi),('multi',multi)]:
                result[d][name]={m:dict(n=int(mask.sum()),rank_separable=int(separated[m][mask].sum()),
                    native_separable=int(separated['native'][mask].sum()),
                    corrected=int((separated[m]&~separated['native']&mask).sum()),
                    harmed=int((~separated[m]&separated['native']&mask).sum())) for m in MODES}
            if seed=='ensemble':
                for i,q in enumerate(ys[d]['valid_ids']):
                    if not multi[i]:continue
                    gold=np.flatnonzero(y[i]);native=scores['native'][i]
                    weak=int(gold[native[gold].argmin()]);wrong=int(np.where(~y[i],native,-np.inf).argmax())
                    rows.append(dict(d=d,q=int(q),gold_schools=gold.tolist(),weakest_true_school=weak,highest_false_school=wrong,
                        rank_separable={m:bool(separated[m][i]) for m in MODES},
                        weakest_true_minus_strongest_false={m:float(z[i,weak]-z[i,wrong]) if np.isfinite(z[i,[weak,wrong]]).all() else None for m,z in scores.items()}))
        aggregate={name:{m:{key:sum(result[d][name][m][key] for d in c['sources']) for key in result[c['sources'][0]][name][m]} for m in MODES} for name in ['all','single','multi']}
        reports[str(seed)]=dict(aggregate=aggregate,districts=result)
    assert reports['ensemble']['aggregate']['multi']['native']['rank_separable']==28
    assert reports['ensemble']['aggregate']['single']['native']['rank_separable']==953
    F.write(OUT/'rows.json',rows)
    F.write(OUT/'report.json',dict(reports=reports,source_labels_used_for_postscore_diagnostic=True,
        no_threshold_fit=True,not_for_training=True,target_predictions_generated=False,
        limitations='Counterfactual pooling on weights trained for native sum. Separability uses source labels and is not exact-set accuracy. Neither positive nor negative changes prove a retrained variant effective.',
        score_manifest_sha256=F.sha(OUT/'score_freeze.json'),code_sha256=F.sha(__file__)))
    print('AGGREGATION_DIAGNOSIS',reports['ensemble']['aggregate'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','score']);globals()[p.parse_args().phase]()
