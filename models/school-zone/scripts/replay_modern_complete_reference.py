"""Replay fitted weights and isolate frozen full-set readout (post-primary ablation).

No new parameter selection. Generate and seal outputs before separate scoring.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import experiment_modern_complete_reference as E

OUT=E.OUT/'frozen_readout_ablation'
def generate():
    E.guard();E.verify()
    E.write(OUT/'protocol.json',dict(prior_primary_results_seen=True,new_fitting=False,new_selection=False,
        purpose='Check checkpoint replay and distinguish complete-set readout from finetuning.',
        comparator='Same initial Modern weights and complete-reference branch, step0; spatial coefficient and all other parameters unchanged.',
        independent_validation=False))
    public=E.read(E.OUT/'public_plan.json');support=E.read(E.OUT/'released_support.json');audit={}
    for d in E.DISTRICTS:
        E.log('REPLAY BUILD',d);data=E.augmented(d,support[d],public[d]['all_groups'])
        query=np.array([r['q'] for r in public[d]['rows'] if not r['selected']]);meta=E.read(E.CURRENT/'metadata'/f'{d}.json')
        predictions={};weights={}
        for arm in ['frozen_complete','complete','categorical']:
            pp=[];zz=[]
            for seed in E.SEEDS:
                model=E.model(seed)
                if arm!='frozen_complete':
                    model.load_state_dict(torch.load(E.OUT/'weights'/d/f'{arm}_{seed}.pt',map_location=E.DEVICE,weights_only=True))
                weights[f'{arm}/{seed}']=float(model.set_log_weight.exp().detach().cpu())
                p,z=E.infer(model,data,query);pp.append(p);zz.append(z)
            p=np.mean(pp,0);z=np.mean(zz,0)
            raw=[(int(i),) for i in p.argmax(1)] if arm=='categorical' else [data['cats'][i] for i in z.argmax(1)]
            predictions[arm+'_raw']=[dict(district=d,site_id=meta['sites'][i]['coordinate_group_id'],building_name=meta['sites'][i]['building_name'],after=sorted(data['school_ids'][s] for s in ss)) for i,ss in zip(query,raw)]
            predictions[arm+'_policy']=E.decode(d,p,raw,data,query)
        original=E.read(E.OUT/'predictions'/f'{d}.json')
        for arm in ['complete','categorical']:
            for suffix in ['raw','policy']:
                now=predictions[arm+'_'+suffix];old=original[arm+'_tuned_'+suffix]
                assert [(r['site_id'],r['after']) for r in now]==[(r['site_id'],r['after']) for r in old],(d,arm,suffix)
        E.write(OUT/'predictions'/f'{d}.json',predictions)
        audit[d]=dict(fitted_predictions_replayed=True,queries=len(query),learned_set_weights=weights)
        E.log('REPLAY OK',d)
    E.write(OUT/'replay_audit.json',audit)
    E.write(OUT/'prediction_freeze.json',{str(p.relative_to(OUT)):E.sha(p) for p in (OUT/'predictions').glob('*.json')})
    E.write(OUT/'code_sha256.json',{str(Path(__file__).relative_to(E.REPO)):E.sha(__file__)})

def score():
    for p,h in E.read(OUT/'prediction_freeze.json').items():assert E.sha(OUT/p)==h
    gold={(r['district'],r['site_id']):set(r['gold']) for r in E.read(E.GOLD)}
    rows={};diagnostics={}
    support=E.read(E.OUT/'released_support.json');public=E.read(E.OUT/'public_plan.json')
    for d in E.DISTRICTS:
        unknown={r['site_id']:r for r in public[d]['rows'] if not r['selected']}
        for arm,pred in E.read(OUT/'predictions'/f'{d}.json').items():
            for r in pred:
                truth=gold[d,r['site_id']];p=set(r['after']);xy=unknown[r['site_id']]['xy']
                rows.setdefault(arm,[]).append(dict(district=d,site_id=r['site_id'],correct=p==truth,tp=len(p&truth),fp=len(p-truth),fn=len(truth-p),
                    block=f'{d}/{int(xy[0]//1000)}/{int(xy[1]//1000)}'))
        possible=0
        for sid,q in unknown.items():
            legal=[set(r['schools']) for r in support[d] if r['group']!=q['group']]
            possible+=gold[d,sid] in legal
        diagnostics[d]=dict(unknown=len(unknown),exact_set_present_in_legal_complete_references=possible,ceiling_percent=100*possible/len(unknown))
    metrics={a:{g:E.metric([r for r in rr if g=='all' or r['district']==g]) for g in ['all']+E.DISTRICTS} for a,rr in rows.items()}
    changes={}
    rng=np.random.default_rng(20261114)
    for suffix in ['raw','policy']:
        fitted=rows['complete_'+suffix];frozen=rows['frozen_complete_'+suffix]
        a=np.array([r['correct'] for r in fitted]);b=np.array([r['correct'] for r in frozen]);delta=a.astype(int)-b.astype(int)
        blocks={d:sorted({r['block'] for r in fitted if r['district']==d}) for d in E.DISTRICTS}
        bi={block:np.array([i for i,r in enumerate(fitted) if r['block']==block]) for bb in blocks.values() for block in bb}
        samples=[]
        for _ in range(2000):
            ix=np.concatenate([bi[k] for d in E.DISTRICTS for k in rng.choice(blocks[d],len(blocks[d]),replace=True)])
            samples.append(100*delta[ix].mean())
        changes[suffix]=dict(corrected=int((a&~b).sum()),harmed=int((~a&b).sum()),delta_pp=100*float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
    result=dict(metrics=metrics,finetuning_minus_frozen=changes,candidate_coverage_post_scoring_only=diagnostics,independent_validation=False)
    E.write(OUT/'results.json',result)
    E.log('FROZEN ABLATION',{a:round(v['all']['exact'],3) for a,v in metrics.items()},changes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['generate','score']);args=p.parse_args();globals()[args.phase]()
