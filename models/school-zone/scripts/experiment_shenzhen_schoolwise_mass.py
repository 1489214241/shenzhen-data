"""Round09: isolate categorical property-mass dilution in Bernoulli retrieval.

Same inputs, observed labels, parent checkpoints, budget and decoder as round08
new_references_labels. Reuse that exact arm as control; fit only two treatments.
No target labels until predictions are frozen. This is development evidence.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import experiment_shenzhen_reference_retraining as E
from shenzhen_schoolwise_reference_mass import batch,schoolwise_bias
from shenzhen_partial_model import BernoulliRetrieval,masked_bce,predict_sets
from shenzhen_balanced_ranking import masked_lsep
from shenzhen_exact_calibration import exact_thresholds
from shenzhen_reference_retraining import direct_positive_union
from experiment_shenzhen_partial_labels import checkpoint,log
from zoning import model as M,DEVICE

F=E.F
OUT=E.ROOT/'round09_schoolwise_mass'
ARMS=['property_mass','school_mass','school_mass_context']
FIT=ARMS[1:]
OBJECTIVES=E.OBJECTIVES
TARGETS=E.TARGETS


def verify():
    E.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    for p in (OUT/'code').rglob('*.py'):
        assert F.sha(F.REPO/p.relative_to(OUT/'code'))==F.sha(p),p
    return F.read(OUT/'protocol.json')


def load(c):return E.load(c,'new_references_labels')


def prepare():
    F.guard();c=E.verify();E.check(E.OUT,'probability_freeze.json');E.check(E.OUT,'prediction_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve round09')
    c=dict(c,round=9,arms=ARMS,date='2026-11-12',
        rationale='Native group_bias divides one property across all observed schools, suppressing independent positive evidence when another school is added.',
        property_mass='Exact frozen round08 new_references_labels control, no extra optimization.',
        school_mass='Deduplicate (property,school) in pair log-mass only; original context unchanged.',
        school_mass_context='Same schoolwise bias plus recomputed six context features using that bias. Context fractions still use cross-school totals.',
        inputs='All arms use identical new-reference inputs and positive-only labels from round08.',
        initialization='Same per-seed round06 balanced_rank checkpoint and fresh Adam as round08.',
        supervision='Identical saved round08 new_references_labels weights and labels across all arms.',
        comparison='Matched round08 control and frozen round07 candidate; both source-only calibration objectives,10seeds.',
        mechanism_limit='No claim that reference count is the only failure or that removing dilution necessarily improves exact sets.',
        prior_target_results_seen=True,adopted=False,independent_validation=False)
    F.write(OUT/'protocol.json',c)
    for folder in ['facts','policies']:
        shutil.copytree(E.OUT/folder,OUT/folder)
    (OUT/'weights').mkdir()
    for d in c['sources']+TARGETS:
        shutil.copyfile(E.OUT/'weights'/f'new_references_labels_{d}.npz',OUT/'weights'/f'{d}.npz')
    links=F.read(E.OUT/'input_links.json')['files'].copy()
    for p in [E.OUT/'input_freeze.json',E.OUT/'probability_freeze.json',E.OUT/'prediction_freeze.json']:
        links[str(p)]=F.sha(p)
    history={};provenance={}
    oldhistory=F.read(E.OUT/'training_history.json')
    for seed in c['seeds']:
        key=f'new_references_labels/{seed}';h=oldhistory[key]
        for folder,srcname,destname in [
            ('checkpoints',f'new_references_labels_{seed}.pt',f'property_mass_{seed}.pt'),
            ('probabilities',f'new_references_labels/{seed}.npz',f'property_mass/{seed}.npz')]:
            src=E.OUT/folder/srcname;dest=OUT/folder/destname
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
            assert F.sha(src)==F.sha(dest);links[str(src)]=F.sha(src)
        history[f'property_mass/{seed}']=h
        provenance[str(seed)]=dict(source_arm=key,checkpoint_sha256=h['checkpoint_sha256'],
                                  probabilities_sha256=h['probabilities_sha256'],additional_steps=0)
    F.write(OUT/'control_provenance.json',provenance)
    F.write(OUT/'input_links.json',dict(files=links))
    xs,ys=load(c);audit={}
    for d,x in xs.items():
        b=schoolwise_bias(x.groups,x.ref_sites,x.ref_schools)
        assert np.all(b>=x.group_bias-1e-6)
        audit[d]=dict(ref_pairs=len(b),changed_pairs=int(np.count_nonzero(abs(b-x.group_bias)>1e-6)),
                      max_log_mass_change=float((b-x.group_bias).max()) if len(b) else 0,
                      positives=int((ys[d]['labels']==1).sum()),negatives=int((ys[d]['labels']==0).sum()))
        if d in TARGETS:assert not (ys[d]['labels']==0).any() and not len(ys[d]['valid_ids'])
    F.write(OUT/'mechanism_input_audit.json',audit)
    shutil.copytree(E.OUT/'code',OUT/'code')
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_schoolwise_reference_mass.py',
              F.REPO/'tests/test_shenzhen_schoolwise_reference_mass.py']:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.relative_to(OUT).parts[0] not in ['checkpoints','probabilities']},target_gold_read=False))
    F.write(OUT/'training_history.json',history)
    log('PREPARED',audit)


def probabilities(model,x,ids,chunk,mode):
    out=[]
    with torch.no_grad():
        for lo in range(0,len(ids),chunk):
            out.append(torch.sigmoid(model(batch(x,ids[lo:lo+chunk],mode))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,x.school_n),np.float32)


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():
        E.check(OUT,'probability_freeze.json');log('MODELS_ALREADY_FROZEN');return
    xs,ys=load(c);frames={a:[] for a in FIT}
    for arm in FIT:
        for d,x in xs.items():
            ids=ys[d]['train_ids'];weights=np.load(OUT/'weights'/f'{d}.npz')['weights']
            for lo in range(0,len(ids),c['chunk']):
                q=ids[lo:lo+c['chunk']];t=batch(x,q,arm)
                t['labels']=torch.from_numpy(ys[d]['labels'][q]).to(DEVICE)
                t['sample_weight']=torch.tensor(weights[q],dtype=torch.float32,device=DEVICE)
                frames[arm].append(t)
    history=F.read(OUT/'training_history.json');ph=F.sha(OUT/'protocol.json')
    start=M.LinearRetrieval(next(iter(xs.values())))
    log('FRAMES_READY',dict(arms=len(FIT),gpu_gb=torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0))
    for seed in c['seeds']:
        parent=torch.load(E.PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=DEVICE,weights_only=False)
        for arm in FIT:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256']
                continue
            model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
            if ck.exists():
                state=torch.load(ck,map_location=DEVICE,weights_only=False);assert state['protocol_hash']==ph
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);begin=state['step'];losses=state['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bc=rank=0.
                for t in frames[arm]:
                    logits=model(t);w=t['sample_weight'];b=masked_bce(logits,t['labels'],w);r=masked_lsep(logits,t['labels'],w)
                    loss=b+c['ranking_coefficient']*r;assert torch.isfinite(loss);loss.backward();bc+=float(b.detach());rank+=float(r.detach())
                norm=float(nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bc,ranking=rank,regularizer=float(reg.detach()),grad_norm=norm))
                if (step+1)%10==0:
                    checkpoint(ck,model,opt,'full',step+1,losses,ph);log('TRAIN',arm,seed,step+1,losses[-1])
            model.eval();payload={}
            for d,x in xs.items():
                ids=np.arange(x.n) if d in TARGETS else ys[d]['valid_ids']
                payload[d]=probabilities(model,x,ids,c['chunk'],arm)
                if d in c['sources']:
                    y=ys[d]['labels'];q=np.flatnonzero((y>=0).all(1)&((y==1).sum(1)>1))
                    payload[d+'_train_multi']=probabilities(model,x,q,c['chunk'],arm)
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest))
            F.write(OUT/'training_history.json',history);log('MODEL_COMPLETE',arm,seed)
    assert len(history)==30
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints']
        for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    log('ALL_MODELS_FROZEN')


def calibrate():
    F.guard();c=verify();E.check(OUT,'probability_freeze.json')
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve predictions')
    _,ys=load(c);summary={}
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in loaded[0]}
        cal=exact_thresholds([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']])
        F.write(OUT/'source_calibration'/f'{arm}.json',cal);summary[arm]=cal['selections']
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],loaded)]:
            for d in TARGETS:
                meta=F.read(E.OLD/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
                policy=F.read(OUT/'policies'/f'{d}.json');graph=F.read(E.OLD/'source_graphs'/f'{d}.json');facts=defaultdict(set)
                for f in F.read(OUT/'facts'/f'{d}.json')['facts']:facts[f['site_id']].add(f['school_id'])
                for obj in OBJECTIVES:
                    rows=[];threshold=cal['selections'][obj]['threshold']
                    for q,site in enumerate(meta['sites']):
                        positive=[sid[j] for j in meta['known'].get(str(q),[])]
                        dec=predict_sets(np.asarray(values[d][q],np.float64),threshold,sid,policy,graph,site,positive)
                        rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],
                            after=direct_positive_union(dec,facts[site['coordinate_group_id']]),decision=dec,
                            late_source_positives=sorted(facts[site['coordinate_group_id']])))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    if arm=='property_mass':
                        assert rows==F.read(E.OUT/'predictions/new_references_labels'/obj/label/f'{d}.json')
        log('CALIBRATED',arm,summary[arm])
    F.write(OUT/'calibration_summary.json',summary)
    files=list((OUT/'predictions').rglob('*.json'))+list((OUT/'source_calibration').glob('*.json'))+[OUT/'calibration_summary.json']
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in files},target_gold_read=False,control_replayed_exactly=True))


def score():
    import pickle
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c=verify();E.check(OUT,'prediction_freeze.json');E.check(E.SUPPLEMENT,'prediction_freeze.json')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scores')
    for d,h in F.read(E.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(E.BASE/'scored_cases.json')};xy={}
    for d in TARGETS:
        with (E.NEW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={};lines=['第九轮：按学校保留同楼盘参照质量','同输入、同预算、同来源校准；开发实验，未采用。','']
    for arm in ARMS:
        results[arm]={}
        for obj in OBJECTIVES:
            reports={}
            for label in ['ensemble']+[str(s) for s in c['seeds']]:
                rows=[];unscored=[]
                for d in TARGETS:
                    control={r['site_id']:r for r in F.read(OUT/'predictions/property_mass'/obj/label/f'{d}.json')}
                    parent={r['site_id']:r for r in F.read(E.SUPPLEMENT/'predictions/balanced_rank'/obj/label/f'{d}.json')}
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r.update(gold=set(g['gold']),before=control[r['site_id']]['after'],parent=parent[r['site_id']]['after'],
                            semantic_pending=g['semantic_pending'],d=d)
                        pt=xy[d,r['site_id']];r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                report={}
                for group,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1])]:
                    item=dict(after=metrics(rs,'after'))
                    for comp in ['before','parent']:
                        delta=[int(set(r['after'])==r['gold'])-int(set(r[comp])==r['gold']) for r in rs]
                        item[comp]=dict(metrics=metrics(rs,comp),corrected=delta.count(1),harmed=delta.count(-1),
                            delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
                    report[group]=item
                reports[label]=report
                if label=='ensemble':
                    F.write(OUT/'diagnostics_not_training'/f'{arm}_{obj}.json',[dict(r,gold=sorted(r['gold'])) for r in rows])
                    F.write(OUT/'unscored'/f'{arm}_{obj}.json',unscored)
            results[arm][obj]=reports;lines.append(arm+'/'+obj)
            for group in ['全部']+TARGETS+['单校','多校']:
                r=reports['ensemble'][group];a=r['after'];b=r['before'];p=r['parent']
                lines.append(f"{group}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 误报/漏报{a['fp']}/{a['fn']}; 对匹配对照改正/损害{b['corrected']}/{b['harmed']},95%差值区间{b['delta_95_pp']}; 对第七轮改正/损害{p['corrected']}/{p['harmed']}")
            log('SCORED',arm,obj,reports['ensemble']['全部'])
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','calibrate','score']);a=p.parse_args()
    dict(prepare=prepare,train=train,calibrate=calibrate,score=score)[a.phase]()
