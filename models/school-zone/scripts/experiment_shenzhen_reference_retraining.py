"""Round08: same-parent, same-budget reference and partial-label retraining.

All arms receive identical late source supplementation. Compare unchanged
inputs, new reference bank alone, and reference bank plus positive-only labels.
Separate prepare/train/calibrate/score keeps target labels out of optimization.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch
from torch import nn
import shenzhen_full_inputs as F
from zoning import model as M,DEVICE,shenzhen_policy as SP
from shenzhen_partial_model import BernoulliRetrieval,masked_bce,predict_sets
from shenzhen_balanced_ranking import masked_lsep
from shenzhen_exact_calibration import exact_thresholds
from shenzhen_reference_retraining import fixed_source_weights,direct_positive_union
from experiment_shenzhen_partial_labels import probabilities,checkpoint,log

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OLD=ROOT/'round03_partial_labels'
PARENT=ROOT/'round06_balanced_ranking'
SUPPLEMENT=ROOT/'round07_community_facts'
NEW=F.REPO/'data/深圳市/社区归属参照输入_20261112'
OUT=ROOT/'round08_reference_retraining'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'
TARGETS=['龙华区','龙岗区']
ARMS=['continued_rank','new_references','new_references_labels']
OBJECTIVES=['micro_exact','balanced_exact']


def check(folder,filename):
    for rel,h in F.read(folder/filename)['files'].items():assert F.sha(folder/rel)==h,(folder,rel)


def load(c,arm):
    xs={};ys={}
    for d in c['sources']+TARGETS:
        new=d in TARGETS and arm!='continued_rank'
        with ((NEW/d/'inputs.pkl') if new else (OLD/'inputs'/f'{d}.pkl')).open('rb') as f:xs[d]=pickle.load(f)
        label_path=NEW/d/'labels.npz' if d in TARGETS and arm=='new_references_labels' else OLD/'inputs'/f'{d}_labels.npz'
        ys[d]=dict(np.load(label_path))
    return xs,ys


def verify():
    check(OUT,'input_freeze.json')
    links=F.read(OUT/'input_links.json')
    for path,h in links['files'].items():assert F.sha(path)==h,path
    for p in (OUT/'code').rglob('*.py'):assert F.sha(F.REPO/p.relative_to(OUT/'code'))==F.sha(p),p
    return F.read(OUT/'protocol.json')


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    for p,m in [(OLD,'input_freeze.json'),(PARENT,'probability_freeze.json'),(SUPPLEMENT,'prediction_freeze.json'),(NEW,'manifest.json')]:check(p,m)
    old=F.read(PARENT/'protocol.json')
    c=dict(round=8,date='2026-11-12',sources=old['sources'],targets=TARGETS,seeds=old['seeds'],arms=ARMS,objectives=OBJECTIVES,
        steps=40,lr=.003,reg=.01,width=32,chunk=64,ranking_coefficient=1.,
        initialization='Same round06 balanced_rank checkpoint per seed, fresh Adam, preserve regularization anchor.',
        rationale='Community facts improved direct output by less than1pp; test whether wider reference coverage improves predictions beyond those direct additions.',
        continued_rank='Old reference inputs and original partial labels, matched40extra steps.',
        new_references='New source-positive reference bank, original partial labels, same40steps.',
        new_references_labels='Same new reference bank plus newly observed positive target cells, all other target cells unknown.',
        source_weights='Frozen source complete-single/multi50/50 mass, source row weights identical across arms; original global group denominator retained.',
        target_weights='Each target property keeps original per-group weight; extra positive-only groups add partial BCE mass. No target ranking or negatives.',
        features='41pair/27school; old anchor features retained; new partial refs excluded from learned negative boundary labels, self-property references masked.',
        decoder='Original round03 known positives and graph, same predict_sets; then identical round07 direct positive union for ALL arms. No new-fact transitive expansion.',
        threshold='All source holdout probability breakpoints, source-only micro and balanced exact calibration; same1246source validation rows.',
        comparison='Both matched continued_rank and frozen round07 balanced_rank,10seeds plus ensemble;1761originalscored,41unscored.',
        parent_choice='Previously developed balanced_rank candidate, target scores already seen; explicitly a development experiment, not independent model selection.',
        target_gold_read=False,target_polygons_read=False,target_cardinality_used=False,independent_validation=False,adopted=False,
        partial_sets_are_not_complete=True,adoption='Requires independent confirmation and district/multi/FP/FN/harm review; no automatic adoption.')
    F.write(OUT/'protocol.json',c)
    (OUT/'weights').mkdir(parents=True,exist_ok=True)
    bx,by=load(c,'continued_rank');source_weights=None;weight_audit={};input_audit={}
    for arm in ARMS:
        xs,ys=load(c,arm)
        ws,wa=fixed_source_weights({d:x.groups for d,x in xs.items()},by,ys,c['sources'])
        if source_weights is None:source_weights={d:ws[d].copy() for d in c['sources']}
        for d in c['sources']:
            np.testing.assert_array_equal(ws[d],source_weights[d])
            for k in by[d]:np.testing.assert_array_equal(ys[d][k],by[d][k])
        weight_audit[arm]=wa;input_audit[arm]={}
        for d,x in xs.items():
            assert x.np_==41 and x.na==27
            np.savez_compressed(OUT/'weights'/f'{arm}_{d}.npz',weights=ws[d])
            input_audit[arm][d]=dict(reference_pairs=len(x.ref_sites),reference_schools=len(set(x.ref_schools)),
                train_rows=len(ys[d]['train_ids']),positive_cells=int((ys[d]['labels']==1).sum()),
                negative_cells=int((ys[d]['labels']==0).sum()),validation_rows=len(ys[d]['valid_ids']))
            if d in TARGETS:
                assert not (ys[d]['labels']==0).any() and not len(ys[d]['valid_ids'])
                np.testing.assert_array_equal(x.anchor,bx[d].anchor)
                np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
    for d in TARGETS:
        F.write(OUT/'policies'/f'{d}.json',SP.load_policy(d))
        F.write(OUT/'facts'/f'{d}.json',F.read(SUPPLEMENT/'facts'/f'{d}.json'))
    F.write(OUT/'weight_audit.json',weight_audit);F.write(OUT/'input_audit.json',input_audit)
    files={}
    for folder,m in [(OLD,'input_freeze.json'),(NEW,'manifest.json')]:
        for rel,h in F.read(folder/m)['files'].items():files[str(folder/rel)]=h
    for s in c['seeds']:files[str(PARENT/'checkpoints'/f'balanced_rank_{s}.pt')]=F.sha(PARENT/'checkpoints'/f'balanced_rank_{s}.pt')
    files[str(SUPPLEMENT/'prediction_freeze.json')]=F.sha(SUPPLEMENT/'prediction_freeze.json')
    F.write(OUT/'input_links.json',dict(files=files))
    code=[Path(__file__),F.REPO/'scripts/shenzhen_reference_retraining.py',F.REPO/'scripts/shenzhen_partial_model.py',
        F.REPO/'scripts/shenzhen_balanced_ranking.py',F.REPO/'scripts/shenzhen_exact_calibration.py',
        F.REPO/'scripts/experiment_shenzhen_partial_labels.py',F.REPO/'scripts/experiment_shenzhen_set_decode.py',
        F.REPO/'scripts/shenzhen_scope_implication.py',F.REPO/'tests/test_shenzhen_reference_retraining.py']+list((F.REPO/'zoning').glob('*.py'))
    for p in code:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    log('PREPARED',input_audit)


def frames(c,xs,ys,arm,districts):
    out=[]
    for d in districts:
        ids=ys[d]['train_ids'];weights=np.load(OUT/'weights'/f'{arm}_{d}.npz')['weights']
        for lo in range(0,len(ids),c['chunk']):
            q=ids[lo:lo+c['chunk']];t=M.batch(xs[d],q)
            t['labels']=torch.from_numpy(ys[d]['labels'][q]).to(DEVICE)
            t['sample_weight']=torch.tensor(weights[q],dtype=torch.float32,device=DEVICE);out.append(t)
    return out


def train():
    F.guard();c=verify()
    if (OUT/'probability_freeze.json').exists():
        check(OUT,'probability_freeze.json');log('ALL_MODELS_ALREADY_COMPLETE');return
    inputs={a:load(c,a) for a in ARMS}
    bx,by=inputs['continued_rank'];shared=frames(c,bx,by,'continued_rank',c['sources'])
    batches={a:shared+frames(c,*inputs[a],a,TARGETS) for a in ARMS}
    log('FRAMES', {a:len(v) for a,v in batches.items()}, 'GPU_GB',torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0)
    start=M.LinearRetrieval(next(iter(bx.values())));ph=F.sha(OUT/'protocol.json')
    history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        parent=torch.load(PARENT/'checkpoints'/f'balanced_rank_{seed}.pt',map_location=DEVICE,weights_only=False)
        for arm in ARMS:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'probabilities'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['probabilities_sha256']
                log('MODEL_ALREADY_COMPLETE',key);continue
            model=BernoulliRetrieval(41,27,start,seed,c['width']).to(DEVICE);model.load_state_dict(parent['model'])
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);begin=0;losses=[]
            if ck.exists():
                saved=torch.load(ck,map_location=DEVICE,weights_only=False);assert saved['protocol_hash']==ph
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);begin=saved['step'];losses=saved['history']
            for step in range(begin,c['steps']):
                model.train();opt.zero_grad();reg=model.regularization(c['reg']);reg.backward();bce_total=rank_total=0.
                for t in batches[arm]:
                    logits=model(t);w=t['sample_weight'];bce=masked_bce(logits,t['labels'],w)
                    rank=masked_lsep(logits,t['labels'],w);loss=bce+c['ranking_coefficient']*rank
                    assert torch.isfinite(loss);loss.backward();bce_total+=float(bce.detach());rank_total+=float(rank.detach())
                norm=float(nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
                losses.append(dict(bce=bce_total,ranking=rank_total,regularizer=float(reg.detach()),grad_norm=norm))
                if (step+1)%10==0:
                    checkpoint(ck,model,opt,'full',step+1,losses,ph);log('TRAIN',arm,seed,step+1,losses[-1])
            model.eval();xs,ys=inputs[arm];payload={}
            for d,x in xs.items():
                ids=np.arange(x.n) if d in TARGETS else ys[d]['valid_ids']
                payload[d]=probabilities(model,x,ids,c['chunk'])
                if d in c['sources']:
                    y=ys[d]['labels'];ids=np.flatnonzero((y>=0).all(1)&((y==1).sum(1)>1))
                    payload[d+'_train_multi']=probabilities(model,x,ids,c['chunk'])
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(loss=losses,checkpoint_sha256=F.sha(ck),probabilities_sha256=F.sha(dest))
            F.write(OUT/'training_history.json',history);log('MODEL_COMPLETE',arm,seed)
    F.write(OUT/'probability_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['probabilities','checkpoints'] for p in (OUT/folder).rglob('*') if p.is_file()},target_gold_read=False))
    log('ALL_MODELS_FROZEN')


def calibrate():
    F.guard();c=verify();check(OUT,'probability_freeze.json')
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve calibrated predictions')
    _,ys=load(c,'continued_rank');summary={}
    for arm in ARMS:
        loaded=[dict(np.load(OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([v[d] for v in loaded]).mean(0) for d in loaded[0]}
        cal=exact_thresholds([mean[d] for d in c['sources']],[ys[d]['valid_labels'] for d in c['sources']])
        F.write(OUT/'source_calibration'/f'{arm}.json',cal);summary[arm]=cal['selections']
        for label,values in [('ensemble',mean)]+[(str(s),v) for s,v in zip(c['seeds'],loaded)]:
            for d in TARGETS:
                meta=F.read(OLD/'inputs'/f'{d}.json');sid=[s['school_id'] for s in meta['schools']]
                policy=F.read(OUT/'policies'/f'{d}.json');graph=F.read(OLD/'source_graphs'/f'{d}.json');facts=defaultdict(set)
                for f in F.read(OUT/'facts'/f'{d}.json')['facts']:facts[f['site_id']].add(f['school_id'])
                for obj in OBJECTIVES:
                    rows=[];t=cal['selections'][obj]['threshold']
                    for q,site in enumerate(meta['sites']):
                        positive=[sid[j] for j in meta['known'].get(str(q),[])]
                        decision=predict_sets(np.asarray(values[d][q],np.float64),t,sid,policy,graph,site,positive)
                        after=direct_positive_union(decision,facts[site['coordinate_group_id']])
                        rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],
                            after=after,decision=decision,late_source_positives=sorted(facts[site['coordinate_group_id']])))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
        log('CALIBRATED',arm,summary[arm])
    F.write(OUT/'calibration_summary.json',summary)
    files=list((OUT/'predictions').rglob('*.json'))+list((OUT/'source_calibration').glob('*.json'))+[OUT/'calibration_summary.json']
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in files},target_gold_read=False))


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    c=verify();check(OUT,'prediction_freeze.json');check(SUPPLEMENT,'prediction_freeze.json')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scored results')
    for d,h in F.read(BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(BASE/'scored_cases.json')};xy={}
    for d in TARGETS:
        with (NEW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        xy.update({(d,s):p for s,p in zip(dist.site_ids,dist.coords)})
    results={};lines=['第八轮：新参照与部分正例监督，匹配预算重训',
        '所有方案使用相同第六轮排序模型起点、40步、来源校准与第七轮最终正例补充。开发实验，未采用。','']
    for arm in ARMS:
        results[arm]={}
        for obj in OBJECTIVES:
            reports={}
            for label in ['ensemble']+[str(s) for s in c['seeds']]:
                rows=[];unscored=[]
                for d in TARGETS:
                    control={r['site_id']:r for r in F.read(OUT/'predictions/continued_rank'/obj/label/f'{d}.json')}
                    parent={r['site_id']:r for r in F.read(SUPPLEMENT/'predictions/balanced_rank'/obj/label/f'{d}.json')}
                    for r in F.read(OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        g=gold[d,r['site_id']]
                        if not g['has_map_gold']:unscored.append(r);continue
                        r['gold']=set(g['gold']);r['before']=control[r['site_id']]['after'];r['parent']=parent[r['site_id']]['after']
                        r['semantic_pending']=g['semantic_pending'];r['d']=d;pt=xy[d,r['site_id']]
                        r['block']=str((int(pt[0]//1000),int(pt[1]//1000)));rows.append(r)
                assert len(rows)==1761 and len(unscored)==41
                report={}
                for group,rs in [('全部',rows)]+[(d,[r for r in rows if r['district']==d]) for d in TARGETS]+[
                    ('单校',[r for r in rows if len(r['gold'])==1]),('多校',[r for r in rows if len(r['gold'])>1]),
                    ('语义待核',[r for r in rows if r['semantic_pending']]),('未标记语义待核',[r for r in rows if not r['semantic_pending']])]:
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
                lines.append(f"{group}: {a['correct']}/{a['n']}={a['exact_accuracy']:.2%}; 误报/漏报{a['fp']}/{a['fn']}; 对同预算对照改正/损害{b['corrected']}/{b['harmed']},95%差值区间{b['delta_95_pp']}; 对第七轮改正/损害{p['corrected']}/{p['harmed']}")
            log('SCORED',arm,obj,reports['ensemble']['全部'])
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    (OUT/'结果与限制.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(OUT/'completion_integrity.json',dict(models=30,seeds=10,arms=3,objectives=2,
        scored_sites=1761,unscored_sites=41,gold_unchanged=True,target_labels_not_used_for_training=True,adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','calibrate','score']);a=p.parse_args()
    dict(prepare=prepare,train=train,calibrate=calibrate,score=score)[a.phase]()
