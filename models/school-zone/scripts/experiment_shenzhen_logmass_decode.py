"""Round26: original-logit odds ratios vs sigmoid probability ratios.

No fitting, no target inference; reuse exact round24 checkpoints and source rows.
The scalar control is the sole primary candidate. Vector-model results are a
predeclared diagnostic, not a fallback if the primary candidate fails.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
import experiment_shenzhen_set_evidence as D
from shenzhen_logmass_decode import odds_ratio, probability_ratio
from experiment_shenzhen_relative_decode import relative, select
from shenzhen_exact_calibration import exact_thresholds, calibration_intervals, threshold_metrics
from zoning.evaluate import bootstrap

F=D.F;R=D.R;X=D.X
OUT=R.ROOT/'round26_logmass_decode'
MODES=['probability_ratio','odds_ratio']


def prepare():
    F.guard();c=D.verify();R.check(D.OUT,'probability_freeze.json')
    assert F.read(R.ROOT/'round24_independent_completion_audit.json')['all_source_probabilities_replayed']
    if OUT.exists():raise FileExistsError('Preserve decoder experiment')
    protocol=dict(sources=c['sources'],targets=c['targets'],seeds=c['seeds'],arms=c['arms'],chunk=c['chunk'],
        width=c['width'],embedding_width=c['embedding_width'],source_gate=c['source_gate'],modes=MODES,
        steps=0,weights_changed=False,reference_bank_changed=False,labels_changed=False,
        hypothesis='Sigmoid ratio changes under a shared single-model logit shift even when reference mass ratios do not. Test stable odds ratios from actual logits, without clipped inverse sigmoid.',
        primary_candidate='scalar_control_odds_ratio',strong_control='scalar_control_probability_ratio',
        vector_arm='Predeclared representation interaction diagnostic only; never a fallback for primary failure.',
        ensemble='Preserve arithmetic mean Bernoulli probabilities; stable log-odds of that mixture, then divide odds by the maximum odds. Do not average per-seed ratios.',
        shift_limitation='Exact common-shift invariance applies to each individual model. A heterogeneous Bernoulli mixture is not generally common-shift invariant.',
        primary_objective='balanced_exact',source_selection='Original source gate AND balanced_exact at least strong control AND both frozen missing-support stress counts at least strong control.',
        stress='Reuse original two untrained support partitions with full-source threshold, no stress recalibration.',
        score_semantics='Normalized odds are ranking/selection scores, not calibrated probabilities or eligibility.',
        source_validation='Same1246development rows,1186single60multi; reused, not independent confirmation.',
        target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,target_predictions_generated=False,
        independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',protocol);links=dict(F.read(D.OUT/'input_links.json')['files'])
    for name in ['input_freeze.json','probability_freeze.json']:
        links[str(D.OUT/name)]=F.sha(D.OUT/name)
        links.update({str(D.OUT/p):h for p,h in F.read(D.OUT/name)['files'].items()})
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_logmass_decode.py',F.REPO/'tests/test_shenzhen_logmass_decode.py',
              R.ROOT/'round24_independent_completion_audit.json',D.OUT/'diagnostics_not_training/ranking_and_threshold.json']:
        links[str(p)]=F.sha(p)
        if p.suffix=='.py':
            dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('PREPARED',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def collect():
    F.guard();c=verify()
    if (OUT/'logit_freeze.json').exists():R.check(OUT,'logit_freeze.json');print('ALREADY_COMPLETE');return
    xs,ys=X.load(c,'all_multi_training');start=R.M.LinearRetrieval(next(iter(xs.values())))
    completed=F.read(OUT/'collection_audit.json') if (OUT/'collection_audit.json').exists() else {}
    for arm in c['arms']:
        for seed in c['seeds']:
            key=f'{arm}/{seed}';path=OUT/'logits'/arm/f'{seed}.npz'
            if key in completed:assert F.sha(path)==completed[key]['logits_sha256'];continue
            ck=D.OUT/'checkpoints'/f'{arm}_{seed}.pt';saved=torch.load(ck,map_location=R.DEVICE,weights_only=False)
            model=D.new_model(start,seed,c) if arm=='vector_residual' else R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
            model.load_state_dict(saved['model']);model.eval();old=dict(np.load(D.OUT/'probabilities'/arm/f'{seed}.npz'));payload={};rows=0
            with torch.no_grad():
                for d in c['sources']:
                    for suffix in ['', '_train_multi','_stress_0','_stress_1']:
                        ids=ys[d]['valid_ids'];x=xs[d]
                        if suffix=='_train_multi':
                            ids=ys[d]['train_ids'];z=ys[d]['labels'][ids];ids=ids[(z>=0).all(1)&((z==1).sum(1)>1)]
                        elif suffix:x=D.S.view(x,d,suffix[1:])
                        values=[];replay=[]
                        for lo in range(0,len(ids),c['chunk']):
                            logits=model(R.M.batch(x,ids[lo:lo+c['chunk']]))
                            values.append(logits.cpu().numpy());replay.append(torch.sigmoid(logits).cpu().numpy())
                        payload[d+suffix]=np.concatenate(values) if values else np.zeros((0,x.school_n),np.float32)
                        prob=np.concatenate(replay) if replay else np.zeros((0,x.school_n),np.float32)
                        np.testing.assert_array_equal(prob,old[d+suffix]);rows+=len(prob)
            assert rows==3832 and all(np.isfinite(p).all() for p in payload.values())
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**payload)
            completed[key]=dict(rows=rows,checkpoint_sha256=F.sha(ck),logits_sha256=F.sha(path),original_probabilities_bitwise_replayed=True)
            F.write(OUT/'collection_audit.json',completed);print('COLLECTED',key,flush=True)
    F.write(OUT/'logit_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in (OUT/'logits').rglob('*.npz')},target_predictions_generated=False))


def report():
    F.guard();c=verify();R.check(OUT,'logit_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve report')
    _,ys=X.load(c,'all_multi_training');labels=[ys[d]['valid_labels'] for d in c['sources']]
    reports={};predictions={};rounding={};train_labels=[]
    for d in c['sources']:
        z=ys[d]['labels'][ys[d]['train_ids']];train_labels.append(z[(z>=0).all(1)&((z==1).sum(1)>1)])
    for arm in c['arms']:
        raw=[dict(np.load(OUT/'logits'/arm/f'{s}.npz')) for s in c['seeds']]
        old=[dict(np.load(D.OUT/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        for mode in MODES:
            key=arm+'_'+mode
            scores={k:odds_ratio(np.stack([p[k] for p in raw])) if mode=='odds_ratio' else relative(np.stack([p[k] for p in old]).mean(0)) for k in raw[0]}
            ps=[scores[d] for d in c['sources']];cal=exact_thresholds(ps,labels);v=cal['selections']['balanced_exact'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0])
            passed=v['single_correct']>=g['min_single_correct'] and v['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
                balanced_exact=v['single_correct']/1186/2+v['multi_correct']/60/2,
                train_multi=threshold_metrics([scores[d+'_train_multi'] for d in c['sources']],train_labels,v['threshold']),
                stress={s:threshold_metrics([scores[d+'_'+s] for d in c['sources']],labels,v['threshold']) for s in ['stress_0','stress_1']},
                seeds={str(seed):threshold_metrics([odds_ratio(raw[i][d][None]) if mode=='odds_ratio' else relative(old[i][d]) for d in c['sources']],labels,v['threshold']) for i,seed in enumerate(c['seeds'])})
            predictions[key]=[select(np.asarray(p,np.float64),v['threshold']) for p in ps]
            if mode=='probability_ratio':
                prior=F.read(D.OUT/'source_report.json')['reports'][arm+'_relative']
                assert reports[key]['selections']==prior['selections']
                differences=0;largest=0.;saturated=0
                for d in c['sources']:
                    high=probability_ratio(np.stack([p[d] for p in raw]));low=scores[d]
                    differences+=int((select(high,v['threshold'])!=select(low,v['threshold'])).any(1).sum())
                    largest=max(largest,float(np.max(np.abs(high-low))))
                    saturated+=sum(int(((p[d]==0)|(p[d]==1)).sum()) for p in old)
                rounding[arm]=dict(high_precision_probability_ratio_changed_rows_at_old_threshold=differences,max_score_difference=largest,
                    individual_model_saturated_probability_cells=saturated,used_to_change_control=False)
            F.write(OUT/'calibration'/f'{key}.json',cal)
            print('REPORT',key,{k:v for k,v in reports[key].items() if k!='seeds'},flush=True)
    rows=[]
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            y=labels[di][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(y.sum()>1),
                gold=np.flatnonzero(y).tolist(),predicted={k:np.flatnonzero(v[di][i]).tolist() for k,v in predictions.items()},
                correct={k:bool((v[di][i]==y).all()) for k,v in predictions.items()}))
    paired={}
    for arm in c['arms']:
        a=arm+'_odds_ratio';b=arm+'_probability_ratio';paired[arm]={}
        for name,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rr]
            paired[arm][name]=dict(n=len(rr),before_correct=sum(r['correct'][b] for r in rr),after_correct=sum(r['correct'][a] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    candidate=c['primary_candidate'];baseline=c['strong_control']
    stress_pass=all(reports[candidate]['stress'][s]['correct']>=reports[baseline]['stress'][s]['correct'] for s in ['stress_0','stress_1'])
    selected=candidate if reports[candidate]['source_gate_passed'] and reports[candidate]['balanced_exact']>=reports[baseline]['balanced_exact'] and stress_pass else None
    F.write(OUT/'paired_source_rows.json',rows)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=paired,rounding_audit=rounding,primary_stress_gate_passed=stress_pass,
        selected_target_candidate=selected,target_predictions_generated=False,independent_validation=False,adopted=False))
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','collect','report']);globals()[p.parse_args().phase]()
