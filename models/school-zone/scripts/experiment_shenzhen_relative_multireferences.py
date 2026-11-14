"""Source-only 3-bank by 2-decoder ablation after round16; no retraining."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import experiment_shenzhen_source_multireferences as X
from experiment_shenzhen_relative_decode import relative,select
from shenzhen_exact_calibration import exact_thresholds,calibration_intervals,threshold_metrics
from zoning.evaluate import bootstrap

F=X.F;R=X.R
OUT=R.ROOT/'round17_relative_multireferences'


def prepare():
    F.guard();c=X.verify();R.check(X.OUT,'probability_freeze.json')
    assert F.read(R.ROOT/'round16_independent_completion_audit.json')['all_source_probabilities_replayed']
    if OUT.exists():raise FileExistsError('Preserve decoder experiment')
    F.write(OUT/'protocol.json',dict(sources=c['sources'],seeds=c['seeds'],banks=X.ARMS,decoders=['absolute','relative'],source_gate=c['source_gate'],
        intervention='Reuse all30 frozen round16 models; compare p>=t with p/max(p)>=alpha, one source-calibrated scalar per branch.',
        primary='balanced_exact; complete-set equality; per-seed same ensemble-calibrated thresholds',
        motivation='Richer source references improved top1 and multi ranking, but absolute decoding lost singleton accuracy.',
        previous_failure='Round15 relative HGB decoder passed source but harmed Shenzhen multi cases. This is a new native-reference input condition, not evidence that relative decoding generally works.',
        selection='Prefer all_multi_training_relative if gate passes and balanced_exact >= control_absolute, else complete_existing_relative under same requirement; no target score used.',
        limitation='No ranking repair; scores normalized by maximum are not calibrated probabilities. Same reused development validation.',
        independent_validation=False,adopted=False,target_gold_read=False,target_cardinality_used=False,target_polygon_read=False))
    links=dict(F.read(X.OUT/'input_links.json')['files'])
    for name in ['input_freeze.json','probability_freeze.json']:
        links[str(X.OUT/name)]=F.sha(X.OUT/name);links.update({str(X.OUT/p):h for p,h in F.read(X.OUT/name)['files'].items()})
    for p in [X.OUT/'source_report.json',R.ROOT/'round16_independent_completion_audit.json']:links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/experiment_shenzhen_relative_decode.py',F.REPO/'scripts/shenzhen_exact_calibration.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links));F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def report():
    F.guard();c=verify()
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve source outcomes')
    _,ys=X.load(c,'control');labels=[ys[d]['valid_labels'] for d in c['sources']];old=F.read(X.OUT/'source_report.json')['reports']
    reports={};predictions={};rows=[]
    for bank in c['banks']:
        loaded=[dict(np.load(X.OUT/'probabilities'/bank/f'{s}.npz')) for s in c['seeds']]
        mean={d:np.stack([p[d] for p in loaded]).mean(0) for d in c['sources']}
        for mode in c['decoders']:
            key=bank+'_'+mode;transform=relative if mode=='relative' else lambda p:p
            ps=[transform(mean[d]) for d in c['sources']];cal=exact_thresholds(ps,labels);chosen=cal['selections']['balanced_exact'];g=c['source_gate']
            attainable=sum(k>1 for _,_,k in calibration_intervals(ps,labels)[0]);assert attainable==old[bank]['attainable_multi']
            if mode=='absolute':assert cal['selections']==old[bank]['selections']
            passed=chosen['single_correct']>=g['min_single_correct'] and chosen['multi_correct']>=g['min_multi_correct'] and attainable>=g['min_attainable_multi']
            reports[key]=dict(selections=cal['selections'],attainable_multi=attainable,source_gate_passed=bool(passed),
                balanced_exact=chosen['single_correct']/1186/2+chosen['multi_correct']/60/2,
                seeds={str(s):threshold_metrics([transform(p[d]) for d in c['sources']],labels,chosen['threshold']) for s,p in zip(c['seeds'],loaded)})
            predictions[key]=[select(np.asarray(p,np.float64),chosen['threshold']) for p in ps]
            F.write(OUT/'calibration'/f'{key}.json',cal);print(key,{k:v for k,v in reports[key].items() if k!='seeds'},flush=True)
    selected=next((k for k in ['all_multi_training_relative','complete_existing_relative'] if reports[k]['source_gate_passed'] and reports[k]['balanced_exact']>=reports['control_absolute']['balanced_exact']),None)
    for di,d in enumerate(c['sources']):
        coords=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            y=labels[di][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in coords[q])),multi=bool(y.sum()>1),gold=np.flatnonzero(y).tolist(),
                predicted={k:np.flatnonzero(v[di][i]).tolist() for k,v in predictions.items()},correct={k:bool((v[di][i]==y).all()) for k,v in predictions.items()}))
    assert len(rows)==1246 and sum(r['multi'] for r in rows)==60
    paired={}
    for bank in c['banks']:
        before=bank+'_absolute';after=bank+'_relative';paired[bank]={}
        for label,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][after])-int(r['correct'][before]) for r in rr]
            paired[bank][label]=dict(n=len(rr),before_correct=sum(r['correct'][before] for r in rr),after_correct=sum(r['correct'][after] for r in rr),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rr,delta,seed=20261104,draws=3000))
    F.write(OUT/'paired_source_rows.json',rows)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=paired,selected_target_candidate=selected,target_predictions_generated=False,adopted=False,independent_validation=False))
    print('NEXT_TARGET_CANDIDATE',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','report']);globals()[p.parse_args().phase]()
