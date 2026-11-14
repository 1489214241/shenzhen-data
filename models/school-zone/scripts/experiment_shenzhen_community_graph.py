"""Round21: community-only spatial holdout, sealed before school prediction."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import shutil
import numpy as np
from scipy.spatial.distance import cdist
import shenzhen_full_inputs as F
from shenzhen_community_graph import graph,infer

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'round21_community_graph'
SOURCES=[ROOT/'community_membership_sources/review_v1/membership_facts.json',
         ROOT/'municipal_residential_membership_sources/review_v1/membership_facts.json']


def verify():
    for p,h in F.read(OUT/'input_freeze.json')['files'].items():assert F.sha(p)==h,p


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve community experiment')
    cfg=dict(round=21,primary='lgc',comparators=['harmonic','nearest'],districts=list(F.SZ.DISTRICTS),
        seed=20261131,spatial_block_m=1000,anchor_buffer_m=100,split_buckets=dict(train=list(range(6)),calibration=[6,7],confirmation=[8,9]),
        graph_k=10,max_edge_m=1500,alpha=.99,confidence_thresholds=[0,.5,.6,.7,.8,.85,.9,.95,.975,.99],
        calibration_gate=dict(min_accepted=15,min_accuracy=.95,min_wilson_lower=.80),
        confirmation_gate=dict(min_accepted=20,min_accuracy=.95,min_wilson_lower=.85,min_blocks=5,min_spatial_lower=.85,min_per_district=5),
        confidence='Normalized top-class mass for graph methods; distance margin to nearest differently labelled anchor for1NN. Not calibrated probabilities.',
        selection='Largest calibration acceptance count meeting gate; tie higher threshold. Primary fixed, no comparator fallback.',
        comparison_gate='Primary at least as many correct as1NN on exactly the primary accepted confirmation points.',
        zero_support='Graph components without anchors unknown; no school-location fallback; never omit unseen-community holdout rows.',
        graph_inputs='Coordinates only, independent official community anchors only; no school labels or polygons or model predictions.',
        class_mass_prior=False,school_predictions_generated=False,target_gold_read=False,independent_validation=False,adopted=False,
        validation_boundary='New module holdout of existing legal community facts. Not untouched overall city data or independently confirmed school accuracy.')
    F.write(OUT/'protocol.json',cfg)
    members=[f for p in SOURCES for f in F.read(p)['facts']]
    assert len(members)==227 and len({(f['district'],f['site_id']) for f in members})==227
    inputs=list(SOURCES);counts={}
    for d in cfg['districts']:
        path=F.SZ.DATA/d/'小区数据/坐标组.json';inputs.append(path);sites=F.read(path)
        if isinstance(sites,dict):raise TypeError('Inspect unexpected coordinate schema before proceeding')
        xy=F.project([s['representative_point_wgs84'] for s in sites]);sid={s['coordinate_group_id']:i for i,s in enumerate(sites)}
        rows=[]
        for m in members:
            if m['district']!=d:continue
            q=sid[m['site_id']];block=tuple(int(v//cfg['spatial_block_m']) for v in xy[q])
            bucket=int(hashlib.sha256(f"{cfg['seed']}|{d}|{block[0]}|{block[1]}".encode()).hexdigest(),16)%10
            split=next(k for k,v in cfg['split_buckets'].items() if bucket in v)
            rows.append(dict(q=q,site_id=m['site_id'],site_names=m['site_names'],identity=[m['street'],m['community']],block=list(block),split=split))
        # Spatial buffer removes nearby support without consulting identities.
        for split,against in [('train',['calibration','confirmation']),('calibration',['confirmation'])]:
            peers=[r['q'] for r in rows if r['split'] in against]
            for row in rows:
                if row['split']==split and peers and cdist(xy[[row['q']]],xy[peers]).min()<=cfg['anchor_buffer_m']:
                    row['excluded_from']=split;row['split']='buffer_excluded'
        F.write(OUT/'inputs'/f'{d}.json',dict(site_ids=[s['coordinate_group_id'] for s in sites],xy=xy.tolist()))
        for split in ['train','calibration','confirmation','buffer_excluded']:
            F.write(OUT/'labels'/split/f'{d}.json',[r for r in rows if r['split']==split])
        counts[d]=dict(population=len(sites),memberships=len(rows),splits=dict(Counter(r['split'] for r in rows)),
            communities=len({tuple(r['identity']) for r in rows}))
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_community_graph.py',F.REPO/'tests/test_shenzhen_community_graph.py']:
        target=OUT/'code'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);inputs.append(p)
    inputs.extend(p for p in OUT.rglob('*') if p.is_file())
    for p in (ROOT/'community_graph_method_review').glob('*'):
        if p.is_file():inputs.append(p)
    F.write(OUT/'population.json',counts)
    inputs.append(OUT/'population.json')
    F.write(OUT/'input_freeze.json',dict(files={str(p):F.sha(p) for p in inputs},target_gold_read=False))
    print('FROZEN',counts,flush=True)


def wilson(correct,n):
    if not n:return 0.
    z=1.959963984540054;p=correct/n
    return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def subset(rows,t):
    return [] if t is None else [r for r in rows if r['supported'] and r['nearest_m']<=1500 and r['confidence']>=t]


def statistics(rows):
    n=len(rows);correct=sum(r['correct'] for r in rows)
    return dict(n=n,correct=correct,accuracy=correct/n if n else None,wilson_lower=wilson(correct,n),
        blocks=len({(r['district'],tuple(r['block'])) for r in rows}),unseen_community=sum(not r['class_has_anchor'] for r in rows))


def spatial_interval(rows):
    if not rows:return [0.,1.]
    rng=np.random.default_rng(20261131);ds=sorted({r['district'] for r in rows});parts=[]
    for d in ds:
        blocks=sorted({tuple(r['block']) for r in rows if r['district']==d})
        pairs=np.array([[sum(r['correct'] for r in rows if r['district']==d and tuple(r['block'])==b),sum(r['district']==d and tuple(r['block'])==b for r in rows)] for b in blocks])
        parts.append(pairs[rng.integers(0,len(pairs),(3000,len(pairs)))].sum(1))
    totals=sum(parts);scores=totals[:,0]/totals[:,1]
    return np.quantile(scores,[.025,.975]).tolist()


def join_labels(split,method,c):
    rows=[]
    for d in c['districts']:
        labels=F.read(OUT/'labels'/split/f'{d}.json');saved=F.read(OUT/'predictions'/f'{d}.json')
        classes={tuple(x) for x in saved['classes']}
        for label in labels:
            pred=saved['methods'][method][label['q']]
            rows.append(dict(label,district=d,**pred,correct=pred['identity']==label['identity'],class_has_anchor=tuple(label['identity']) in classes))
    return rows


def run():
    F.guard();verify()
    if (OUT/'prediction_freeze.json').exists():raise FileExistsError('Preserve community predictions')
    c=F.read(OUT/'protocol.json');ga={}
    for d in c['districts']:
        inp=F.read(OUT/'inputs'/f'{d}.json');xy=np.array(inp['xy']);train=F.read(OUT/'labels/train'/f'{d}.json')
        classes=sorted({tuple(r['identity']) for r in train});idx={v:i for i,v in enumerate(classes)}
        w,ga[d]=graph(xy,k=c['graph_k'],max_edge_m=c['max_edge_m'])
        predictions=infer(w,xy,[r['q'] for r in train],[idx[tuple(r['identity'])] for r in train],classes,alpha=c['alpha'])
        methods={}
        for method,p in predictions.items():
            methods[method]=[dict(identity=list(classes[int(p['predicted'][q])]),confidence=float(p['confidence'][q]),
                nearest_m=float(p['nearest_m'][q]),supported=bool(p['supported'][q])) for q in range(len(xy))]
        F.write(OUT/'predictions'/f'{d}.json',dict(classes=[list(x) for x in classes],methods=methods))
    # Neither calibration nor confirmation labels have been used for inference.
    F.write(OUT/'graph_audit.json',ga)
    F.write(OUT/'prediction_freeze.json',dict(files={str(p):F.sha(p) for p in (OUT/'predictions').glob('*.json')},calibration_labels_used=False,confirmation_labels_used=False))
    calibration={}
    for method in [c['primary']]+c['comparators']:
        rows=join_labels('calibration',method,c);candidates=[];gate=c['calibration_gate']
        for t in c['confidence_thresholds']:
            stats=statistics(subset(rows,t));ok=stats['n']>=gate['min_accepted'] and stats['accuracy']>=gate['min_accuracy'] and stats['wilson_lower']>=gate['min_wilson_lower']
            candidates.append(dict(threshold=t,**stats,passed=ok))
        possible=[r for r in candidates if r['passed']];chosen=max(possible,key=lambda r:(r['n'],r['threshold'])) if possible else None
        calibration[method]=dict(all=statistics(rows),candidates=candidates,selected=chosen)
    F.write(OUT/'calibration.json',calibration)
    F.write(OUT/'selection_freeze.json',dict(calibration_sha256=F.sha(OUT/'calibration.json'),confirmation_labels_used=False,primary=c['primary']))
    reports={};gate=c['confirmation_gate'];thresholds={m:(v['selected']['threshold'] if v['selected'] else None) for m,v in calibration.items()}
    for method in calibration:
        rows=join_labels('confirmation',method,c);accepted=subset(rows,thresholds[method]);stats=statistics(accepted);ci=spatial_interval(accepted)
        per={d:statistics([r for r in accepted if r['district']==d]) for d in c['districts']}
        passed=stats['n']>=gate['min_accepted'] and stats['accuracy']>=gate['min_accuracy'] and stats['wilson_lower']>=gate['min_wilson_lower'] and stats['blocks']>=gate['min_blocks'] and ci[0]>=gate['min_spatial_lower'] and all(v['n']>=gate['min_per_district'] for v in per.values())
        reports[method]=dict(all=statistics(rows),accepted=stats,spatial95=ci,per_district=per,threshold=thresholds[method],gate_passed=passed)
        F.write(OUT/'validation_rows'/f'{method}.json',dict(calibration=join_labels('calibration',method,c),confirmation=rows))
    primary=join_labels('confirmation',c['primary'],c);chosen=subset(primary,thresholds[c['primary']]);nn={(r['district'],r['q']):r for r in join_labels('confirmation','nearest',c)}
    delta=[int(r['correct'])-int(nn[r['district'],r['q']]['correct']) for r in chosen]
    paired=dict(n=len(chosen),corrected=delta.count(1),harmed=delta.count(-1),net_correct=sum(delta))
    eligible=reports[c['primary']]['gate_passed'] and paired['net_correct']>=0
    F.write(OUT/'results.json',dict(reports=reports,paired_nearest_on_primary_accepted=paired,eligible_for_school_experiment=eligible,
        target_gold_read=False,school_predictions_generated=False,independent_validation=False,adopted=False))
    print('COMMUNITY_VALIDATION',reports,paired,'ELIGIBLE',eligible,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);globals()[p.parse_args().phase]()
