"""Portable frozen categorical teacher + complete historical policy replay.

Consumes only the recovered 53-input contract and already published code.
Does not import the old acquisition scripts or open target evaluation labels.
"""
import argparse,collections,copy,hashlib,json,pickle,sys
from pathlib import Path,PureWindowsPath

PUBLIC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PUBLIC/'models/school-zone'))
sys.path.insert(0,str(PUBLIC/'models/school-zone/scripts'))
from modern_artifacts import read,sha,check,CONTRACT
import numpy as np
import torch
from zoning import DEVICE,model as MD
from zoning.modernnca_multilabel import MultiLabelModernNCA
from shenzhen_partial_model import predict_sets
from shenzhen_scope_implication import expand_original

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def supplements(root,district,rows):
    base=root/'outputs/shenzhen_resolution_20261112'
    paths={
        'community_round07':base/'round08_reference_retraining/facts'/f'{district}.json',
        'road_corridor':base/'road_corridor_scope_estimates'/district/'estimates.json',
        'residential_community':base/'round20_residential_membership/facts'/f'{district}.json',
        'community_road':base/'round25_joint_geography/facts'/f'{district}.json',
        'named_region':base/'named_region_facts_v1'/district/'facts.json',
    }
    graph=read(base/'round27_scope_graph_refresh/graphs'/f'{district}.json')
    positive=collections.defaultdict(lambda:collections.defaultdict(set))
    negative=collections.defaultdict(set)
    for kind,path in paths.items():
        for fact in read(path)['facts']:positive[fact['site_id']][kind].add(fact['school_id'])
    scope=read(base/'explicit_scope_facts_v1'/district/'facts.json')
    for fact in scope['positive_repairs']:positive[fact['site_id']]['explicit_positive_repair'].add(fact['school_id'])
    for fact in scope['negative_facts']:
        assert not fact['conflicts_with_prior_positive']
        negative[fact['site_id']].add(fact['school_id'])
    for fact in read(base/'administrative_identity_review_v1/school_facts'/f'{district}.json')['facts']:
        assert {'official_community_membership','reviewed_administrative_identity'}<={p['kind'] for p in fact['proof']}
        assert fact['district']==district and not fact['complete_school_set'] and not fact['enrollment_eligibility']
        positive[fact['site_id']]['reviewed_administrative_identity'].add(fact['school_id'])
    result=[];seen=set()
    for original in rows:
        row=copy.deepcopy(original);site=row['site_id'];assert site not in seen;seen.add(site)
        roots=set(row['decision']['raw_school_ids'])|set(row['decision']['positive_lower_bound'])
        before=set(row['after']);assert roots<=before
        facts={k:sorted(v) for k,v in positive[site].items()}
        assert not (set().union(*positive[site].values())&negative[site])
        after=before|set(expand_original(roots,graph))
        for values in facts.values():after.update(values)
        removed=after&negative[site];after-=negative[site];assert after
        row.update(after=sorted(after),cumulative_supplements=dict(original_root_school_ids=sorted(roots),source_positives=facts,
            explicit_exclusions=sorted(negative[site]),removed_school_ids=sorted(removed),added_school_ids=sorted(after-before),
            transitive_expansion=False,complete_school_set=False,enrollment_eligibility=False))
        result.append(row)
    assert set(positive)<=seen and set(negative)<=seen
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();root=args.root.resolve();c=read(CONTRACT)
    assert not check(root,[r for r in c['files'] if r['group']=='k1'])
    expected=read(CONTRACT.parent/'replay_expectations.json')
    # Original absolute paths remain unchanged inside frozen JSON. Resolve them
    # explicitly here, without rewriting the files or relying on an F: drive.
    review=root/'outputs/shenzhen_resolution_20261112/administrative_identity_review_v1'
    m=read(review/'manifest.json')
    for absolute,h in m['sources'].items():
        rel=PureWindowsPath(absolute).relative_to(PureWindowsPath(c['original_root']))
        assert sha(root/Path(*rel.parts))==h
    for rel,h in m['files'].items():assert sha(review/Path(*PureWindowsPath(rel).parts))==h
    if args.out.exists():raise FileExistsError('Preserve replay output')
    args.out.mkdir(parents=True)
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            name=str(values[0]).replace('\\','/')
            if '/Gold/' in name or '/scored_rows/' in name or '/scored_cases.json' in name:
                raise PermissionError('Evaluation labels forbidden in inference')
            if name.startswith(c['original_root'].replace('\\','/')) and not name.startswith(str(root).replace('\\','/')):
                raise PermissionError('Hidden original workspace dependency: '+name)
    sys.addaudithook(guard)
    report=dict(device=DEVICE,torch=torch.__version__,numpy=np.__version__,gold_read=False,training_performed=False,districts={})
    for d in ['龙华区','龙岗区']:
        base=root/'outputs/shenzhen_resolution_20261112'
        current=base/'community_methods_20261113'
        with (current/'inputs/current'/f'{d}.pkl').open('rb') as f:x=pickle.load(f)
        meta=read(current/'metadata'/f'{d}.json');payload=[];seed_checks={}
        for seed in [20261231,20261232,20261233]:
            model=MultiLabelModernNCA(41,27,seed,32).to(DEVICE).eval()
            path=root/'outputs/modernnca_shenzhen_multilabel_20261113/weights/categorical'/f'{seed}.pt'
            model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
            values=[]
            with torch.no_grad():
                for lo in range(0,x.n,64):
                    t=MD.batch(x,np.arange(lo,min(lo+64,x.n)))
                    values.append(model(t).softmax(1).cpu().numpy())
            probs=np.concatenate(values);assert np.isfinite(probs).all();payload.append(probs)
            seed_checks[str(seed)]=hashlib.sha256(probs.tobytes()).hexdigest()==expected['districts'][d]['probabilities'][str(seed)]
        mean=np.stack(payload).mean(0);ids=[s['school_id'] for s in meta['schools']]
        policy=read(base/'round08_reference_retraining/policies'/f'{d}.json')
        graph=read(base/'round03_partial_labels/source_graphs'/f'{d}.json');rows=[]
        for q,site in enumerate(meta['sites']):
            decision=predict_sets(np.asarray(mean[q],np.float64),2.,ids,policy,graph,site,[ids[j] for j in meta['known'].get(str(q),[])])
            rows.append(dict(district=d,site_id=site['coordinate_group_id'],building_name=site['building_name'],after=decision['predicted_school_ids'],decision=decision))
        rows=supplements(root,d,rows)
        full=signature(rows)==expected['districts'][d]['complete_prediction_records_sha256']
        sets=signature([[r['site_id'],r['after']] for r in rows])==expected['districts'][d]['school_sets_sha256']
        (args.out/f'{d}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
        np.savez_compressed(args.out/f'{d}_probabilities.npz',**{str(s):v for s,v in zip([20261231,20261232,20261233],payload)})
        report['districts'][d]=dict(rows=x.n,probability_seeds_bitwise_equal=seed_checks,full_records_equal=full,school_sets_equal=sets)
        print(d,report['districts'][d],flush=True)
    report['passed']=all(all(v['probability_seeds_bitwise_equal'].values()) and v['full_records_equal'] and v['school_sets_equal'] for v in report['districts'].values())
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if not report['passed']:raise SystemExit('Replay mismatch: inspect report, including numerical environment')

if __name__=='__main__':main()
