"""Read-only descriptive comparison of frozen city predictions and legal references."""
from pathlib import Path
import collections,json,pickle,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zoning import data,rules,predict,changsha_sets
sys.path.insert(0,str(Path(__file__).resolve().parent))
import shenzhen_full_inputs as F

OUT=F.REPO/'outputs/shenzhen_full_model_20261111'


def summarize(rows):
    n=len(rows)
    return dict(n=n,correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/n if n else None,
        single=sum(len(r['gold'])==1 for r in rows),multi=sum(len(r['gold'])>1 for r in rows),
        any_gold_reference=sum(r['any_gold_ref'] for r in rows),all_gold_references=sum(r['all_gold_refs'] for r in rows),
        any_gold_reference_2km=sum(r['any_gold_ref_2km'] for r in rows),
        errors_no_gold_reference=sum(not r['correct'] and not r['any_gold_ref'] for r in rows),
        errors_with_gold_reference=sum(not r['correct'] and r['any_gold_ref'] for r in rows),
        omission_only=sum(set(r['pred'])<set(r['gold']) for r in rows),
        disjoint=sum(not set(r['pred'])&set(r['gold']) for r in rows))


def main():
    cfg=F.read(F.REPO/'configs/default.json');rc=cfg['rules'];all_rows=[];inputs={};chs=[];input_hashes={}
    for d in data.CITIES['长沙']+list(F.SZ.DISTRICTS):
        with (OUT/'inputs'/f'{d}.pkl').open('rb') as f:x=pickle.load(f)
        inputs[d]=dict(sites=x.n,school_n=x.school_n,reference_sites=len(set(x.ref_sites)),
            reference_assignments=len(x.ref_sites),schools_with_references=len(set(x.ref_schools)),
            singleton_training_sites=len(x.supervised))
        if d in data.CITIES['长沙']:
            dist=data.load(d);rr=rules.run(dist,rc['preset'],rc['hangzhou_names'],rc['hangzhou_boundary'],rc['hangzhou_community_rule'])
            p=F.REPO/'outputs/hz_comm_ER/probs'/f'{d}.npz';input_hashes[str(p.relative_to(F.REPO))]=F.sha(p)
            P=np.load(p)['probs'].mean(0)
            # Reproduce current complete-set output using the archived probabilities.
            scored=changsha_sets.score_rows(dist,predict.answers(dist,rr,P));chs+=scored
            base=[dict(q=r['q'],gold=r['truth'],pred=r['pred'],correct=r['exact'],known=r['q'] in rr.known) for r in scored if r['valid']]
            inputs[d]['rule_known_sites']=len(rr.known)
            inputs[d]['known_kind']=dict(collections.Counter(rr.kind.values()))
            inputs[d]['strict_community_sites']=int(dist.community_strict.sum())
            inputs[d]['community_feature_sites']=int((dist.community_school>0).any((1,2)).sum())
            inputs[d]['name_feature_sites']=int((dist.name_match>0).any((1,2)).sum())
        else:
            with (F.OUT/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
            meta=F.read(OUT/'inputs'/f'{d}.json');pos={s['coordinate_group_id']:q for q,s in enumerate(meta['sites'])};sid=dist.school_ids
            base=[dict(q=pos[r['site_id']],gold=[sid.index(s) for s in r['gold']],pred=[sid.index(s) for s in r['after']],
                       correct=set(r['gold'])==set(r['after']),known=str(pos[r['site_id']]) in meta['known'],semantic_pending=r['semantic_pending'])
                  for r in F.read(OUT/'scored_cases.json') if r['district']==d and r['has_map_gold']]
            inputs[d].update(F.read(F.OUT/d/'input_summary.json'))
        distances=np.linalg.norm(dist.coords[:,None]-dist.coords[x.ref_sites][None],axis=-1)
        for r in base:
            q=r['q'];valid=~x.self_mask[q]
            legal=set(x.ref_schools[valid].tolist());near=set(x.ref_schools[valid & (distances[q]<=2000)].tolist());g=set(r['gold'])
            r.update(district=d,city='长沙' if d in data.CITIES['长沙'] else '深圳',
                any_gold_ref=bool(g&legal),all_gold_refs=g<=legal,any_gold_ref_2km=bool(g&near))
            all_rows.append(r)
    saved=F.read(F.REPO/'outputs/hz_comm_ER/eval_recheck.json')['changsha_sets']
    reproduced=changsha_sets.report(chs)
    assert reproduced['overall']==saved['overall'],'Changsha stored scores no longer reproduce'
    cities={}
    for city in ['长沙','深圳']:
        rr=[r for r in all_rows if r['city']==city]
        cities[city]=dict(all=summarize(rr),single=summarize([r for r in rr if len(r['gold'])==1]),
            multi=summarize([r for r in rr if len(r['gold'])>1]),
            model_only=summarize([r for r in rr if not r['known']]),
            single_model_only=summarize([r for r in rr if not r['known'] and len(r['gold'])==1]),
            with_gold_reference=summarize([r for r in rr if r['any_gold_ref']]),
            without_gold_reference=summarize([r for r in rr if not r['any_gold_ref']]))
    sz=cities['深圳'];cs=cities['长沙'];w=cs['multi']['n']/cs['all']['n']
    std=(1-w)*sz['single']['accuracy']+w*sz['multi']['accuracy']
    report=dict(status='descriptive_posthoc_audit_no_tuning',inputs=inputs,cities=cities,
        districts={d:summarize([r for r in all_rows if r['district']==d]) for d in inputs},
        changsha_complete_set_scores_reproduced=True,changsha_scored=3858,changsha_invalid_truth_excluded=5,
        shenzhen_extra_verified_reference_budget=0,
        shenzhen_at_changsha_single_multi_mix=dict(accuracy=std,warning='Arithmetic reweighting only, not a causal effect or a new measured model result'),
        caveats=['Reference availability is audited against held-out map labels only after frozen prediction; never fed back into input.',
                 'Changsha leave-one-district-out training includes three same-city districts; Shenzhen uses eight external-city districts and no other Shenzhen target labels.',
                 'Changsha standard evaluation is point-level; Shenzhen uses deduplicated coordinate groups. Equal metric does not imply equal task distribution.',
                 'Additional Changsha k1/k3 experiments use separate acquisition/query pools and cannot be substituted for this K0 comparison.'],
        source_sha256=input_hashes,diagnostic_code_sha256=F.sha(Path(__file__)))
    F.write(OUT/'深圳与长沙对照诊断.json',report)
    F.write(OUT/'深圳与长沙参照覆盖逐例.json',all_rows)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
