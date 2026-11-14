"""Post-score source-only reference and label-set coverage diagnostics.

Reads source validation labels solely for explanation, never to build target
rules, negatives, references or train weights. No target answer/polygon access.
"""
from collections import Counter, defaultdict
import numpy as np
import experiment_shenzhen_convex_threshold as C
F, R, E = C.F, C.R, C.E
OUT = R.ROOT/'source_spatial_rank_diagnosis_20261113'


def summarize(rows):
    n = len(rows)
    return dict(n=n, rank_separable=sum(r['rank_separable'] for r in rows),
        all_gold_schools_have_reference=sum(r['all_gold_schools_have_reference'] for r in rows),
        all_gold_schools_reference_within_1km=sum(r['all_gold_schools_reference_within_1km'] for r in rows),
        exact_answer_set_seen_in_complete_training=sum(r['exact_answer_set_seen_in_complete_training'] for r in rows),
        exact_reference_set_available=sum(r['exact_reference_set_available'] for r in rows))


def run():
    F.guard(); c = C.verify(); R.check(C.OUT, 'prediction_freeze.json')
    assert F.read(R.ROOT/'round37_independent_completion_audit.json')['all_metrics_independently_recomputed']
    if OUT.exists(): raise FileExistsError('Preserve source diagnosis')
    xs, ys = E.load(c, 'features_labels')
    banks = [dict(np.load(C.OUT/'cache'/f'{s}.npz')) for s in c['seeds']]
    rows=[]; training_distribution={}
    for d in c['sources']:
        x=xs[d]; y=ys[d]; train_ids=banks[0][d+'_train_ids']
        np.testing.assert_array_equal(y['labels'][train_ids],banks[0][d+'_train_y'])
        assert not set(x.groups[train_ids]) & set(x.groups[y['valid_ids']])
        np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        counts=Counter(); groups=defaultdict(set); masses=defaultdict(float)
        weights=np.load(E.OUT/'weights/features_labels'/f'{d}.npz')['weights']
        for q in train_ids:
            key=tuple(np.flatnonzero(y['labels'][q]==1).tolist())
            counts[key]+=1;groups[key].add(str(x.groups[q]));masses[key]+=float(weights[q])
        training_distribution[d]=[dict(schools=list(k),rows=counts[k],property_groups=len(groups[k]),weight_mass=masses[k]) for k in sorted(counts)]
        # The bank stores one school per relation column. Deduplicate property
        # identity before discussing complete reference-set representation.
        ref_columns=defaultdict(list)
        for j,q in enumerate(x.ref_sites):ref_columns[str(x.groups[q])].append(j)
        ref_sets={g:tuple(sorted(set(x.ref_schools[js].tolist()))) for g,js in ref_columns.items()}
        for view,ids in [('train',train_ids),('valid',y['valid_ids'])]:
            z=np.stack([b[d+'_'+view+'_z'].astype(float) for b in banks]).mean(0)
            labels=y['labels'][ids] if view=='train' else y['valid_labels']
            for i,q in enumerate(ids):
                gold=tuple(np.flatnonzero(labels[i]==1).tolist())
                if len(gold)<=1:continue
                valid=~x.self_mask[q]
                dist=x.pair[q,:,x.pair_blocks['distance'].start].astype(float)*300
                assert (dist>=0).all() and np.isfinite(dist).all()
                nearest=[]
                for school in gold:
                    ds=dist[valid&(x.ref_schools==school)]
                    nearest.append(float(ds.min()) if len(ds) else None)
                exact_groups=[g for g,s in ref_sets.items() if s==gold and g!=str(x.groups[q])]
                exact_distance=min((float(dist[ref_columns[g]].min()) for g in exact_groups),default=None)
                high_wrong=int(np.where(labels[i]==0,z[i],-np.inf).argmax())
                low_true=int(np.where(labels[i]==1,z[i],np.inf).argmin())
                seen_groups=groups.get(gold,set())-{str(x.groups[q])}
                rows.append(dict(d=d,view=view,q=int(q),property_group=str(x.groups[q]),gold_schools=list(gold),
                    rank_separable=bool(z[i,low_true]>z[i,high_wrong]),lowest_true_school=low_true,highest_false_school=high_wrong,
                    weakest_true_minus_strongest_false=float(z[i,low_true]-z[i,high_wrong]),
                    nearest_gold_school_reference_m=nearest,
                    all_gold_schools_have_reference=all(v is not None for v in nearest),
                    all_gold_schools_reference_within_1km=all(v is not None and v<=1000 for v in nearest),
                    exact_answer_set_seen_in_complete_training=bool(seen_groups),
                    other_training_property_groups_with_same_set=len(seen_groups),
                    exact_reference_set_available=bool(exact_groups),exact_reference_property_groups=len(exact_groups),
                    nearest_exact_reference_set_m=exact_distance))
    summary={}
    for view in ['train','valid']:
        rs=[r for r in rows if r['view']==view]
        summary[view]={'all':summarize(rs)}
        for label,predicate in [
            ('rank_blocked',lambda r:not r['rank_separable']),
            ('rank_separable',lambda r:r['rank_separable']),
            ('seen_training_set',lambda r:r['exact_answer_set_seen_in_complete_training']),
            ('unseen_training_set',lambda r:not r['exact_answer_set_seen_in_complete_training']),
            ('all_school_refs',lambda r:r['all_gold_schools_have_reference']),
            ('missing_school_ref',lambda r:not r['all_gold_schools_have_reference']),
            ('all_school_refs_within_1km',lambda r:r['all_gold_schools_reference_within_1km'])]:
            summary[view][label]=summarize([r for r in rs if predicate(r)])
        summary[view]['districts']={d:summarize([r for r in rs if r['d']==d]) for d in c['sources']}
    assert summary['train']['all']['n']==94 and summary['valid']['all']['n']==60
    assert summary['train']['all']['rank_separable']==84 and summary['valid']['all']['rank_separable']==28
    F.write(OUT/'rows.json',rows)
    F.write(OUT/'training_labelset_distribution.json',training_distribution)
    F.write(OUT/'summary.json',dict(summary=summary,
        source_only=True,source_validation_labels_used_for_diagnosis=True,not_for_training=True,
        target_gold_read=False,target_predictions_generated=False,
        limitations='Observational source error strata, not causal effects. Reference-set coincidence is representation coverage, not enrollment eligibility. Training label-set identities are not deployment inputs.',
        source_prediction_manifest_sha256=F.sha(C.OUT/'prediction_freeze.json'),code_sha256=F.sha(__file__)))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.iterdir() if p.is_file()},
        sources={str(C.OUT/'input_freeze.json'):F.sha(C.OUT/'input_freeze.json'),str(C.OUT/'prediction_freeze.json'):F.sha(C.OUT/'prediction_freeze.json'),str(__file__):F.sha(__file__)}))
    print('SOURCE_SPATIAL_DIAGNOSIS',summary,flush=True)


if __name__=='__main__':run()
