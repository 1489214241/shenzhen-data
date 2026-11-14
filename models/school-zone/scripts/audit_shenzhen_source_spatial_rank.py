"""Independent source diagnosis check using raw coordinates and label loops."""
from collections import defaultdict
import numpy as np
import experiment_shenzhen_convex_threshold as C
F,R,E=C.F,C.R,C.E
OUT=R.ROOT/'source_spatial_rank_diagnosis_20261113'


def main():
    F.guard();c=C.verify();R.check(OUT,'manifest.json');R.check(C.OUT,'prediction_freeze.json')
    dest=OUT/'independent_coordinate_audit.json'
    if dest.exists():raise FileExistsError('Preserve coordinate audit')
    xs,ys=E.load(c,'features_labels');records=F.read(OUT/'rows.json')
    banks=[dict(np.load(C.OUT/'cache'/f'{s}.npz')) for s in c['seeds']]
    lookup={};coordinates={};refsets={};trainsets={};exact=[];maxdistanceerror=0.;clipped=0
    for d in c['sources']:
        x=xs[d];coordinates[d]=np.load(F.REPO/'data'/d/'arrays.npz')['coords'];refsets[d]=defaultdict(set);trainsets[d]=defaultdict(set)
        for site,school in zip(x.ref_sites,x.ref_schools):refsets[d][str(x.groups[site])].add(int(school))
        for q in banks[0][d+'_train_ids']:
            truth=frozenset(j for j,label in enumerate(ys[d]['labels'][q]) if label==1)
            trainsets[d][truth].add(str(x.groups[q]))
        for view in ['train','valid']:
            ids=banks[0][d+'_train_ids'] if view=='train' else ys[d]['valid_ids']
            z=np.stack([b[d+'_'+view+'_z'].astype(float) for b in banks]).mean(0)
            y=ys[d]['labels'][ids] if view=='train' else ys[d]['valid_labels']
            for q,scores,labels in zip(ids,z,y):lookup[(d,view,int(q))]=(scores,labels)
    for row in records:
        d,view,q=row['d'],row['view'],row['q'];x=xs[d];scores,labels=lookup[(d,view,q)]
        truth={j for j,label in enumerate(labels) if label==1};assert sorted(truth)==row['gold_schools']
        good=all(scores[p]>scores[n] for p in truth for n in range(len(scores)) if n not in truth)
        assert good==row['rank_separable']
        group=str(x.groups[q]);available=[j for j,site in enumerate(x.ref_sites) if str(x.groups[site])!=group]
        dist=np.linalg.norm(coordinates[d][x.ref_sites]-coordinates[d][q],axis=1)
        actual=[]
        for school in sorted(truth):
            values=[float(dist[j]) for j in available if x.ref_schools[j]==school]
            actual.append(min(values) if values else None)
        for a,b in zip(actual,row['nearest_gold_school_reference_m']):
            assert (a is None)==(b is None)
            if a is not None:
                if a>15000:clipped+=1
                maxdistanceerror=max(maxdistanceerror,abs(min(a,15000)-b))
                assert abs(min(a,15000)-b)<.003
        assert all(v is not None for v in actual)==row['all_gold_schools_have_reference']
        assert all(v is not None and v<=1000 for v in actual)==row['all_gold_schools_reference_within_1km']
        matching=[g for g,schools in refsets[d].items() if schools==truth and g!=group]
        assert len(matching)==row['exact_reference_property_groups']
        seen=trainsets[d].get(frozenset(truth),set())-{group}
        assert len(seen)==row['other_training_property_groups_with_same_set']
        assert bool(seen)==row['exact_answer_set_seen_in_complete_training']
        precise=min((float(dist[j]) for j in available if str(x.groups[x.ref_sites[j]]) in matching),default=None)
        prior=row['nearest_exact_reference_set_m'];assert (precise is None)==(prior is None)
        if precise is not None:assert abs(min(precise,15000)-prior)<.003
        exact.append(dict(d=d,view=view,q=q,rank_separable=good,nearest_gold_reference_m=actual,nearest_exact_set_reference_m=precise))
    quantiles={}
    for view in ['train','valid']:
        for good in [False,True]:
            values=[r['nearest_exact_set_reference_m'] for r in exact if r['view']==view and r['rank_separable']==good and r['nearest_exact_set_reference_m'] is not None]
            quantiles[f'{view}/{good}']=dict(n=len(values),quantiles_min_25_50_75_max=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else [])
    assert len(records)==154
    F.write(dest,dict(rows=154,all_rank_and_reference_strata_independently_verified=True,
        distance_feature_cap_m=15000,clipped_nearest_school_distances=clipped,max_capped_distance_difference_m=maxdistanceerror,
        precise_coordinate_distances=exact,exact_reference_distance_quartiles=quantiles,
        target_gold_read=False,not_for_training=True,code_sha256=F.sha(__file__),source_manifest_sha256=F.sha(OUT/'manifest.json')))
    print('SPATIAL_COORDINATE_AUDIT',quantiles,'CLIPPED',clipped,flush=True)


if __name__=='__main__':main()
