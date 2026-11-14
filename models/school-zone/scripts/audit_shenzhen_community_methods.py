"""Independent per-file metric audit after all predictions have been frozen."""
from collections import defaultdict
import numpy as np
import experiment_shenzhen_community_methods as C


def main():
    C.verify();C.verify('prediction_freeze.json')
    result=C.F.read(C.OUT/'results.json')
    gold={(r['district'],r['site_id']):r for r in C.F.read(C.ROOT.parent/'shenzhen_aligned_model_20261111/scored_cases.json')}
    labels=['ensemble']+list(map(str,C.SEEDS));correct={};groups_checked=0
    for arm in C.ARMS:
        correct[arm]={}
        for label in labels:
            entries=[]
            for d in C.TARGETS:
                pred=C.F.read(C.OUT/'predictions'/arm/label/f'{d}.json')
                assert len({r['site_id'] for r in pred})==len(pred)
                for r in pred:
                    g=gold[d,r['site_id']]
                    if not g['has_map_gold']:continue
                    t=set(g['gold']);p=set(r['after'])
                    entries.append((d,r['site_id'],len(t),p==t,len(p-t),len(t-p)))
            assert len(entries)==1761
            correct[arm][label]={(d,s):bool(ok) for d,s,_,ok,_,_ in entries}
            groups={'all':entries,**{d:[r for r in entries if r[0]==d] for d in C.TARGETS},
                    'single':[r for r in entries if r[2]==1],'multi':[r for r in entries if r[2]>1]}
            for group,rows in groups.items():
                m=result['reports'][arm][label][group]
                assert (m['n'],m['correct'],m['fp'],m['fn'])==(len(rows),sum(r[3] for r in rows),sum(r[4] for r in rows),sum(r[5] for r in rows))
                groups_checked+=1
    for comparison,stats in result['comparisons'].items():
        a,b=comparison.split('_vs_')
        for label in labels:
            changes=[int(correct[a][label][key])-int(value) for key,value in correct[b][label].items()]
            if label=='ensemble':
                assert sum(v==1 for v in changes)==stats['all']['corrected']
                assert sum(v==-1 for v in changes)==stats['all']['harmed']
            else:assert sum(changes)==stats['seed_delta'][label]
    # Probability-preserving refactor controls must truly be identical per seed.
    for arm in ['changsha_E','changsha_ER']:
        for d in C.TARGETS:
            for s in C.SEEDS:
                path=lambda a:C.OUT/'probabilities'/a/d/f'{s}.npz'
                np.testing.assert_array_equal(np.load(path(arm))['probabilities'],np.load(path('current'))['probabilities'])
    reference_consistency={}
    for d in C.TARGETS:
        meta=C.F.read(C.OUT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']]
        additions=C.F.read(C.OUT/'additions/hangzhou_ER'/f'{d}.json')
        hit=total=unscored=0
        for q,ss in additions.items():
            site=meta['sites'][int(q)]['coordinate_group_id'];g=gold[d,site]
            if not g['has_map_gold']:unscored+=1;continue
            for s in ss:total+=1;hit+=ids[s] in g['gold']
        reference_consistency[d]=dict(map_consistent_positive_relations=hit,scored_positive_relations=total,unscored_sites=unscored,
            warning='Post-score diagnostic only; map consistency is not verified administrative identity.')
    assert result['selected_method']=='current' or all(result['selection_checks'][result['selected_method']].values())
    C.F.write(C.OUT/'score_audit.json',dict(metric_groups=groups_checked,all_counts_recomputed=True,
        all_seed_deltas_recomputed=True,changsha_control_probabilities_identical=True,selected_method=result['selected_method']))
    C.F.write(C.OUT/'diagnostics_not_training/new_reference_map_consistency.json',reference_consistency)
    print('SCORE_AUDIT_PASSED',groups_checked,reference_consistency,flush=True)


if __name__=='__main__':main()
