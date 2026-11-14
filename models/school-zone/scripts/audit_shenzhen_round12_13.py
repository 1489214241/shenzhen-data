"""Artifact/input and complete-population scoring verification; never training."""
import pickle
import numpy as np
import experiment_shenzhen_conjunction_replay as A
import experiment_shenzhen_corridor_supplement as B
from shenzhen_resident_feature_patch import patch_features

F=A.F;R=A.R


def load(p):
    with p.open('rb') as handle:return pickle.load(handle)


def compare(rows,gold):
    result=dict(n=0,correct_before=0,correct_after=0,corrected=0,harmed=0,fp=0,fn=0,unscored=0)
    for r in rows:
        g=gold[(r['district'],r['site_id'])]
        if not g['has_map_gold']:result['unscored']+=1;continue
        truth=set(g['gold']);old=set(r['before']);new=set(r['after'])
        before=old==truth;after=new==truth
        result['n']+=1;result['correct_before']+=before;result['correct_after']+=after
        result['corrected']+=after and not before;result['harmed']+=before and not after
        result['fp']+=len(new-truth);result['fn']+=len(truth-new)
    return result


def main():
    A.verify_links();R.check(A.OUT,'prediction_freeze.json');B.verify()
    inputs={}
    for d in R.TARGETS:
        before=load(A.CONTROL_INPUT/d/'inputs.pkl');after=load(A.NEW/d/'inputs.pkl')
        oldrules=load(A.CONTROL_INPUT/d/'rules.pkl');newrules=load(A.NEW/d/'rules.pkl')
        np.testing.assert_array_equal(oldrules.negative,newrules.negative)
        assert oldrules.known==newrules.known and oldrules.kind==newrules.kind
        reproduced=patch_features(before,oldrules.statuses,newrules.statuses,oldrules.negative)
        np.testing.assert_array_equal(reproduced.pair,after.pair);np.testing.assert_array_equal(reproduced.anchor,after.anchor)
        for key in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
        for filename in ['district.pkl','labels.npz','control_labels.npz','source_facts.json']:
            assert F.sha(A.CONTROL_INPUT/d/filename)==F.sha(A.NEW/d/filename)
        assert np.array_equal(after.self_mask,after.groups[:,None]==after.groups[after.ref_sites][None])
        assert not (np.load(A.NEW/d/'labels.npz')['labels']==0).any()
        inputs[d]=dict(status_cells=int(np.count_nonzero(oldrules.statuses!=newrules.statuses)),
            changed_schools=np.flatnonzero((oldrules.statuses!=newrules.statuses).any((0,2))).tolist(),
            pair_cells=int(np.count_nonzero(before.pair!=after.pair)),anchor_cells=int(np.count_nonzero(before.anchor!=after.anchor)),
            all_dependent_features_reproduced_exactly=True,references_labels_coordinates_unchanged=True,own_property_mask_verified=True)
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};assert len(gold)==1802
    replays={};c=F.read(A.OUT/'protocol.json');report=F.read(A.OUT/'results.json')['reports']
    for obj in c['objectives']:
        replays[obj]={}
        for label in ['ensemble']+[str(s) for s in c['seeds']]:
            rows=[]
            for d in R.TARGETS:
                parent={r['site_id']:r for r in F.read(A.CONTROL/'predictions'/obj/label/f'{d}.json')}
                for row in F.read(A.OUT/'predictions'/obj/label/f'{d}.json'):
                    row['before']=parent[row['site_id']]['after'];rows.append(row)
            v=compare(rows,gold);expected=report[obj][label]['all']
            assert v['n']==1761 and v['unscored']==41
            assert (v['correct_after'],v['fp'],v['fn'],v['corrected'],v['harmed'])==(expected['after']['correct'],expected['after']['fp'],expected['after']['fn'],expected['corrected'],expected['harmed'])
            replays[obj][label]=v
    assert len(list((A.OUT/'predictions').rglob('*.json')))==44
    F.write(R.ROOT/'round12_independent_completion_audit.json',dict(inputs=inputs,results=replays,
        original_gold_hashes_unchanged=True,prediction_files=44,control_model_probabilities_exact=F.read(A.OUT/'replay_audit.json')['probabilities'],adopted=False))
    report=F.read(B.OUT/'results.json')['reports'];c=F.read(B.OUT/'protocol.json');supplements={}
    for arm in c['arms']:
        supplements[arm]={}
        for obj in c['objectives']:
            supplements[arm][obj]={}
            for label in c['labels']:
                rows=[]
                for d in R.TARGETS:
                    factset={(r['site_id'],r['school_id']) for r in F.read(B.SOURCE/d/'estimates.json')['facts']}
                    parent={r['site_id']:r for r in F.read(B.parent_predictions(arm,obj,label,d))}
                    for row in F.read(B.OUT/'predictions'/arm/obj/label/f'{d}.json'):
                        assert set(row['before'])==set(parent[row['site_id']]['after'])
                        expected=set(row['before'])|{s for q,s in factset if q==row['site_id']}
                        assert set(row['after'])==expected
                        assert set(row['added_school_ids'])==expected-set(row['before'])
                        rows.append(row)
                v=compare(rows,gold);expected=report[arm][obj][label]['all']
                assert v['n']==1761 and v['unscored']==41
                assert (v['correct_after'],v['fp'],v['fn'],v['corrected'],v['harmed'])==(expected['after']['correct'],expected['after']['fp'],expected['after']['fn'],expected['corrected'],expected['harmed'])
                supplements[arm][obj][label]=v
    assert len(list((B.OUT/'predictions').rglob('*.json')))==88
    F.write(R.ROOT/'round13_independent_completion_audit.json',dict(results=supplements,prediction_files=88,
        exact_source_estimate_union_verified=True,original_gold_hashes_unchanged=True,no_transitive_additions=True,adopted=False))
    print('ROUND12_13_AUDIT_PASSED',inputs,flush=True)


if __name__=='__main__':main()
