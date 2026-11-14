"""Independently enumerate the only allowed label and feature cell changes."""
from pathlib import Path
import pickle
import numpy as np
import build_shenzhen_explicit_negative_inputs as B


def load(p):
    with p.open('rb') as f:return pickle.load(f)


def main():
    B.F.guard();dest=B.ROOT/'explicit_negative_input_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    for p,h in B.F.read(B.OUT/'input_links.json')['files'].items():assert B.F.sha(p)==h,p
    for p,h in B.F.read(B.OUT/'manifest.json')['files'].items():assert B.F.sha(B.OUT/p)==h,p
    report={}
    for d in B.F.SZ.DISTRICTS:
        before=load(B.BASE/d/'inputs.pkl');after=load(B.OUT/d/'inputs.pkl')
        rules=load(B.BASE/d/'rules.pkl');changed_rules=load(B.OUT/d/'rules.pkl');dist=load(B.OUT/d/'district.pkl')
        facts=B.F.read(B.FACTS/d/'facts.json')['negative_facts'];pairs={(dist.site_ids.index(f['site_id']),dist.school_ids.index(f['school_id'])) for f in facts}
        ys=dict(np.load(B.BASE/d/'labels.npz'));new=dict(np.load(B.OUT/d/'labels.npz'))
        assert {tuple(v) for v in np.argwhere(ys['labels']!=new['labels'])}==pairs
        for q,j in pairs:assert ys['labels'][q,j]==-1 and new['labels'][q,j]==0
        for k in ys:
            if k not in ['labels','train_ids']:np.testing.assert_array_equal(ys[k],new[k])
        assert new['train_ids'].tolist()==np.flatnonzero((new['labels']>=0).any(1)).tolist()
        feature_pairs={(q,j) for q,j in pairs if not rules.negative[q,j]}
        assert {tuple(v) for v in np.argwhere(changed_rules.negative!=rules.negative)}==feature_pairs
        np.testing.assert_array_equal(changed_rules.statuses,rules.statuses)
        assert changed_rules.known==rules.known and changed_rules.kind==rules.kind
        expected_anchor={(q,j,before.anchor_blocks['policy'].stop-1) for q,j in feature_pairs}
        expected_pair={(q,r,before.pair_blocks['policy'].stop-1) for q,j in feature_pairs for r,s in enumerate(before.ref_schools) if s==j}
        assert {tuple(v) for v in np.argwhere(before.anchor!=after.anchor)}==expected_anchor
        assert {tuple(v) for v in np.argwhere(before.pair!=after.pair)}==expected_pair
        for q,j,k in expected_anchor:assert before.anchor[q,j,k]==0 and after.anchor[q,j,k]==1
        for q,r,k in expected_pair:assert before.pair[q,r,k]==0 and after.pair[q,r,k]==1
        for key in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
        assert B.F.sha(B.BASE/d/'district.pkl')==B.F.sha(B.OUT/d/'district.pkl')
        assert dist.answers=={} and dist.train_answers=={}
        report[d]=dict(negative_labels=len(pairs),new_feature_cells=len(feature_pairs),
            anchor_cells=len(expected_anchor),pair_cells=len(expected_pair),train_rows=len(new['train_ids']))
    B.F.write(dest,dict(districts=report,only_declared_cells_changed=True,all_other_labels_and_features_identical=True,
        training_performed=False,target_gold_read=False,adopted=False,audit_code_sha256=B.F.sha(__file__)))
    print('EXPLICIT_NEGATIVE_INPUTS_VERIFIED',report,flush=True)


if __name__=='__main__':main()
