"""Integrate only the resident-group correction into the exact round08 inputs."""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import shenzhen_full_inputs as F
from shenzhen_resident_feature_patch import patch_features

BASE=F.REPO/'data/深圳市/社区归属参照输入_20261112'
BEFORE=F.REPO/'data/深圳市/局部条件修复输入_20261112'
AFTER=F.REPO/'data/深圳市/居民小组单项修复输入_20261112'
OUT=F.REPO/'data/深圳市/居民小组修复模型输入_20261112'


def read_pickle(p):
    with p.open('rb') as f:return pickle.load(f)


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve model inputs')
    for folder in [BASE,BEFORE,AFTER]:
        for rel,h in F.read(folder/'manifest.json')['files'].items():assert F.sha(folder/rel)==h,(folder,rel)
    audit={}
    for d in F.SZ.DISTRICTS:
        x=read_pickle(BASE/d/'inputs.pkl');rules=read_pickle(BASE/d/'rules.pkl');dist=read_pickle(BASE/d/'district.pkl')
        assert dist.answers=={} and dist.train_answers=={}
        old=read_pickle(BEFORE/d/'rules.pkl');new=read_pickle(AFTER/d/'rules.pkl')
        assert old.known==new.known and old.kind==new.kind
        np.testing.assert_array_equal(old.negative,new.negative)
        change=old.statuses!=new.statuses
        # Four previously reviewed school activations are not part of this
        # intervention; carry only the isolated three-school status delta.
        np.testing.assert_array_equal(rules.statuses[change],old.statuses[change])
        patched=rules.statuses.copy();patched[change]=new.statuses[change]
        y=patch_features(x,rules.statuses,patched,rules.negative)
        rr=deepcopy(rules);rr.statuses=patched
        for k in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            np.testing.assert_array_equal(getattr(x,k),getattr(y,k))
        assert rr.known==rules.known and rr.kind==rules.kind
        pair_delta={k:int(np.count_nonzero(x.pair[:,:,s]!=y.pair[:,:,s])) for k,s in x.pair_blocks.items()}
        anchor_delta={k:int(np.count_nonzero(x.anchor[:,:,s]!=y.anchor[:,:,s])) for k,s in x.anchor_blocks.items()}
        assert all(not v for k,v in pair_delta.items() if k not in ['policy','agreement','disagreement'])
        assert all(not v for k,v in anchor_delta.items() if k!='policy')
        assert np.isfinite(y.pair).all() and np.isfinite(y.anchor).all()
        dest=OUT/d;dest.mkdir(parents=True)
        for name in ['labels.npz','control_labels.npz','district.pkl','source_facts.json']:
            shutil.copyfile(BASE/d/name,dest/name)
        for name,value in [('inputs.pkl',y),('rules.pkl',rr)]:
            with (dest/name).open('wb') as f:pickle.dump(value,f,protocol=5)
        meta=F.read(BASE/d/'model_metadata.json');meta['resident_group_scope_repair']=True
        F.write(dest/'model_metadata.json',meta)
        labels=dict(np.load(dest/'labels.npz'));assert not (labels['labels']==0).any()
        audit[d]=dict(status_cells_changed=int(change.sum()),changed_school_indices=np.flatnonzero(change.any((0,2))).tolist(),
            pair_changed_cells=pair_delta,anchor_changed_cells=anchor_delta,reference_sets_unchanged=True,
            labels_unchanged=F.sha(dest/'labels.npz')==F.sha(BASE/d/'labels.npz'),
            coordinates_unchanged=F.sha(dest/'district.pkl')==F.sha(BASE/d/'district.pkl'),
            own_property_mask_unchanged=True,negative_labels=0)
        print(d,audit[d],flush=True)
    F.write(OUT/'input_audit.json',dict(districts=audit,target_gold_read=False,target_polygon_read=False,
        adopted=False,scope='Resident-group status correction only; no new references, labels, administrative guesses or ring changes.'))
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_resident_feature_patch.py',F.REPO/'zoning/evidence.py']:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        parents={str(p/'manifest.json'):F.sha(p/'manifest.json') for p in [BASE,BEFORE,AFTER]},
        target_gold_read=False,target_polygon_read=False,adopted=False))


if __name__=='__main__':main()
