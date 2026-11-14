"""Rebuild current independent inputs with the reviewed parser; keep old snapshots."""
import pickle
import numpy as np
import shenzhen_full_inputs as F
import build_shenzhen_aligned_inputs as B

OUT=F.REPO/'data/深圳市/修复实验输入_20261112'
OLD=F.REPO/'data/深圳市/规则对齐输入_20261111'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve input snapshot')
    B.OUT=OUT
    B.main()
    audit={}
    for d in ['龙华区','龙岗区']:
        with (OLD/d/'district.pkl').open('rb') as f:old=pickle.load(f)
        with (OUT/d/'district.pkl').open('rb') as f:new=pickle.load(f)
        assert old.site_ids==new.site_ids and old.school_ids==new.school_ids
        np.testing.assert_array_equal(old.coords,new.coords)
        np.testing.assert_array_equal(old.school_xy,new.school_xy)
        assert new.answers=={} and new.train_answers=={}
        before=F.read(OLD/d/'model_metadata.json');after=F.read(OUT/d/'model_metadata.json')
        a=np.load(OLD/d/'model_arrays.npz');b=np.load(OUT/d/'model_arrays.npz')
        diff={k:int(np.count_nonzero(a[k]!=b[k])) for k in a.files}
        audit[d]=dict(array_changed_cells=diff,known_sets_unchanged=before['known']==after['known'],
            known_kind_unchanged=before['known_kind']==after['known_kind'],coordinates_unchanged=True,
            source_labels_empty=True,sites=len(new.site_ids),schools=len(new.school_ids))
    F.write(OUT/'与对齐输入差异.json',dict(districts=audit,gold_read=False))
    print(audit,flush=True)


if __name__=='__main__':main()
