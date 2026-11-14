"""Rebuild frozen raw inputs with staged scoped-ring and resident-group fixes.

Retains the reviewed four-school policies and the exact independent coordinates,
roads and administrative geometry. Does not change round09 inputs while it runs.
"""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import shenzhen_full_inputs as F
import build_shenzhen_aligned_inputs as B
import shenzhen_aligned_policy as AP
from shenzhen_resident_group_scope import ResidentGroupParser
from zoning import shenzhen_policy as SP

OLD=F.REPO/'data/深圳市/局部条件修复输入_20261112'
OUT=F.REPO/'data/深圳市/居民小组约束输入_20261112'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve snapshot')
    for rel,h in F.read(OLD/'manifest.json')['files'].items():assert F.sha(OLD/rel)==h
    policies={d:F.read(OLD/d/'reviewed_policy.json') for d in F.SZ.DISTRICTS}
    original_parser,original_load,original_out=AP.Parser,SP.load_policy,B.OUT
    try:
        AP.Parser=ResidentGroupParser;SP.load_policy=lambda d:deepcopy(policies[d]);B.OUT=OUT
        B.main()
    finally:
        AP.Parser,SP.load_policy,B.OUT=original_parser,original_load,original_out
    audit={}
    for d in F.SZ.DISTRICTS:
        F.write(OUT/d/'reviewed_policy.json',policies[d])
        with (OLD/d/'district.pkl').open('rb') as f:before=pickle.load(f)
        with (OUT/d/'district.pkl').open('rb') as f:after=pickle.load(f)
        assert before.site_ids==after.site_ids and before.school_ids==after.school_ids
        np.testing.assert_array_equal(before.coords,after.coords)
        np.testing.assert_array_equal(before.school_xy,after.school_xy)
        assert after.answers=={} and after.train_answers=={}
        old=F.read(OLD/d/'model_metadata.json');new=F.read(OUT/d/'model_metadata.json')
        oldobs=F.read(OLD/d/'policy_observations.json');newobs=F.read(OUT/d/'policy_observations.json')
        changes=[dict(school_id=x['school_id'],school_name=x['school_name'],before=x,after=y)
                 for x,y in zip(oldobs,newobs) if x!=y]
        F.write(OUT/d/'school_observation_changes.json',changes)
        op={(int(q),int(s)) for q,ss in old['known'].items() for s in ss}
        npairs={(int(q),int(s)) for q,ss in new['known'].items() for s in ss}
        a=np.load(OLD/d/'model_arrays.npz');b=np.load(OUT/d/'model_arrays.npz')
        audit[d]=dict(sites=len(after.site_ids),schools=len(after.school_ids),changed_school_observations=len(changes),
            changed_school_names=[r['school_name'] for r in changes],
            changed_array_cells={k:int(np.count_nonzero(a[k]!=b[k])) for k in a.files},
            reference_pairs_added=[dict(site_index=q,school_index=s) for q,s in sorted(npairs-op)],
            reference_pairs_removed=[dict(site_index=q,school_index=s) for q,s in sorted(op-npairs)],
            coordinates_unchanged=True,source_labels_empty=True)
        print(d,audit[d],flush=True)
    F.write(OUT/'input_difference_audit.json',dict(districts=audit,target_gold_read=False,target_polygon_read=False,
        adopted=False,round09_inputs_changed=False,
        limitation='This staged raw snapshot includes previously audited scoped rings plus the resident-group guard; it is not a scored model or confirmed enrollment truth.'))
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_resident_group_scope.py',F.REPO/'scripts/shenzhen_scoped_ring.py',
              F.REPO/'scripts/shenzhen_qualified_scope.py',F.REPO/'tests/test_shenzhen_resident_group_scope.py']:
        dest=OUT/'code'/p.relative_to(F.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    manifest=F.read(OUT/'manifest.json')
    manifest.update(schema='shenzhen_resident_group_scope_input_v1',parent_manifest=str(OLD/'manifest.json'),
        parent_manifest_sha256=F.sha(OLD/'manifest.json'),
        files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    F.write(OUT/'manifest.json',manifest)


if __name__=='__main__':main()
