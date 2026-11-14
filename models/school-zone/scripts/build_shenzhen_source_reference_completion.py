"""Freeze source-only reference co-label completion, without new reference sites."""
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment_shenzhen_reference_retraining as R
from shenzhen_source_reference_completion import complete

F=R.F
OUT=F.REPO/'data/深圳市/来源多校参照补全输入_20261112'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve input intervention')
    cfg=R.verify();xs,ys=R.load(cfg,'new_references_labels');audit={}
    F.write(OUT/'protocol.json',dict(sources=cfg['sources'],
        intervention='Add positive co-labels from existing complete SOURCE TRAINING labels at existing reference sites; no new coordinates.',
        exclusions='No validation labels, no target complete labels; old references conflicting with source complete labels are left unchanged and receive no completion.',
        unchanged='All old pair columns and anchor features, training/validation labels, coordinates, learned-border feature values, and target reference bank.',
        new_pair='Reuse same-reference location geometry/crossing/border evidence, change school-policy/agreement fields, named-reference=0.',
        limitation='Source training labels can inform a richer source reference bank, but target references remain partial policy evidence; this may still create a transfer gap.',
        source_labels_already_used_for_training=True,target_gold_read=False,target_polygon_read=False,adopted=False))
    for d in cfg['sources']:
        x=xs[d];y=ys[d];out,added,conflicts=complete(x,y['labels'],y['train_ids'])
        assert not set(x.groups[y['train_ids']])&set(x.groups[y['valid_ids']])
        assert all(q in y['train_ids'] and q in set(x.ref_sites) for q,s in added)
        assert all(y['labels'][q,s]==1 for q,s in added)
        np.testing.assert_array_equal(out.pair[:,:len(x.ref_sites)],x.pair)
        np.testing.assert_array_equal(out.anchor,x.anchor)
        np.testing.assert_array_equal(out.groups,x.groups)
        assert set(out.ref_sites)==set(x.ref_sites)
        assert not any(out.groups[q] in set(x.groups[y['valid_ids']]) for q,s in added)
        np.testing.assert_array_equal(out.self_mask,out.groups[:,None]==out.groups[out.ref_sites][None])
        assert np.isfinite(out.pair).all() and np.isfinite(out.anchor).all()
        dest=OUT/d/'inputs.pkl';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('wb') as handle:pickle.dump(out,handle,protocol=5)
        np.savez_compressed(OUT/d/'labels.npz',**y)
        details=dict(source_training_positive_pairs=[dict(site=q,school=s) for q,s in added],conflicts_held=conflicts)
        F.write(OUT/d/'added_pairs.json',details)
        audit[d]=dict(old_reference_pairs=len(x.ref_sites),new_reference_pairs=len(out.ref_sites),added_positive_pairs=len(added),
            completed_existing_reference_sites=len({q for q,s in added}),conflicting_source_reference_sites_held=len(conflicts),
            old_reference_schools=len(set(x.ref_schools)),new_reference_schools=len(set(out.ref_schools)),
            old_pair_features_exact=True,school_anchors_exact=True,new_reference_locations=0,validation_reference_labels_added=0,
            new_source_labels_added=0,query_own_property_mask_verified=True)
        print(d,audit[d],flush=True)
    # Target inputs are links only, never enriched from target labels.
    target=F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
    R.check(target,'manifest.json')
    links={str(R.OUT/'input_freeze.json'):F.sha(R.OUT/'input_freeze.json'),str(target/'manifest.json'):F.sha(target/'manifest.json')}
    links.update(F.read(R.OUT/'input_links.json')['files'])
    for d in cfg['sources']:
        for p in [R.OLD/'inputs'/f'{d}.pkl',R.OLD/'inputs'/f'{d}_labels.npz']:links[str(p)]=F.sha(p)
    for d in R.TARGETS:
        for p in [target/d/'inputs.pkl',R.NEW/d/'labels.npz']:links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_source_reference_completion.py',F.REPO/'tests/test_shenzhen_source_reference_completion.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links));F.write(OUT/'input_audit.json',audit)
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        target_gold_read=False,target_polygon_read=False,adopted=False))


if __name__=='__main__':main()
