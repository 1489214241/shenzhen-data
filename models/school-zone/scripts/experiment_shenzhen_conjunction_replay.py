"""Frozen-checkpoint input repair replay. Prepare cannot access target Gold.

No training, new thresholds, new references, policy expansion or geometry.
Compare resident-only round10 inputs to the isolated conjunction-state repair,
using the same exact round08 new_references_labels checkpoints.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch
import experiment_shenzhen_reference_retraining as R
from shenzhen_partial_model import BernoulliRetrieval, predict_sets
from shenzhen_reference_retraining import direct_positive_union
from experiment_shenzhen_partial_labels import probabilities

F = R.F
CONTROL_INPUT = F.REPO/'data/深圳市/居民小组修复模型输入_20261112'
CONTROL = R.ROOT/'round10_resident_replay'
NEW = F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
OUT = R.ROOT/'round12_conjunction_replay'
ARM = 'new_references_labels'


def verify_links():
    for path, digest in F.read(OUT/'input_links.json')['files'].items():
        assert F.sha(path) == digest, path


def decode(values, district, threshold):
    meta = F.read(R.OLD/'inputs'/f'{district}.json')
    sid = [s['school_id'] for s in meta['schools']]
    policy = F.read(R.OUT/'policies'/f'{district}.json')
    graph = F.read(R.OLD/'source_graphs'/f'{district}.json')
    facts = defaultdict(set)
    for fact in F.read(R.OUT/'facts'/f'{district}.json')['facts']:
        facts[fact['site_id']].add(fact['school_id'])
    rows = []
    for q, site in enumerate(meta['sites']):
        positive = [sid[j] for j in meta['known'].get(str(q), [])]
        decision = predict_sets(np.asarray(values[q], np.float64), threshold, sid, policy, graph, site, positive)
        rows.append(dict(district=district, site_id=site['coordinate_group_id'], building_name=site['building_name'],
            after=direct_positive_union(decision, facts[site['coordinate_group_id']]), decision=decision,
            late_source_positives=sorted(facts[site['coordinate_group_id']])))
    return rows


def prepare():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve frozen replay')
    c = R.verify()
    for folder, manifest in [(R.OUT,'probability_freeze.json'), (R.OUT,'prediction_freeze.json'), (NEW,'manifest.json'),
                             (CONTROL,'prediction_freeze.json'),(CONTROL_INPUT,'manifest.json')]:
        R.check(folder,manifest)
    old, _ = R.load(c,ARM)
    for d in R.TARGETS:
        with (CONTROL_INPUT/d/'inputs.pkl').open('rb') as handle: old[d] = pickle.load(handle)
    patched = {}
    for d in R.TARGETS:
        with (NEW/d/'inputs.pkl').open('rb') as handle: patched[d] = pickle.load(handle)
        for key in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            np.testing.assert_array_equal(getattr(old[d],key),getattr(patched[d],key))
    protocol = dict(seeds=c['seeds'], objectives=R.OBJECTIVES, arm=ARM, checkpoint_parent=str(R.OUT),
        intervention='Only reviewed incomplete-boundary conjunction delta; previous resident-group correction preserved in both arms.',
        control_input=str(CONTROL_INPUT),control_predictions=str(CONTROL),
        threshold='Exact frozen round08 source thresholds; no recalibration.', training_steps=0,
        scored_sites=1761, unscored_sites=41, target_gold_read=False, target_polygon_read=False,
        independent_validation=False, adopted=False)
    F.write(OUT/'protocol.json',protocol)
    links = {}
    for folder, manifest in [(NEW,'manifest.json'), (CONTROL_INPUT,'manifest.json'), (CONTROL,'prediction_freeze.json'), (R.OUT,'input_freeze.json'),
                              (R.OUT,'probability_freeze.json'), (R.OUT,'prediction_freeze.json')]:
        links[str(folder/manifest)] = F.sha(folder/manifest)
        for rel, digest in F.read(folder/manifest)['files'].items(): links[str(folder/rel)] = digest
    for path, digest in F.read(R.OUT/'input_links.json')['files'].items(): links[path] = digest
    code = [Path(__file__), F.REPO/'scripts/shenzhen_resident_feature_patch.py']
    for p in code:
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
        links[str(p)] = F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    cal = F.read(R.OUT/'source_calibration'/f'{ARM}.json')['selections']
    F.write(OUT/'thresholds.json',cal)
    audit = {}; corrected = []; original = []
    start = R.M.LinearRetrieval(next(iter(old.values())))
    for seed in c['seeds']:
        model = BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
        ck = torch.load(R.OUT/'checkpoints'/f'{ARM}_{seed}.pt',map_location=R.DEVICE,weights_only=False)
        model.load_state_dict(ck['model']); model.eval()
        frozen = dict(np.load(CONTROL/'probabilities'/f'{seed}.npz'))
        values = {}; audit[str(seed)] = {}
        for d in R.TARGETS:
            ids = np.arange(old[d].n)
            replay = probabilities(model,old[d],ids,c['chunk'])
            np.testing.assert_array_equal(replay,frozen[d])
            values[d] = probabilities(model,patched[d],ids,c['chunk'])
            audit[str(seed)][d] = dict(control_bitwise_equal=True,
                changed_probability_cells=int(np.count_nonzero(values[d]!=replay)),
                max_absolute_change=float(np.abs(values[d]-replay).max()))
        original.append(frozen); corrected.append(values)
        dest = OUT/'probabilities'/f'{seed}.npz'; dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**values)
        print('REPLAY',seed,audit[str(seed)],flush=True)
    mean = {d:np.stack([v[d] for v in corrected]).mean(0) for d in R.TARGETS}
    old_mean = {d:np.stack([v[d] for v in original]).mean(0) for d in R.TARGETS}
    labels = ['ensemble']+[str(s) for s in c['seeds']]
    for label, values, control in zip(labels,[mean]+corrected,[old_mean]+original):
        for obj in R.OBJECTIVES:
            for d in R.TARGETS:
                expected = F.read(CONTROL/'predictions'/obj/label/f'{d}.json')
                assert decode(control[d],d,cal[obj]['threshold']) == expected
                rows = decode(values[d],d,cal[obj]['threshold'])
                assert [r['site_id'] for r in rows] == [r['site_id'] for r in expected]
                F.write(OUT/'predictions'/obj/label/f'{d}.json',rows)
    F.write(OUT/'replay_audit.json',dict(probabilities=audit,control_predictions_exact=True,
        same_checkpoints=True,same_thresholds=True,same_references=True,same_labels=True,same_coordinates=True))
    verify_links()
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',flush=True)


def score():
    from zoning.evaluate import bootstrap
    from experiment_shenzhen_set_decode import metrics
    if (OUT/'results.json').exists(): raise FileExistsError('Preserve scored results')
    verify_links(); R.check(OUT,'prediction_freeze.json')
    c = F.read(OUT/'protocol.json')
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():
        assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json') == h
    gold = {(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')}
    xy = {}
    for d in R.TARGETS:
        with (NEW/d/'district.pkl').open('rb') as handle: district = pickle.load(handle)
        xy.update({(d,s):p for s,p in zip(district.site_ids,district.coords)})
    results = {}; changes = {}
    for obj in c['objectives']:
        results[obj] = {}
        for label in ['ensemble']+[str(s) for s in c['seeds']]:
            rows = []; unscored = []
            for d in R.TARGETS:
                control = {r['site_id']:r for r in F.read(CONTROL/'predictions'/obj/label/f'{d}.json')}
                for row in F.read(OUT/'predictions'/obj/label/f'{d}.json'):
                    key = (d,row['site_id']); g = gold[key]
                    row['before'] = control[row['site_id']]['after']
                    if not g['has_map_gold']: unscored.append(row); continue
                    row['gold'] = set(g['gold']); row['d'] = d
                    row['block'] = str(tuple(int(v//1000) for v in xy[key]))
                    rows.append(row)
            assert len(rows)==1761 and len(unscored)==41
            reports = {}
            for group, rs in [('all',rows)]+[(d,[r for r in rows if r['district']==d]) for d in R.TARGETS]+[
                ('single',[r for r in rows if len(r['gold'])==1]),('multi',[r for r in rows if len(r['gold'])>1])]:
                delta = [int(set(r['after'])==r['gold'])-int(set(r['before'])==r['gold']) for r in rs]
                reports[group] = dict(after=metrics(rs,'after'), before=metrics(rs,'before'),
                    corrected=delta.count(1), harmed=delta.count(-1), delta_pp=float(np.mean(delta)*100),
                    delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
            results[obj][label] = reports
            if label=='ensemble':
                changes[obj] = [dict(r,gold=sorted(r['gold'])) for r in rows if set(r['before'])!=set(r['after'])]
                F.write(OUT/'unscored'/f'{obj}.json',unscored)
        print(obj,results[obj]['ensemble']['all'],flush=True)
    F.write(OUT/'diagnostics_not_training'/'changed_sets.json',changes)
    F.write(OUT/'results.json',dict(reports=results,adopted=False,independent_validation=False))
    F.write(OUT/'completion_integrity.json',dict(checkpoints=10,new_training_steps=0,prediction_files=44,
        source_thresholds_unchanged=True,control_replayed_exactly=True,scored_sites=1761,
        unscored_sites=41,gold_hashes_unchanged=True,target_gold_read_only_after_prediction_freeze=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['prepare','score'])
    globals()[parser.parse_args().phase]()
