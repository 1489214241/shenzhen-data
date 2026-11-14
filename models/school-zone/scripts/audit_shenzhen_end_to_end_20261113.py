"""Read-only pipeline audit. Post-score diagnostics never become model inputs."""
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / 'outputs/shenzhen_end_to_end_audit_20261113'
ROOT = REPO / 'outputs/shenzhen_resolution_20261112'
OLD = ROOT / 'round03_partial_labels'
CURRENT = ROOT / 'community_methods_20261113'
MODERN = REPO / 'outputs/modernnca_shenzhen_multilabel_20261113'
PARTIAL = REPO / 'outputs/modernnca_partial_sets_20261113'
CELLS = REPO / 'outputs/modernnca_scope_cells_20261113'
RAW = REPO / 'data/深圳市/修复实验输入_20261112'
LATE = REPO / 'data/深圳市/累计文字监督与参照_20261113'
DATA = REPO / 'data/深圳市/标准数据_20261111'
HASHES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    HASHES[str(p)] = sha(p)
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pkl(p):
    HASHES[str(p)] = sha(p)
    return pickle.loads(Path(p).read_bytes())


def arrays(p):
    HASHES[str(p)] = sha(p)
    return dict(np.load(p))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def nodes(x):
    if isinstance(x, dict):
        if 'op' in x:
            yield x
        for v in x.values():
            yield from nodes(v)
    elif isinstance(x, list):
        for v in x:
            yield from nodes(v)


def main():
    transform = Transformer.from_crs(4326, 32650, always_xy=True)
    project = lambda a: np.array(transform.transform(*np.asarray(a, float).T)).T
    inventory = {}
    school_rows = []
    runtime = {}
    protocol = read(MODERN / 'protocol.json')
    for d in protocol['targets']:
        meta = read(CURRENT / 'metadata' / f'{d}.json')
        dist = pkl(RAW / d / 'district.pkl')
        x = pkl(CURRENT / 'inputs/current' / f'{d}.pkl')
        old = pkl(OLD / 'inputs' / f'{d}.pkl')
        y = arrays(OLD / 'inputs' / f'{d}_labels.npz')
        late = arrays(LATE / d / 'labels.npz')
        lx = pkl(LATE / d / 'reference_inputs.pkl')
        ldist = pkl(LATE / d / 'district.pkl')
        assert dist.site_ids == ldist.site_ids and dist.school_ids == ldist.school_ids
        assert [s['school_id'] for s in meta['schools']] == dist.school_ids
        assert [s['coordinate_group_id'] for s in meta['sites']] == dist.site_ids
        assert dist.answers == {} and dist.train_answers == {}
        for field in ['pair', 'anchor', 'ref_sites', 'ref_schools', 'self_mask', 'groups', 'y']:
            np.testing.assert_array_equal(getattr(x, field), getattr(old, field))
        np.testing.assert_array_equal(x.self_mask, x.groups[:, None] == x.groups[x.ref_sites][None])
        assert np.isfinite(dist.coords).all() and np.isfinite(dist.school_xy).all()
        expected = project([s['representative_point_wgs84'] for s in meta['sites']])
        coordinate_error = float(np.max(np.abs(expected - dist.coords)))
        assert coordinate_error < 1e-6
        assert dist.crs == 32650 and meta['projected_crs'] == 32650
        texts = read(DATA / d / 'texts.json')
        assert {t['school_id'] for t in texts} == set(dist.school_ids)
        text_by = {t['school_id']: t for t in texts}
        old_observations = read(RAW / d / 'policy_observations.json')
        reviewed = read(ROOT / 'scope_graph_refresh_review/graphs' / f'{d}_changes.json')['school_inventory']
        assert {t['school_id'] for t in reviewed} == set(dist.school_ids)
        catalogue = read(CELLS / 'catalogues' / f'{d}.json')
        branch_by = {b['school_id']: b for b in catalogue['branch_inventory']}
        known_kind = collections.Counter(meta['known_kind'].values())
        label_counts = lambda lab: dict(positive=int((lab == 1).sum()), negative=int((lab == 0).sum()),
                                        observed_rows=int((lab >= 0).any(1).sum()),
                                        schools_with_positive=int((lab == 1).any(0).sum()))
        changes = dict(new_positive=int(((y['labels'] == -1) & (late['labels'] == 1)).sum()),
                       new_explicit_negative=int(((y['labels'] == -1) & (late['labels'] == 0)).sum()),
                       changed_existing_labels=int(((y['labels'] >= 0) & (y['labels'] != late['labels'])).sum()))
        assert changes['changed_existing_labels'] == 0
        coords_school_meta = [s.get('school_point_wgs84') for s in meta['schools']]
        school_meta_replay = []
        for j, p in enumerate(coords_school_meta):
            if p is not None:
                school_meta_replay.append(float(np.linalg.norm(project([p])[0] - dist.school_xy[j])))
        # Metadata for independently acquired campus anchors can intentionally differ.
        inventory[d] = dict(sites=x.n, schools=x.school_n,
            official_text_matched=sum(bool(t['independent_texts']) for t in texts),
            official_text_years=dict(collections.Counter(str(t['year']) for r in texts for t in r['independent_texts'])),
            current_features_equal_round03=True, input_summary=read(RAW / d / 'input_summary.json'),
            coordinates=dict(projected_crs=dist.crs, all_school_and_site_finite=True,
                             site_projection_max_abs_error_m=coordinate_error,
                             school_metadata_rows=len(school_meta_replay),
                             school_metadata_projection_max_distance_m=max(school_meta_replay)),
            current_labels=label_counts(y['labels']), archived_late_labels=label_counts(late['labels']),
            archived_not_current_differences=changes,
            current_references=dict(pairs=len(x.ref_sites), sites=len(set(x.ref_sites)), schools=len(set(x.ref_schools))),
            archived_late_references=dict(pairs=len(lx.ref_sites), sites=len(set(lx.ref_sites)), schools=len(set(lx.ref_schools))),
            known_kind=dict(known_kind),
            current_schools_by_ast_op=dict(collections.Counter(op for r in old_observations for op in {n['op'] for n in nodes(r['ast'])})),
            reviewed_schools_by_ast_op=dict(collections.Counter(op for r in reviewed for op in {n['op'] for n in nodes(r['ast'])})),
            scope_cell_skip_reasons=dict(collections.Counter(b['skip_reason'] for b in catalogue['branch_inventory'] if b['skip_reason'])),
            target_complete_rows=int((y['labels'] >= 0).all(1).sum()), self_group_mask_verified=True,
            late_candidate_not_adopted=True)
        for r in reviewed:
            j = dist.school_ids.index(r['school_id'])
            b = branch_by[r['school_id']]
            school_rows.append(dict(district=d, school_id=r['school_id'], school_name=r['school_name'],
                source_text=r['source_text'], evidence=r['evidence'], scope_status=r['scope_status'],
                ast_operations=sorted({n['op'] for n in nodes(r['ast'])}),
                unresolved_nodes=[n for n in nodes(r['ast']) if n['op'] in ['unknown', 'conditional', 'ref']],
                current_reference_count=int((x.ref_schools == j).sum()),
                current_positive_count=int((y['labels'][:, j] == 1).sum()),
                current_negative_count=int((y['labels'][:, j] == 0).sum()),
                archived_late_reference_count=int((lx.ref_schools == j).sum()),
                scope_cell_skip_reason=b['skip_reason'], admitted_branches=b['admitted_branches'],
                independent_text_records=text_by[r['school_id']]['independent_texts']))
        runtime[d] = (meta, dist, x, y)
    source_summary = {}
    for d in protocol['sources']:
        x = pkl(OLD / 'inputs' / f'{d}.pkl')
        y = arrays(OLD / 'inputs' / f'{d}_labels.npz')
        lab = y['labels'][y['train_ids']]
        complete = (lab >= 0).all(1)
        source_summary[d] = dict(schools=x.school_n, reference_schools=len(set(x.ref_schools)),
            complete_train=int(complete.sum()), partial_train=int((~complete).sum()),
            complete_train_cardinality=dict(collections.Counter(map(int, (lab[complete] == 1).sum(1)))),
            validation_cardinality=dict(collections.Counter(map(int, y['valid_labels'].sum(1)))))
    inputs = dict(districts=inventory, sources=source_summary,
                  existing_balancing=read(MODERN / 'input_audit.json')['balancing'],
                  partial_set_input_audit=read(PARTIAL / 'input_audit.json'),
                  target_gold_used_in_this_input_inventory=False)
    write(OUT / 'input_inventory.json', inputs)
    write(OUT / 'all_202_school_evidence_routes.json', school_rows)

    # Evaluation-only section. Never export these fields into candidate inputs.
    scored = read(PARTIAL / 'scored_rows/learned_set_readout.json')
    native = read(PARTIAL / 'scored_rows/categorical_policy.json')
    source_only = read(REPO / 'outputs/modernnca_learned_sets_20261113/scored_rows/learned_set_readout.json')
    assert len(scored) == len(native) == len(source_only) == 1761
    report = {}
    details = []
    source_gate = collections.Counter()
    for d in protocol['sources']:
        y = arrays(OLD / 'inputs' / f'{d}_labels.npz')['valid_labels']
        cand = read(PARTIAL / 'source_candidates' / f'{d}.json')
        probs = np.stack([arrays(PARTIAL / 'source_probabilities/learned_set_readout' / f'{s}.npz')[d]
                          for s in protocol['seeds']]).mean(0)
        teacher = np.stack([arrays(MODERN / 'probabilities/categorical' / f'{s}.npz')[d]
                            for s in protocol['seeds']]).mean(0)
        for row, j, k in zip(y, probs.argmax(1), teacher.argmax(1)):
            g = set(np.flatnonzero(row)); pred = set(cand[j]['school_ids'])
            if len(g) != 1:
                continue
            source_gate['single_n'] += 1
            source_gate['teacher_correct'] += k in g
            source_gate['head_correct'] += pred == g
            source_gate['head_multi_output'] += len(pred) > 1
            source_gate['teacher_correct_head_multi'] += k in g and len(pred) > 1
            source_gate['teacher_correct_head_wrong_single'] += k in g and len(pred) == 1 and pred != g
            source_gate['teacher_wrong_head_correct'] += k not in g and pred == g
    for d, (meta, dist, x, y) in runtime.items():
        site_rows = read(CELLS / 'site_candidates' / f'{d}.json')
        bysite = {r['site_id']: r for r in site_rows}
        cat = read(CELLS / 'catalogues' / f'{d}.json')
        school_names = {r['school_id']: r['school_name'] for r in meta['schools']}
        qi = {s: q for q, s in enumerate(dist.site_ids)}
        sj = {s: j for j, s in enumerate(dist.school_ids)}
        counts = collections.Counter()
        refs_by_school = {s: set(map(int, x.ref_sites[x.ref_schools == j])) for s, j in sj.items()}
        for r in scored:
            if r['d'] != d:
                continue
            row = bysite[r['site_id']]
            q = qi[r['site_id']]
            truth = set(r['truth'])
            lower_conflict = set(row['positive_lower_bound']) - truth
            exclusion_conflict = set(row['exclusions']) & truth
            present = any(set(c['school_ids']) == truth for c in row['scope_cell_candidates'])
            assert present == r['candidate_present']
            missing_ref = [s for s in truth if not any(x.groups[k] != x.groups[q] for k in refs_by_school[s])]
            counts['n'] += 1
            counts['correct'] += r['correct']
            counts['absent_candidate'] += not present
            counts['present_wrong'] += present and not r['correct']
            counts['hard_positive_conflict'] += bool(lower_conflict)
            counts['hard_exclusion_conflict'] += bool(exclusion_conflict)
            counts['all_gold_schools_have_other_property_reference'] += not missing_ref
            counts['correct_with_all_references'] += (not missing_ref) and r['correct']
            counts['correct_with_missing_reference'] += bool(missing_ref) and r['correct']
            if len(truth) > 1:
                counts['multi'] += 1
                counts['multi_all_schools_referenced'] += not missing_ref
            reason = ('correct' if r['correct'] else 'correct_candidate_selected_wrong' if present else
                      'hard_positive_conflict' if lower_conflict else 'hard_exclusion_conflict' if exclusion_conflict else
                      'candidate_generation_missing_without_hard_conflict')
            counts['partition_' + reason] += 1
            details.append(dict(r, diagnostic_only_not_training=True, candidate_absence_partition=reason,
                hard_positive_conflict_names=[school_names[s] for s in sorted(lower_conflict)],
                missing_reference_school_names=[school_names[s] for s in sorted(missing_ref)],
                truth_school_names=[school_names[s] for s in r['truth']],
                prediction_school_names=[school_names[s] for s in r['prediction']],
                known_kind=row['known_kind'], positive_evidence_kinds=row['positive_evidence_kinds']))
        candidates = [c for s in site_rows for c in s['scope_cell_candidates']]
        report[d] = dict(counts=counts,
            candidate_applicability=collections.Counter(c['applicability'] for c in candidates),
            all_output_rows=len(site_rows), candidate_total=len(candidates),
            sites_with_any_satisfied_candidate=sum(any(c['applicability'] == 'satisfied' for c in s['scope_cell_candidates']) for s in site_rows),
            conditional_catalogue_records=len(cat['candidates']))
        del site_rows, bysite, candidates
    result = read(PARTIAL / 'results.json')
    assert source_gate['head_correct'] == result['source_ensemble']['learned_set_readout']['single_correct']
    target_single_errors = {}
    for name, rr in [('categorical_policy', native), ('partial_set_head', scored), ('source_only_set_head', source_only)]:
        counter = collections.Counter()
        for row in rr:
            if row['size'] != 1 or row['correct']:
                continue
            pred = set(row['prediction']); truth = set(row['truth'])
            counter['correct_school_plus_extras' if truth < pred else
                    'wrong_single_school' if len(pred) == 1 else 'wrong_multischool_set'] += 1
        target_single_errors[name] = counter
    for d, h in result['gold_sha256'].items():
        assert sha(DATA / d / 'Gold/小区_学校Gold.json') == h
    write(OUT / 'diagnostics_not_training/report.json', dict(districts=report,
        source_single_gate_decomposition=source_gate, target_single_error_types=target_single_errors,
        scored_rows=len(scored), semantic_pending=sum(r['semantic_pending'] for r in scored),
        target_gold_unchanged=True, adopted=False, independent_validation=False))
    write(OUT / 'diagnostics_not_training/rows.json', details)
    for rel in ['scripts/experiment_modernnca_shenzhen_multilabel.py', 'scripts/experiment_shenzhen_community_methods.py',
                'scripts/experiment_modernnca_partial_sets.py', 'zoning/modernnca_set_learning.py',
                'zoning/conditional_school_sets.py', 'zoning/conditional_scope_cells.py', 'zoning/evidence.py',
                'scripts/build_shenzhen_aligned_inputs.py', 'zoning/evaluate.py',
                'scripts/shenzhen_current_supplements.py', 'scripts/shenzhen_reviewed_supplements.py']:
        HASHES[str(REPO / rel)] = sha(REPO / rel)
    HASHES[str(Path(__file__))] = sha(Path(__file__))
    assert all(sha(p) == h for p, h in HASHES.items())
    write(OUT / 'manifest.json', dict(inputs=HASHES, outputs={str(p.relative_to(OUT)): sha(p)
         for p in OUT.rglob('*.json') if p.name != 'manifest.json'}, read_only_pipeline_audit=True))
    print(json.dumps(dict(inputs=inventory, diagnostic=report, source_partial_rows=sum(v['partial_train'] for v in source_summary.values())), ensure_ascii=False), flush=True)
    print('AUDIT COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
