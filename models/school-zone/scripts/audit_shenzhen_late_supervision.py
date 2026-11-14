"""Inventory cumulative facts not yet present in round32 training labels.

This is a source-only preflight. No labels, references, model weights, target
predictions or thresholds are changed. Geometry estimates remain distinguished
from textual proofs; neither defines a complete school set.
"""
from pathlib import Path
import pickle
import numpy as np
import shenzhen_current_supplements as C
import experiment_shenzhen_explicit_negative_training as E

F, R = E.F, E.R
OUT = R.ROOT/'late_supervision_inventory_20261113'
TEXTUAL = {'community_round07', 'residential_community', 'named_region', 'explicit_positive_repair'}
GEOGRAPHIC = {'road_corridor', 'community_road'}


def main():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve late-fact inventory')
    R.check(E.B.OUT, 'manifest.json')
    links = {}; summary = {}
    for d in R.TARGETS:
        meta_path = R.OLD/'inputs'/f'{d}.json'
        label_path = E.B.OUT/d/'labels.npz'; input_path = E.B.OUT/d/'inputs.pkl'
        meta = F.read(meta_path); label_file = dict(np.load(label_path)); y = label_file['labels']
        with input_path.open('rb') as h: x = pickle.load(h)
        sites = meta['sites']; schools = meta['schools']
        si = {s['coordinate_group_id']: i for i, s in enumerate(sites)}
        sj = {s['school_id']: j for j, s in enumerate(schools)}
        assert y.shape == (len(sites), len(schools))
        supplement = C.CurrentSupplements(d); links.update(supplement.hashes)
        for p in [meta_path, label_path, input_path]: links[str(p)] = F.sha(p)
        relations = {}; conflicts = []
        for site, kinds in supplement.positive.items():
            for kind, ids in kinds.items():
                assert kind in TEXTUAL | GEOGRAPHIC
                for school in ids:
                    q, j = si[site], sj[school]
                    row = relations.setdefault((q, j), dict(site_id=site, school_id=school,
                        building_name=sites[q]['building_name'], school_name=schools[j]['school_name'],
                        site_index=q, school_index=j, current_label=int(y[q, j]), sources=[]))
                    path = supplement.paths['explicit_scope' if kind == 'explicit_positive_repair' else kind]
                    row['sources'].append(dict(kind=kind, path=str(path), sha256=F.sha(path),
                        evidence_type='textual_proof' if kind in TEXTUAL else 'geographic_estimate'))
                    if y[q, j] == 0: conflicts.append((site, school, kind))
        assert not conflicts
        train = set(map(int, label_file['train_ids']))
        assert train == set(np.flatnonzero((y >= 0).any(1)))
        references = {(int(q), int(j)) for q, j in zip(x.ref_sites, x.ref_schools)}
        extra = []
        for key, row in sorted(relations.items()):
            row['already_reference'] = key in references
            row['has_textual_proof'] = any(s['evidence_type'] == 'textual_proof' for s in row['sources'])
            row['complete_school_set'] = False
            if row['current_label'] == -1: extra.append(row)
        textual = [r for r in extra if r['has_textual_proof']]
        geographic = [r for r in extra if not r['has_textual_proof']]
        summary[d] = dict(sites=len(sites), schools=len(schools), current_training_rows=len(train),
            current_positive_cells=int((y == 1).sum()), current_negative_cells=int((y == 0).sum()),
            current_schools_with_positive=int((y == 1).any(0).sum()), cumulative_source_pairs=len(relations),
            missing_positive_labels=len(extra), missing_textual_positive_labels=len(textual),
            missing_geographic_positive_labels=len(geographic),
            new_training_rows_if_all_added=len({r['site_index'] for r in extra} - train),
            new_training_rows_if_textual_added=len({r['site_index'] for r in textual} - train),
            missing_reference_pairs=sum(not r['already_reference'] for r in extra),
            negative_conflicts=len(conflicts))
        F.write(OUT/d/'inventory.json', dict(summary=summary[d], missing_label_relations=extra,
            all_cumulative_positive_relations=list(relations.values()), target_gold_read=False,
            target_polygon_read=False, labels_modified=False, references_modified=False))
    totals = {k: sum(s[k] for s in summary.values()) for k in [
        'missing_positive_labels', 'missing_textual_positive_labels', 'missing_geographic_positive_labels',
        'new_training_rows_if_all_added', 'new_training_rows_if_textual_added', 'missing_reference_pairs', 'negative_conflicts']}
    F.write(OUT/'summary.json', dict(districts=summary, totals=totals,
        limitation='These facts already affect final outputs. Training/reference integration is an untested intervention, not a guaranteed accuracy gain.',
        policy='Do not turn unmatched schools into negatives; retain the evidence type, all original conditions, temporal receipts and same-property masking.',
        next_test='Separate textual-label supervision from reference-bank expansion and geographic estimates; preserve original weights and use matched controls. No source-gate or target-score claim.',
        target_gold_read=False, target_polygon_read=False, labels_modified=False, references_modified=False,
        training_performed=False, adopted=False))
    links[str(Path(__file__))] = F.sha(__file__)
    links[str(F.REPO/'scripts/shenzhen_current_supplements.py')] = F.sha(F.REPO/'scripts/shenzhen_current_supplements.py')
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'manifest.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('LATE_SUPERVISION_INVENTORY', summary, totals, flush=True)


if __name__ == '__main__': main()
