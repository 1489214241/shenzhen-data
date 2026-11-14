"""Check the identity intervention on every frozen prediction, without Gold."""
from collections import defaultdict
from pathlib import Path
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'administrative_identity_review_v1'
PARENT = ROOT / 'round20_residential_membership'
OUT = ROOT / 'round23_administrative_identity'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve complete intervention audit')
    sources = {str(Path(__file__)): F.sha(__file__)}
    for folder, filename in [(SOURCE, 'manifest.json'), (PARENT, 'prediction_freeze.json')]:
        p = folder / filename
        sources[str(p)] = F.sha(p)
        manifest = F.read(p)
        for rel, h in manifest['files'].items():
            assert F.sha(folder / rel) == h, rel
            sources[str(folder / rel)] = h
        for path, h in manifest.get('sources', {}).items():
            assert F.sha(path) == h, path
            sources[path] = h
    extra_by_district = {}
    for district in F.SZ.DISTRICTS:
        index = defaultdict(set)
        for fact in F.read(SOURCE / 'school_facts' / f'{district}.json')['facts']:
            assert any(w['kind'] == 'official_community_membership' for w in fact['proof'])
            assert any(w['kind'] == 'reviewed_administrative_identity' for w in fact['proof'])
            index[fact['site_id']].add(fact['school_id'])
        extra_by_district[district] = index
    reports = []
    config = F.read(PARENT / 'protocol.json')
    for arm in config['arms']:
        for objective in config['objectives']:
            for label in config['labels']:
                for district, population in [('龙华区', 628), ('龙岗区', 1174)]:
                    p = PARENT / 'predictions' / arm / objective / label / f'{district}.json'
                    rows = F.read(p)
                    assert len(rows) == len({r['site_id'] for r in rows}) == population
                    index = extra_by_district[district]
                    assert set(index) <= {r['site_id'] for r in rows}
                    changed = []
                    for row in rows:
                        after = set(row['after']) | index.get(row['site_id'], set())
                        if after != set(row['after']):
                            changed.append(dict(site_id=row['site_id'], before=row['after'], after=sorted(after)))
                    reports.append(dict(arm=arm, objective=objective, label=label, district=district,
                        checked_rows=len(rows), source_path=str(p), source_sha256=F.sha(p),
                        changed_rows=changed, sets_identical=not changed))
    assert len(reports) == 88
    # No target metric needs to be recomputed when every candidate set is identical.
    assert all(r['sets_identical'] for r in reports), 'Changed predictions require a separately frozen scoring phase'
    F.write(OUT / 'source_only_intervention_audit.json', dict(status='completed_no_prediction_change',
        reports=reports, summary=dict(prediction_files=88,
            checked_rows=sum(r['checked_rows'] for r in reports), changed_rows=0,
            new_source_supported_relations=4, affected_source_sites=2,
            previously_predicted_relations_now_source_supported=4),
        target_gold_read=False, target_polygons_read=False, model_training_performed=False,
        new_accuracy_measured=False, adopted=False,
        conclusion='The identity repair adds evidence for four existing predictions; all seeds, objectives and comparator arms are unchanged. It cannot increase the fixed score.'))
    F.write(OUT / 'manifest.json', dict(sources=sources, files={
        'source_only_intervention_audit.json': F.sha(OUT / 'source_only_intervention_audit.json')}))
    print('CHECKED', len(reports), 'prediction files;', sum(r['checked_rows'] for r in reports),
          'rows; changed sets 0; no scoring or retraining performed', flush=True)


if __name__ == '__main__':
    main()
