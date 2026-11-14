"""Replay the cumulative decoder against ALL frozen round31 predictions.

Reads original decisions and factual inputs only; never scores target labels.
"""
from pathlib import Path
import shutil
from shenzhen_current_supplements import CurrentSupplements, F, R

OUT = R.ROOT / 'current_supplements_replay_20261113'
LATEST = R.ROOT / 'round31_explicit_scope/combined'
ROOTS = {
    'round07': R.ROOT / 'round06_balanced_ranking/predictions/balanced_rank',
    'round12': R.ROOT / 'round12_conjunction_replay/predictions',
}


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve cumulative replay audit')
    R.check(LATEST, 'prediction_freeze.json')
    cfg = F.read(LATEST / 'protocol.json')
    links = {}; checked = {}; total = 0
    for district in R.TARGETS:
        supplement = CurrentSupplements(district)
        links.update(supplement.hashes)
        old_graph = F.read(R.OLD / 'source_graphs' / f'{district}.json')
        edge = lambda g: {(e['trigger_school_id'], e['implied_school_id']) for e in g['edges']}
        assert edge(old_graph) <= edge(supplement.graph)
        for arm in cfg['arms']:
            for objective in cfg['objectives']:
                for label in cfg['labels']:
                    old = ROOTS[arm] / objective / label / f'{district}.json'
                    latest = LATEST / 'predictions' / arm / objective / label / f'{district}.json'
                    rows = F.read(old)
                    rebuilt = supplement.apply(rows)
                    expected = F.read(latest)
                    assert [(r['site_id'], r['building_name'], r['after']) for r in rebuilt] == [
                        (r['site_id'], r['building_name'], r['after']) for r in expected], (arm, objective, label, district)
                    # The routine must preserve the original trigger decision and
                    # must not mutate the saved caller rows.
                    assert rows == F.read(old)
                    assert [r['decision'] for r in rows] == [r['decision'] for r in rebuilt]
                    for p in [old, latest]: links[str(p)] = F.sha(p)
                    key = f'{arm}/{objective}/{label}/{district}'
                    checked[key] = len(rows); total += len(rows)
    assert len(checked) == 88 and total == 1802*2*2*11
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_current_supplements.py',
              F.REPO/'scripts/shenzhen_scope_implication.py']:
        dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); links[str(p)] = F.sha(p)
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'audit.json', dict(prediction_files=len(checked), prediction_rows=total,
        checked=checked, all_round31_predictions_reproduced=True,
        original_roots_preserved=True, caller_inputs_unchanged=True,
        new_model_predictions_generated=False, target_gold_read=False, target_polygon_read=False,
        purpose='One cumulative supplement path for future source-gated trials; no new scientific intervention.'))
    F.write(OUT/'manifest.json', dict(files={str(p.relative_to(OUT)): F.sha(p)
        for p in OUT.rglob('*') if p.is_file()}))
    print('CUMULATIVE_REPLAY_VERIFIED', len(checked), total, flush=True)


if __name__ == '__main__':
    main()
