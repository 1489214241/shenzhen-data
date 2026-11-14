"""Audit frozen sources, supervision, saved predictions, and scoring labels."""
import gc
import json
from pathlib import Path
import numpy as np
import run_modernnca_relation as E
from zoning import train, evaluate


def main():
    frozen = json.loads((E.OUT / 'frozen.json').read_text(encoding='utf-8'))
    for f, h in frozen['files'].items():
        assert E.digest(E.REPO / f) == h, f
        assert E.digest(E.OUT / 'code' / f) == h, f
    summary = {}; artifacts = {}
    for k, s in E.BATCHES:
        root = E.OUT / f'k{k}_s{s}'
        c, D, R, X = E.prepared(k, s)
        cases = json.loads((root / 'scored_cases.json').read_text(encoding='utf-8'))
        expected_cases = [(d, e['q'], e['truth']) for d in D for e in evaluate.eval_rows(D[d], R[d].known)]
        assert [(e['district'], e['q'], e['truth']) for e in cases] == expected_cases
        info = {}
        for d in D:
            audit = json.loads((root / 'training_audit' / f'{d}.json').read_text(encoding='utf-8'))
            query = {e['q'] for e in evaluate.eval_rows(D[d], R[d].known)}
            assert not query.intersection(audit[d]['ids']), (d, 'target query in training')
            for source in D:
                expected = {int(q): int(X[source].y[q]) for q in X[source].supervised}
                if source != d: expected.update(train.uncovered_answers(D[source], X[source]))
                actual = dict(zip(audit[source]['ids'], audit[source]['labels']))
                assert actual == expected, (d, source, 'training questions differ from reference')
            for method in ['reference_matched', 'modernnca_selected']:
                pf = root / method / 'probs' / f'{d}.npz'; probs = np.load(pf)['probs']
                assert probs.shape == (3, D[d].n, D[d].school_n)
                assert np.isfinite(probs).all() and np.all(probs >= 0)
                assert np.allclose(probs.sum(-1), 1, atol=2e-5)
                for j, seed in enumerate(E.SEEDS):
                    sf = root / method / 'seeds' / f'{d}_{seed}.npz'
                    assert np.array_equal(np.load(sf)['P'], probs[j])
                pred = probs.mean(0).argmax(1)
                assert all(int(pred[e['q']]) == e['predictions'][method] for e in cases if e['district'] == d)
                artifacts[str(pf.relative_to(E.OUT))] = E.digest(pf)
            info[d] = dict(training_questions=sum(len(a['ids']) for a in audit.values()),
                           target_training_questions=len(audit[d]['ids']), evaluation_questions=len(query),
                           schools=D[d].school_n, reference_records=len(X[d].ref_sites),
                           removed_for_top10=0)
        summary[root.name] = info
        del X, D, R; gc.collect()
    for f in E.OUT.glob('k*/**/*.pt'): artifacts[str(f.relative_to(E.OUT))] = E.digest(f)
    E.save(E.DOC / '完整性审计.json', dict(status='PASS', source_files=len(frozen['files']),
        batches=summary, checked_prediction_files=len([p for p in artifacts if p.endswith('.npz')]),
        checkpoint_and_prediction_hashes=artifacts))
    print('PASS: 4 batches; all target evaluation IDs excluded; full reference training questions retained; '
          '64 ensemble files / 192 seed predictions; scoring labels equal cached labels; frozen sources unchanged.', flush=True)


if __name__ == '__main__': main()
