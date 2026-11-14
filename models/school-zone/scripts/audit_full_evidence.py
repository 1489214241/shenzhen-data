"""Independent integrity audit of the frozen full-information experiment."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
from run_full_evidence import OUT, DOC, SOURCE, METHODS, SEEDS, BATCHES, loaded, digest, save, GRID, VALIDATION
from zoning import train, evaluate, model as MD
from zoning.full_evidence import tokens, FullEvidenceModel


def audit(require_complete=True):
    frozen = json.loads((OUT / 'frozen.json').read_text(encoding='utf8'))
    assert all(digest(REPO / path) == h for path, h in frozen['files'].items())
    result = dict(source_hashes_pass=True, input_contract={}, predictions={}, selection={}, target_group_overlaps={})
    for k, s in BATCHES:
        name = f'k{k}_s{s}'
        c, D, R, X = loaded(k, s)
        assert digest(SOURCE / name / 'inputs.pkl') == frozen['input_hashes'][name]
        input_info, pred_info, group_info = {}, {}, {}
        for d in D:
            x = X[d]
            ids = np.arange(min(11, D[d].n))
            t = MD.batch(x, ids)
            packed, valid = tokens(t, x.np_)
            nr = len(x.ref_sites)
            assert torch.equal(packed[:, :nr, :-2], t['px'])
            assert torch.equal(packed[:, nr:, x.np_:-2], t['ax'])
            restored_pair = packed[:, :nr, :x.np_].clone(); restored_pair[:, :, 0] *= 10
            assert torch.allclose(restored_pair, t['pair'], atol=1e-5)
            restored_anchor = packed[:, nr:, x.np_:-2][:, :, :x.na].clone(); restored_anchor[:, :, 0] *= 10
            assert torch.allclose(restored_anchor, t['anchor'], atol=1e-5)
            assert torch.equal(packed[:, :nr, -2], t['gb'][None].expand(len(ids), -1))
            assert torch.equal(valid[:, :nr], t['valid']) and valid[:, nr:].all()
            assert torch.equal(t['mem'][-D[d].school_n:], torch.eye(D[d].school_n, device=t['mem'].device))
            expected_refs = sorted({(q, school) for q, schools in R[d].known.items() if q not in D[d].not_reference for school in schools})
            assert expected_refs == list(zip(x.ref_sites.tolist(), x.ref_schools.tolist()))
            input_info[d] = dict(points=D[d].n, references=nr, schools=D[d].school_n,
                                 pair_features=x.np_, school_features=x.na,
                                 query_count=len(evaluate.eval_rows(D[d], R[d].known)), packing_pass=True)
            eval_ids = {row['q'] for row in evaluate.eval_rows(D[d], R[d].known)}
            assert not eval_ids.intersection(x.supervised.tolist())
            overlap = set(D[d].groups[list(eval_ids)]) & set(D[d].groups[x.supervised])
            group_info[d] = dict(shared_policy_known_groups=len(overlap), note='Policy-known training is permitted; query labels excluded. Same-building support tokens masked per question.')
            training_audit = OUT / name / 'training_audit' / f'{d}.json'
            if training_audit.exists():
                recorded = json.loads(training_audit.read_text(encoding='utf8'))
                for dd in D:
                    lab = {int(q):int(X[dd].y[q]) for q in X[dd].supervised}
                    if dd != d: lab.update(train.uncovered_answers(D[dd], X[dd]))
                    if lab:
                        assert recorded[dd]['ids'] == sorted(lab)
                        assert recorded[dd]['labels'] == [lab[q] for q in sorted(lab)]
                input_info[d]['training_ids_labels_pass'] = True
            else:
                assert not require_complete
            for method in METHODS:
                f = OUT / name / method / 'probs' / f'{d}.npz'
                if not f.exists():
                    assert not require_complete
                    continue
                arr = np.load(f)['probs']
                assert arr.shape == (len(SEEDS), D[d].n, D[d].school_n)
                assert np.isfinite(arr).all() and np.allclose(arr.sum(-1), 1, atol=2e-5)
                for j, seed in enumerate(SEEDS):
                    sf = OUT / name / method / 'seeds' / f'{d}_{seed}.npz'
                    assert np.array_equal(arr[j], np.load(sf)['P'])
                    assert sf.with_suffix('.pt').exists() and sf.with_suffix('.json').exists()
                sf = OUT / name / method / 'seeds' / f'{d}_{SEEDS[0]}.npz'
                if method == 'reference':
                    replay = MD.RetrievalScorer(x.np_, x.na, MD.LinearRetrieval(x), SEEDS[0], width=32)
                else:
                    replay = FullEvidenceModel(method, x.np_, x.na, SEEDS[0], dim=32)
                replay.load_state_dict(torch.load(sf.with_suffix('.pt'), map_location='cpu', weights_only=True))
                replay = replay.to(t['px'].device).eval()
                with torch.no_grad():
                    replayed = replay(t).cpu().numpy()
                err = float(np.abs(replayed - arr[0, ids]).max())
                assert err < 2e-5, (name, d, method, err)
                pred_info[f'{d}/{method}'] = dict(sha256=digest(f), shape=list(arr.shape), replay_max_error=err,
                    weight_hashes={str(seed):digest(OUT / name / method / 'seeds' / f'{d}_{seed}.pt') for seed in SEEDS})
                del replay
            del t, packed, valid
        result['input_contract'][name] = input_info
        result['predictions'][name] = pred_info
        result['target_group_overlaps'][name] = group_info
        if require_complete:
            hashes = json.loads((OUT / name / 'prediction_hashes.json').read_text(encoding='utf8'))
            assert len(hashes) == len(D) * len(METHODS)
            assert all(digest(OUT / name / f) == h for f, h in hashes.items())
            cases = json.loads((OUT / name / 'scored_cases.json').read_text(encoding='utf8'))
            expected = [(d, e['q'], e['truth']) for d in D for e in evaluate.eval_rows(D[d], R[d].known)]
            assert [(r['district'], r['q'], r['truth']) for r in cases] == expected
        del c, D, R, X; gc.collect(); torch.cuda.empty_cache()
    for path in sorted((OUT / 'selection').glob('*.json')):
        rec = json.loads(path.read_text(encoding='utf8'))
        d, method = path.stem.rsplit('_', 1)
        assert len(rec['trials']) == len(GRID)
        assert [t['hp'] for t in rec['trials']] == GRID
        assert rec['validation'] == VALIDATION[d]
        chosen = max(rec['trials'], key=lambda z:(z['best']['accuracy'], -z['best']['nll'], -z['best']['step']))
        assert rec['hp'] == chosen['hp'] and rec['steps'] == chosen['best']['step']
        assert all(t['complete'] for t in rec['trials'])
        result['selection'][path.stem] = dict(pass_=True, steps=rec['steps'], at_hard_cap=rec['steps']==1280)
    if require_complete:
        assert len(result['selection']) == len(VALIDATION) * len(METHODS)
    result['complete'] = require_complete
    result['pass'] = True
    dest = DOC / ('完整性审计.json' if require_complete else '输入审计.json')
    save(dest, result)
    print('PASS', dest, 'models', sum(len(x) for x in result['predictions'].values()), flush=True)


if __name__ == '__main__':
    audit('--inputs-only' not in sys.argv)
