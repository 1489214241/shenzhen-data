"""Score frozen predictions with rules, separating manually verified references."""
import sys
import json
import pickle
import hashlib
import gc
from pathlib import Path
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from zoning import evaluate

root = repo / 'outputs/modernnca_relation_20261112'
doc = repo.parent / '科研推进/ModernNCA关系证据改进_20261112'
result = {}
for batch in ['k0_s1', 'k1_s1', 'k1_s2', 'k1_s3']:
    folder = root / batch
    f = folder / 'inputs.pkl'
    h = hashlib.sha256()
    with f.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    assert h.hexdigest() == json.loads((folder / 'inputs.json').read_text(encoding='utf8'))['input_sha256']
    with f.open('rb') as stream:
        c, D, R, X = pickle.load(stream)
    existing = json.loads((doc / '结果.json').read_text(encoding='utf8'))[batch]
    out = {}
    for method in ['modernnca_selected', 'reference_matched', 'historical_reference', 'historical_modernnca_rows']:
        pf = (folder / method / 'probs' if method in ['modernnca_selected', 'reference_matched'] else
              repo / 'outputs/main_pre_v1' / (batch if method == 'historical_reference' else batch + '__tabular/mnca_rows') / 'probs')
        allrows, realrows, rule, verified, model = [], [], [], [], []
        for d in D:
            p = pf / (d + '.npz')
            assert hashlib.sha256(p.read_bytes()).hexdigest() == existing['probability_hashes'][str(p.relative_to(repo))]
            P = np.load(p)['probs'].mean(0)
            rows = evaluate.overall(P, D[d], R[d])
            assert len(rows) == len(D[d].answers)
            for (q, t), row in zip(sorted(D[d].answers.items()), rows):
                allrows.append(row)
                if R[d].kind.get(q) == 'verified':
                    verified.append(row)
                else:
                    realrows.append(row)
                    (rule if q in R[d].known else model).append(row)

        def summ(rows):
            z = evaluate.summary(rows) if rows else {}
            for key, v in z.items():
                selected = [r for r in rows if key == '两城' or r['city'] == key or r['d'] == key]
                if selected:
                    v['correct'] = sum(bool(r['ok']) for r in selected)
            return z

        out[method] = {
            'rule_plus_model_excluding_verified': summ(realrows),
            'legacy_overall_including_verified': summ(allrows),
            'rule_only': summ(rule), 'verified_only': summ(verified), 'model_only': summ(model)}
        assert abs(out[method]['model_only']['两城']['accuracy'] - existing['accuracy'][method]['两城']['accuracy']) < 1e-9
    result[batch] = out
    print(batch, json.dumps({m: {key: v[key].get('两城') for key in v} for m, v in out.items()}, ensure_ascii=False), flush=True)
    del c, D, R, X
    gc.collect()

output = doc / '规则合并补充计分.json'
output.write_text(json.dumps({
    'note': 'Frozen pre_v1 inputs and predictions. Primary overall excludes manually verified reference points. Legacy overall includes them. accuracy uses historical any-hit rule criterion; strict requires exact school set. No training or new policy exclusion.',
    'results': result}, ensure_ascii=False, indent=2), encoding='utf8')
print(output)
