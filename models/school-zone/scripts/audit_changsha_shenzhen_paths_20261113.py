"""Post-score comparison only. No training, inference, or input modification.

Uses archived predictions and scored labels to describe input support. These
answer-dependent diagnostics must never become reference selection rules.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / 'outputs/shenzhen_resolution_20261112/changsha_path_audit_20261113/diagnostics_not_training'
ROOT = REPO / 'outputs/shenzhen_resolution_20261112'
HASHES = {}


def read(path):
    path = Path(path)
    HASHES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == '.pkl':
        with path.open('rb') as handle:
            return pickle.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(rows):
    return dict(n=len(rows), correct=sum(r['correct'] for r in rows),
                exact_accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
                errors=dict(Counter(r['error'] for r in rows if not r['correct'])),
                all_correct_schools_have_reference=sum(r['all_refs'] for r in rows),
                any_correct_school_has_reference=sum(r['any_refs'] for r in rows))


def error(pred, gold):
    if pred == gold:
        return 'exact'
    if not pred & gold:
        return 'disjoint'
    if pred < gold:
        return 'omission_only'
    if gold < pred:
        return 'extra_only'
    return 'extra_and_omission'


def input_stats(x, gold_by_q, school_ids):
    total = hits = sites = consistent = outside = 0
    ref_by_site = {}
    for q, s in zip(x.ref_sites, x.ref_schools):
        ref_by_site.setdefault(int(q), set()).add(school_ids[int(s)])
    for q, ss in ref_by_site.items():
        if q not in gold_by_q:
            outside += 1
            continue
        gg = gold_by_q[q]
        total += len(ss)
        hits += len(ss & gg)
        sites += 1
        consistent += ss <= gg
    return dict(sites=x.n, schools=x.school_n, reference_sites=len(ref_by_site),
                reference_assignments=len(x.ref_sites), reference_schools=len(set(x.ref_schools)),
                map_scored_reference_sites=sites, unscored_reference_sites=outside,
                map_consistent_positive_relations=hits, map_scored_positive_relations=total,
                sites_without_map_contradiction=consistent,
                warning='Map consistency is not proof of policy validity; partial reference sets need not equal complete Gold.')


def diagnose_rows(x, base, school_ids):
    rows = []
    for q, g, p, known, site in base:
        legal = {school_ids[int(s)] for s in x.ref_schools[~x.self_mask[q]]}
        rows.append(dict(q=q, site_id=site, correct=p == g, size=len(g), known=known,
                         all_refs=g <= legal, any_refs=bool(g & legal), error=error(p, g)))
    return rows


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed diagnosis')
    chs = ['天心区', '芙蓉区', '雨花区', '开福区']
    sz = ['龙华区', '龙岗区']
    prior = read(REPO / 'outputs/shenzhen_full_model_20261111/深圳与长沙参照覆盖逐例.json')
    old_gold = read(REPO / 'outputs/shenzhen_aligned_model_20261111/scored_cases.json')
    best = read(ROOT / 'round31_explicit_scope/combined/results.json')
    read(ROOT / 'round31_explicit_scope/combined/protocol.json')
    read(ROOT / 'round06_balanced_ranking/protocol.json')
    read(ROOT / 'round03_partial_labels/protocol.json')
    chs_eval = read(REPO / 'outputs/main_pre_v1/k0_s1/eval.json')
    configs = {k: read(REPO / f'outputs/main_pre_v1/{k}/config.json') for k in ['k0_s1', 'k1_s1']}
    inputs, districts, all_rows, expanded = {}, {}, {}, {}
    for d in chs + sz:
        if d in chs:
            x = read(REPO / 'outputs/shenzhen_full_model_20261111/inputs' / f'{d}.pkl')
            original = [r for r in prior if r['district'] == d]
            ids = list(range(x.school_n))
            base = [(r['q'], set(r['gold']), set(r['pred']), r['known'], str(r['q'])) for r in original]
        else:
            meta = read(ROOT / 'round03_partial_labels/inputs' / f'{d}.json')
            ids = [s['school_id'] for s in meta['schools']]
            pos = {s['coordinate_group_id']: q for q, s in enumerate(meta['sites'])}
            x = read(ROOT / 'round03_partial_labels/inputs' / f'{d}.pkl')
            prediction = read(ROOT / 'round31_explicit_scope/combined/predictions/round07/balanced_exact/ensemble' / f'{d}.json')
            preds = {r['site_id']: set(r['after']) for r in prediction}
            assert len(preds) == len(meta['sites']) == x.n
            base = [(pos[r['site_id']], set(r['gold']), preds[r['site_id']], str(pos[r['site_id']]) in meta['known'], r['site_id'])
                    for r in old_gold if r['district'] == d and r['has_map_gold']]
        gg = {q:g for q,g,_,_,_ in base}
        inputs[d] = input_stats(x, gg, ids)
        rows = diagnose_rows(x, base, ids)
        if d in chs:
            assert all(a['all_refs'] == b['all_gold_refs'] and a['any_refs'] == b['any_gold_ref'] for a,b in zip(rows, original))
        else:
            xx = read(REPO / 'data/深圳市/社区归属参照输入_20261112' / d / 'inputs.pkl')
            expanded[d] = input_stats(xx, gg, ids)
            assert np.array_equal(x.groups, xx.groups)
            del xx
        districts[d] = summarize(rows)
        all_rows[d] = rows
        del x
    cities = {}
    for city, ds in [('长沙', chs), ('深圳', sz)]:
        rows = [r for d in ds for r in all_rows[d]]
        slices = {'all':rows, 'single':[r for r in rows if r['size']==1], 'multi':[r for r in rows if r['size']>1],
                  'rules_have_positive':[r for r in rows if r['known']], 'rules_no_positive':[r for r in rows if not r['known']],
                  'single_rules_no_positive':[r for r in rows if r['size']==1 and not r['known']],
                  'all_school_references_present':[r for r in rows if r['all_refs']],
                  'some_school_reference_missing':[r for r in rows if not r['all_refs']],
                  'multi_all_school_references_present':[r for r in rows if r['size']>1 and r['all_refs']],
                  'multi_some_school_reference_missing':[r for r in rows if r['size']>1 and not r['all_refs']]}
        cities[city] = {k:summarize(v) for k,v in slices.items()}
    assert cities['长沙']['all']['correct'] == chs_eval['changsha_sets']['overall']['exact_set_correct'] == 3362
    assert cities['深圳']['all']['correct'] == best['reports']['round07']['balanced_exact']['ensemble']['all']['after']['correct'] == 959
    assert cities['深圳']['all']['n'] == 1761
    weight = cities['长沙']['multi']['n'] / cities['长沙']['all']['n']
    standardized = (1-weight)*cities['深圳']['single']['exact_accuracy'] + weight*cities['深圳']['multi']['exact_accuracy']
    protocol = dict(status='post_score_descriptive_comparison_not_causal_or_adoption',
                    no_new_training=True, no_new_inference=True, raw_target_gold_or_polygon_read=False,
                    archived_scored_labels_used=True, diagnostic_forbidden_as_training_input=True,
                    compared_changsha='pre_v1 K0 full-population final-set output, not K1/K3 or unknown-only headline',
                    compared_shenzhen='round31 combined/round07 balanced_exact ensemble; round03 model inputs; later facts applied at output',
                    warning='Different training, partial-label semantics, population units and development-selection histories remain.')
    result = dict(protocol=protocol, inputs=inputs, expanded_input_not_used_by_best_model=expanded,
                  cities=cities, districts=districts,
                  shenzhen_at_changsha_single_multi_mix=standardized,
                  standardization_warning='Arithmetic composition only, not an intervention or forecast.',
                  changsha_k_budgets={k:cfg.get('references') for k,cfg in configs.items()})
    OUT.mkdir(parents=True)
    for name, obj in [('report.json', result), ('manifest.json', dict(inputs_sha256=HASHES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))]:
        (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
