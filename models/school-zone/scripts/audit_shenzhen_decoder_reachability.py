"""Quarantined diagnostic of the CURRENT decoder's set feasibility.

Prepare a source-only union representation, verify it against frozen decisions,
then ask whether any raw school set could produce each fixed map answer. Gold
is used ONLY in diagnose, never in training, thresholding or source admission.
This is a decoder ceiling diagnostic, not a deployable prediction method.
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from sympy import Symbol, And, Or, Not
from sympy.logic.inference import satisfiable
import shenzhen_current_supplements as C
from shenzhen_partial_model import predict_sets
from shenzhen_scope_implication import expand_original
from zoning import shenzhen_policy as SP

F, R = C.F, C.R
OUT = R.ROOT/'decoder_reachability_diagnostic_20261113'
LATEST = R.ROOT/'round31_explicit_scope/combined'
ROOTS = {'round07': R.ROOT/'round06_balanced_ranking/predictions/balanced_rank',
         'round12': R.ROOT/'round12_conjunction_replay/predictions'}


def expand(signature, row, selected):
    table = signature['graph_units'] if row['grounded_policy'] else signature['policy_units']
    output = set(row['constant'])
    for s in selected: output.update(table[s])
    return output-set(row['negative'])


def prepare():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve source-only decoder representation')
    links = {}; counts = {}; rng = np.random.RandomState(20261113)
    cfg = F.read(LATEST/'protocol.json')
    for d in R.TARGETS:
        meta_path = R.OLD/'inputs'/f'{d}.json'; policy_path = R.OUT/'policies'/f'{d}.json'
        meta = F.read(meta_path); policy = F.read(policy_path); supplement = C.CurrentSupplements(d)
        school_ids = [s['school_id'] for s in meta['schools']]; rows = []
        graph_units = {s: set(expand_original([s], supplement.graph)) for s in school_ids}
        policy_units = {s: set(v) for s, v in graph_units.items()}
        for relation in policy['relations']:
            if relation['kind'] == 'shared':
                for s in relation.get('trigger_school_ids', []): policy_units[s].update(relation['school_ids'])
        signature = dict(district=d, school_ids=school_ids,
            graph_units={s: sorted(v) for s, v in graph_units.items()},
            policy_units={s: sorted(v) for s, v in policy_units.items()},
            university_groups=[r['school_ids'] for r in policy['relations'] if r['kind'] == 'university'], rows=rows)
        for q, site in enumerate(meta['sites']):
            known = {school_ids[j] for j in meta['known'].get(str(q), [])}
            grounded = SP.apply(policy, site, ())['policy_supported_school_ids']
            constant = set(expand_original(known, supplement.graph)) | set(SP.apply(policy, site, sorted(known))['predicted_school_ids'])
            for values in supplement.positive[site['coordinate_group_id']].values(): constant.update(values)
            rows.append(dict(site_id=site['coordinate_group_id'], building_name=site['building_name'],
                constant=sorted(constant), negative=sorted(supplement.negative[site['coordinate_group_id']]),
                known=sorted(known), grounded_policy=bool(grounded)))
        n = 0
        for arm in cfg['arms']:
            for objective in cfg['objectives']:
                for label in cfg['labels']:
                    root_path = ROOTS[arm]/objective/label/f'{d}.json'
                    latest_path = LATEST/'predictions'/arm/objective/label/f'{d}.json'
                    roots = F.read(root_path); expected = F.read(latest_path)
                    assert [r['site_id'] for r in roots] == [r['site_id'] for r in rows] == [r['site_id'] for r in expected]
                    for row, original, target in zip(rows, roots, expected):
                        raw = original['decision']['raw_school_ids']
                        assert all(len(set(raw) & set(g)) <= 1 for g in signature['university_groups'])
                        assert expand(signature, row, raw) == set(target['after'])
                    n += len(rows)
                    links[str(root_path)] = F.sha(root_path); links[str(latest_path)] = F.sha(latest_path)
        # Additional source-only sets exercise branches beyond frozen predictions.
        # Actual university resolution chooses the raw set; only those resolved
        # roots are passed to the union representation.
        old_graph = F.read(R.OLD/'source_graphs'/f'{d}.json')
        for pattern in range(3):
            actual = []
            for row, site in zip(rows, meta['sites']):
                scores = rng.uniform(size=len(school_ids))
                threshold = [0.2, 0.8, 1.1][pattern]
                decision = predict_sets(scores, threshold, school_ids, policy, old_graph, site, row['known'])
                actual.append(dict(district=d, site_id=row['site_id'], building_name=row['building_name'],
                    decision=decision, after=decision['predicted_school_ids']))
            actual = supplement.apply(actual)
            for row, measured in zip(rows, actual):
                assert expand(signature, row, measured['decision']['raw_school_ids']) == set(measured['after'])
        F.write(OUT/'source_signatures'/f'{d}.json', signature)
        counts[d] = dict(frozen_rows_reproduced=n, random_pattern_rows=3*len(rows))
        links.update(supplement.hashes)
        for p in [meta_path, policy_path, R.OLD/'source_graphs'/f'{d}.json']:
            links[str(p)] = F.sha(p)
        print('DECODER_SOURCE_REPRESENTATION', d, counts[d], flush=True)
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_current_supplements.py',
              F.REPO/'scripts/shenzhen_partial_model.py', F.REPO/'zoning/shenzhen_policy.py']:
        links[str(p)] = F.sha(p)
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'source_audit.json', dict(districts=counts, target_gold_read=False, target_polygon_read=False,
        limitation='Represents current decoder constraints. Its feasibility is not an enrollment judgement or an accuracy claim.'))
    F.write(OUT/'source_freeze.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


def possible(signature, row, gold):
    constant = set(row['constant'])-set(row['negative'])
    if not constant <= gold: return False, [], 'mandatory_extra_school', sorted(constant-gold), 0
    table = signature['graph_units'] if row['grounded_policy'] else signature['policy_units']
    units = {s: set(v)-set(row['negative']) for s, v in table.items()}
    allowed = sorted(s for s, v in units.items() if v <= gold)
    needed = gold-constant
    if not allowed: return False, [], 'no_nonempty_raw_set', [], 0
    missing = needed-set().union(*(units[s] for s in allowed))
    if missing: return False, [], 'required_school_has_no_legal_root', sorted(missing), len(allowed)
    symbols = {s: Symbol('r'+str(i)) for i, s in enumerate(allowed)}
    clauses = [Or(*symbols.values())]
    for s in needed: clauses.append(Or(*(symbols[j] for j in allowed if s in units[j])))
    for group in signature['university_groups']:
        for a, b in combinations(sorted(set(group)&set(allowed)), 2): clauses.append(Or(Not(symbols[a]), Not(symbols[b])))
    witness = satisfiable(And(*clauses))
    feasible = witness is not False
    selected = [s for s in allowed if witness.get(symbols[s], False)] if feasible else []
    if feasible:
        assert selected and expand(signature, row, selected) == gold
        assert all(len(set(selected)&set(g)) <= 1 for g in signature['university_groups'])
    # Independent exhaustive bit-mask check for small sets, MILP for larger ones.
    if len(allowed) <= 16:
        check = False
        needed_order = sorted(needed); full = (1 << len(needed_order))-1
        masks = [sum(1 << i for i, v in enumerate(needed_order) if v in units[s]) for s in allowed]
        groups = [sum(1 << i for i, s in enumerate(allowed) if s in g) for g in signature['university_groups']]
        for choice in range(1, 1 << len(allowed)):
            if any((choice & g).bit_count() > 1 for g in groups): continue
            covered = 0
            for i, mask in enumerate(masks):
                if choice & (1 << i): covered |= mask
            if covered == full: check = True; break
        assert check == feasible
    else:
        from scipy.optimize import milp, Bounds, LinearConstraint
        matrix = [np.ones(len(allowed))]; lower = [1.]; upper = [np.inf]
        for school in needed:
            matrix.append([float(school in units[s]) for s in allowed]); lower.append(1.); upper.append(np.inf)
        for group in signature['university_groups']:
            matrix.append([float(s in group) for s in allowed]); lower.append(-np.inf); upper.append(1.)
        result = milp(np.zeros(len(allowed)), integrality=np.ones(len(allowed)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(np.asarray(matrix), lower, upper))
        assert result.status in [0, 2] and (result.status == 0) == feasible
    return feasible, selected, 'feasible' if feasible else 'university_or_coupled_root_constraints', [], len(allowed)


def diagnose():
    from collections import Counter
    R.check(OUT, 'source_freeze.json')
    for p, h in F.read(OUT/'input_links.json')['files'].items(): assert F.sha(p) == h
    if (OUT/'diagnostic_not_training.json').exists(): raise FileExistsError('Preserve diagnosis')
    hashes = F.read(R.BASE/'results.json')['gold_sha256']
    for d, h in hashes.items(): assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json') == h
    gold = {(r['district'], r['site_id']): r for r in F.read(R.BASE/'scored_cases.json')}
    results = []; summaries = {}
    for d in R.TARGETS:
        signature = F.read(OUT/'source_signatures'/f'{d}.json')
        current = {r['site_id']: set(r['after']) for r in F.read(LATEST/'predictions/round07/balanced_exact/ensemble'/f'{d}.json')}
        for row in signature['rows']:
            g = gold[d, row['site_id']]
            if not g['has_map_gold']: continue
            truth = set(g['gold']); feasible, witness, reason, conflicts, n = possible(signature, row, truth)
            correct = current[row['site_id']] == truth
            assert feasible or not correct
            results.append(dict(district=d, site_id=row['site_id'], building_name=row['building_name'],
                map_gold=sorted(truth), currently_correct=correct, feasible_under_current_decoder=feasible,
                reason=reason, conflicting_or_unreachable_schools=conflicts, raw_witness=witness,
                allowed_roots=n, semantic_pending=g['semantic_pending'], used_for_training=False))
        local = [r for r in results if r['district'] == d]
        summaries[d] = dict(scored=len(local), currently_correct=sum(r['currently_correct'] for r in local),
            feasible=sum(r['feasible_under_current_decoder'] for r in local), reasons=dict(Counter(r['reason'] for r in local)))
        print('DECODER_REACHABILITY', d, summaries[d], flush=True)
    assert len(results) == 1761
    F.write(OUT/'diagnostic_not_training.json', dict(districts=summaries, rows=results, gold_sha256=hashes,
        source_representation_frozen_before_gold=True, original_gold_unchanged=True,
        no_model_or_threshold_changed=True, independent_validation=False, adopted=False,
        interpretation='Feasible means some hypothetical raw set can match the map answer after current rules. It does not show the model can learn or choose that set. Infeasibility identifies a conflict within the present pipeline/Gold contract, not which source is wrong.',
        forbidden_uses='Do not train on witnesses, tune thresholds, change Gold or remove policy rules from these diagnostics.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'diagnose']); globals()[p.parse_args().phase]()
