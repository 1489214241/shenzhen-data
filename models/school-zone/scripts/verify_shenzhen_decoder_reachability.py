"""Recheck the diagnostic using actual singleton decoder calls and set cover."""
from itertools import combinations
from pathlib import Path
import numpy as np
import audit_shenzhen_decoder_reachability as D
from shenzhen_partial_model import predict_sets
from shenzhen_scope_implication import expand_original

F, R = D.F, D.R


def main():
    R.check(D.OUT, 'source_freeze.json')
    diag = F.read(D.OUT/'diagnostic_not_training.json')
    for d, h in diag['gold_sha256'].items(): assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json') == h
    dest = D.OUT/'actual_decoder_verification.json'
    if dest.exists(): raise FileExistsError('Preserve diagnostic verification')
    truth = {(r['district'], r['site_id']): r for r in diag['rows']}
    counts = {}; independent_infeasible = 0
    for d in R.TARGETS:
        meta = F.read(R.OLD/'inputs'/f'{d}.json'); policy = F.read(R.OUT/'policies'/f'{d}.json')
        graph = F.read(R.OLD/'source_graphs'/f'{d}.json'); supplement = D.C.CurrentSupplements(d)
        signature = F.read(D.OUT/'source_signatures'/f'{d}.json'); sids = signature['school_ids']
        universes = [set(r['school_ids']) for r in policy['relations'] if r['kind'] == 'university']
        singletons = checked = witnessed = 0
        for q, site in enumerate(meta['sites']):
            key = (d, site['coordinate_group_id'])
            if key not in truth: continue
            entry = truth[key]; gold = set(entry['map_gold']); src = signature['rows'][q]
            known = [sids[j] for j in meta['known'].get(str(q), [])]
            positives = set().union(*supplement.positive[site['coordinate_group_id']].values())
            negatives = supplement.negative[site['coordinate_group_id']]
            def actual(selected):
                scores = np.asarray([1. if s in selected else 0. for s in sids], np.float64)
                r = predict_sets(scores, .5, sids, policy, graph, site, known)
                roots = set(r['raw_school_ids']) | set(r['positive_lower_bound'])
                output = set(r['predicted_school_ids']) | set(expand_original(roots, supplement.graph)) | positives
                return output-negatives
            units = {s: actual({s}) for s in sids}; singletons += len(sids)
            for school, output in units.items(): assert output == D.expand(signature, src, [school])
            # No representation constants are used in this independent solver:
            # union actual singleton outputs for a nonempty permitted root set.
            allowed = [s for s in sids if units[s] <= gold]
            feasible = False
            if allowed:
                if len(allowed) <= 18:
                    for length in range(1, len(allowed)+1):
                        for choice in combinations(allowed, length):
                            ss = set(choice)
                            if any(len(ss & g) > 1 for g in universes): continue
                            if set().union(*(units[s] for s in choice)) == gold:
                                feasible = True; break
                        if feasible: break
                else:
                    from scipy.optimize import milp, LinearConstraint, Bounds
                    matrix = [np.ones(len(allowed))]; lower = [1.]; upper = [np.inf]
                    for school in gold:
                        matrix.append([float(school in units[s]) for s in allowed]); lower.append(1.); upper.append(np.inf)
                    for group in universes:
                        matrix.append([float(s in group) for s in allowed]); lower.append(-np.inf); upper.append(1.)
                    result = milp(np.zeros(len(allowed)), integrality=np.ones(len(allowed)), bounds=Bounds(0,1),
                        constraints=LinearConstraint(np.asarray(matrix), lower, upper))
                    assert result.status in [0, 2]; feasible = result.status == 0
            assert feasible == entry['feasible_under_current_decoder'], key
            if feasible:
                witness = set(entry['raw_witness']); assert witness and actual(witness) == gold
                witnessed += 1
            else: independent_infeasible += 1
            checked += 1
            if checked % 100 == 0: print('ACTUAL_DECODER_CHECKED', d, checked, flush=True)
        counts[d] = dict(scored_sites=checked, actual_singleton_calls=singletons, executable_witnesses=witnessed)
    assert sum(c['scored_sites'] for c in counts.values()) == 1761
    F.write(dest, dict(districts=counts, independently_recounted_infeasible=independent_infeasible,
        all_actual_singletons_match_representation=True, all_feasible_witnesses_executed=True,
        actual_output_set_cover_agrees=True, original_gold_unchanged=True,
        target_diagnostic_only=True, used_for_training=False, audit_code_sha256=F.sha(__file__)))
    print('ACTUAL_DECODER_DIAGNOSTIC_VERIFIED', counts, independent_infeasible, flush=True)


if __name__ == '__main__': main()
