"""Policy-only multi-school priors, supplied before final set decoding.

Every count is a lower bound unless completeness is independently established.
Exact-count decoding below is an explicit experimental hypothesis, not a fact.
No test answers, geometry, training, or transitive closure are used here.
"""
import numpy as np
from .conditional_school_sets import exact_identity


def build_prior(policy, site, positive=(), negative=(), predicted_root=None):
    positive, negative = set(positive), set(negative)
    if positive & negative:
        raise ValueError('Conflicting input evidence')
    shared = [r for r in policy['relations']
              if r['kind'] == 'shared' and len(r['school_ids']) > 1]
    verified = set(positive)
    proofs, blocked = [], []
    # Triggers are the ORIGINAL independent positives, never inferred members.
    for r in shared:
        named = exact_identity(site, r.get('exact_site_names', []))
        triggered = bool(positive & set(r.get('trigger_school_ids', [])))
        if not (named or triggered):
            continue
        if set(r['school_ids']) & negative:
            blocked.append(r['rule_id'])
            continue
        verified.update(r['school_ids'])
        proofs.append(dict(rule_id=r['rule_id'], basis='exact_named_site' if named
                           else 'independent_positive_root', school_ids=r['school_ids']))
    estimated, estimates = set(verified), []
    # Unknown scope applicability remains a MODEL ESTIMATE, never a known count.
    if len(verified) < 2 and predicted_root is not None:
        for r in shared:
            if predicted_root not in r.get('trigger_school_ids', []):
                continue
            if set(r['school_ids']) & negative:
                blocked.append(r['rule_id'])
                continue
            estimated.update(r['school_ids'])
            estimates.append(dict(rule_id=r['rule_id'], basis='native_top1_scope_estimate',
                                  school_ids=r['school_ids']))
    return dict(verified_members=sorted(verified), estimated_members=sorted(estimated),
                verified_min_count=len(verified), estimated_count_hypothesis=len(estimated),
                proofs=proofs, estimated_proofs=estimates, blocked_rules=sorted(set(blocked)),
                complete_school_set=False, exact_count_verified=False,
                enrollment_eligibility=False, transitive_expansion=False)


def rank_with_count(scores, school_ids, count, excluded=(), university_groups=(),
                    allowed_shared_members=()):
    """Exactly k if feasible, before output; university members do not add choices.

Explicit base/shared evidence can support co-membership despite an overlapping
university list. That permission cannot spread to other university members.
"""
    scores = np.asarray(scores)
    if scores.shape != (len(school_ids),) or not np.isfinite(scores).all():
        raise ValueError('Invalid school scores')
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Count must be positive')
    excluded, allowed = set(excluded), set(allowed_shared_members)
    groups = [set(g) for g in university_groups]
    order = sorted(range(len(school_ids)), key=lambda j: (-float(scores[j]), school_ids[j]))
    chosen = set()
    for j in order:
        school = school_ids[j]
        if school in excluded:
            continue
        if any(school in g and chosen & g and
               not ({school} | (chosen & g)) <= allowed for g in groups):
            continue
        chosen.add(school)
        if len(chosen) == count:
            break
    return sorted(chosen), len(chosen) == count


def decode_prior(baseline, prior, scores, school_ids, mode, estimated=False,
                 excluded=(), university_groups=()):
    members = set(prior['estimated_members' if estimated else 'verified_members'])
    if len(members) < 2:
        return sorted(baseline), dict(active=False, reason='no_applicable_multi_prior')
    if mode == 'minimum':
        output = sorted((set(baseline) | members) - set(excluded))
        feasible = True
    elif mode == 'group_exact':
        output = sorted(members - set(excluded))
        feasible = len(output) == len(members)
    elif mode == 'count_exact':
        # This ablation reveals count alone: it deliberately withholds member IDs.
        # An exception list only prevents university rules erasing proven sharing.
        output, feasible = rank_with_count(scores, school_ids, len(members), excluded,
                                          university_groups, members)
    else:
        raise ValueError(mode)
    return output, dict(active=True, mode=mode, desired_count=len(members), feasible=feasible,
                        applicability='estimated' if estimated and len(prior['verified_members']) < 2
                        else 'text_supported', exact_count_is_hypothesis=mode != 'minimum',
                        count_is_lower_bound=True)
