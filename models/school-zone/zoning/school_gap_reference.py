"""Nested per-school reference gap filling with an explicit simulation oracle."""
import numpy as np


def fill_school_gaps(schools, verified, candidates, targets=(1,2,3),seed=20261231):
    """School IDs ordered by caller, distinct public project groups counted.

    verified: partial positive rows {school, group}. No completeness assumed.
    candidates: oracle rows {q, group, schools}. Selection explicitly inspects
    school membership to simulate asking that school for a reference. It does
    not maximize cross-school gain from hidden complete labels. Once selected,
    that candidate's full set is released and can credit multiple schools.
    """
    have={s:set() for s in schools}
    for row in verified:have[row['school']].add(row['group'])
    initial={s:len(v) for s,v in have.items()}
    order=np.random.default_rng(seed).permutation(len(candidates))
    chosen=[];chosen_groups=set();result={}
    for target in targets:
        for school in schools:
            while len(have[school])<target:
                candidate=next((candidates[i] for i in order if school in candidates[i]['schools']
                    and candidates[i]['group'] not in have[school]
                    and candidates[i]['group'] not in chosen_groups),None)
                if candidate is None:break
                chosen.append(dict(candidate,requested_school=school,first_target=target))
                chosen_groups.add(candidate['group'])
                for s in candidate['schools']:have[s].add(candidate['group'])
        result[target]=dict(selected=list(chosen),counts={s:len(v) for s,v in have.items()},
                            deficits={s:target-len(v) for s,v in have.items() if len(v)<target},initial_counts=initial)
    return result
