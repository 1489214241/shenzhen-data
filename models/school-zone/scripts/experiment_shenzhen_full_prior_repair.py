"""Adaptive integration repair: raw hypotheses must obey original-root scopes.

The previous target result has been seen. Preserve it, report both stages.
"""
import argparse
from pathlib import Path
import experiment_shenzhen_full_prior as F
from zoning.full_policy_prior import one_pass

BASE=F.OUT
OUT=F.REPO/'outputs/modern_shenzhen_full_prior_repaired_20261114'
original_pool=F.candidate_pool
POLICY={'relations':[]}
GRAPHS=[]


def corrected_pool(records,prior,baseline,raw,scope_positive,excluded,compose):
    # Only ORIGINAL model-selected roots may trigger expansion. Existing source
    # templates and completed members are never recursively expanded.
    if not records and raw:
        raw=one_pass(raw,POLICY,GRAPHS)
    return original_pool(records,prior,baseline,raw,scope_positive,excluded,compose)


def install():
    F.OUT=OUT
    F.candidate_pool=corrected_pool
    POLICY['relations'].clear();GRAPHS.clear()
    for d in ['龙华区','龙岗区']:
        POLICY['relations'].extend(F.read(F.INPUT/d/'reviewed_policy.json')['relations'])
        GRAPHS.append(F.read(OUT/'inventory'/f'{d}.json')['graph'])


def prepare():
    F.OUT=OUT
    F.prepare()
    c=F.read(OUT/'protocol.json')
    c.update(previous_full_run=str(BASE),previous_full_target_results_seen=True,
        integration_repair='Expand ORIGINAL model raw roots once by the same direct scope graph/shared rules before adding that hypothesis. Previously a free raw candidate bypassed scope constraints and dominated additive utility.',
        no_new_fitting_or_threshold=True,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',c)
    links=F.read(OUT/'inputs_sha256.json')
    links[str(OUT/'protocol.json')]=F.sha(OUT/'protocol.json')
    links[str(Path(__file__).resolve())]=F.sha(Path(__file__))
    links[str(BASE/'prediction_freeze.json')]=F.sha(BASE/'prediction_freeze.json')
    F.write(OUT/'inputs_sha256.json',links)
    F.M.log('REPAIRED PROTOCOL SEALED; PRIOR RESULT SEEN; NO NEW TARGET SCORES')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','infer','score','audit'])
    phase=ap.parse_args().phase
    if phase=='prepare':prepare()
    else:
        install();getattr(F,phase)()
