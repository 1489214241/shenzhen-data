"""One declared source-only convergence extension; no target scores read yet."""
import argparse
from pathlib import Path

import experiment_modernnca_evidence_v2 as E
from complete_shenzhen_evidence_v2 import OUT as INPUT

BASE = E.OUT
OUT = E.M.REPO / 'outputs/modernnca_evidence_v2_extended_20261113'


def install():
    E.OUT = OUT; E.INPUT = INPUT


def prepare():
    E.M.guard()
    if OUT.exists():
        raise FileExistsError('Preserve convergence experiment')
    assert not (BASE / 'prediction_freeze.json').exists()
    assert not (BASE / 'results.json').exists()
    c = E.read(BASE / 'protocol.json')
    c.update(checkpoints=[0, 20, 40, 80, 160, 320], input_manifest=str(INPUT / 'manifest.json'),
        source_only_extension_reason='At80steps both trained candidates failed source-single safeguard; lr.001 recovered from601/623at20 to908at80. Extend once to320 before any target inference/scoring. Independently found2verified textual references are completed in the routed snapshot.',
        initial_source_results_seen=True, original_80step_experiment=str(BASE),
        extension_not_independent_confirmation=True, extension_limit_steps=320)
    manifest = E.read(INPUT / 'manifest.json')
    links = dict(manifest['inputs'])
    links.update({str(INPUT / p): h for p, h in manifest['files'].items()})
    links[str(INPUT / 'manifest.json')] = E.sha(INPUT / 'manifest.json')
    for p, h in E.read(BASE / 'inputs_sha256.json').items():
        links[p] = h
    for p in [BASE / 'protocol.json', BASE / 'selection_freeze.json', BASE / 'input_route_audit.json']:
        links[str(p)] = E.sha(p)
    for p, h in links.items():
        assert E.sha(p) == h, p
    E.write(OUT / 'protocol.json', c)
    E.write(OUT / 'inputs_sha256.json', links)
    code = E.read(BASE / 'code_sha256.json')
    code.update({str(Path(__file__).relative_to(E.M.REPO)): E.sha(Path(__file__)),
                 'scripts/complete_shenzhen_evidence_v2.py': E.sha(E.M.REPO / 'scripts/complete_shenzhen_evidence_v2.py')})
    E.write(OUT / 'code_sha256.json', code)
    for p in code:
        dest = OUT / 'code' / p; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((E.M.REPO / p).read_bytes())
    E.write(OUT / 'protocol_receipt.json', dict(sha256=E.sha(OUT / 'protocol.json'), target_gold_read=False))
    print('SOURCE-ONLY EXTENSION SEALED', OUT, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['prepare', 'train', 'infer', 'score', 'audit_inputs', 'audit_outputs'])
    phase = ap.parse_args().phase
    if phase == 'prepare':
        prepare()
    else:
        install()
        if phase.startswith('audit_'):
            import audit_modernnca_evidence_v2 as A
            getattr(A, phase.removeprefix('audit_'))()
        else:
            getattr(E, phase)()
