"""Complete two ledger-proven reference routes, preserving the first snapshot."""
import os
import pickle
from pathlib import Path

import numpy as np

import build_shenzhen_evidence_v2 as B
from shenzhen_partial_reference_extension import extend

OUT = B.M.REPO / 'data/深圳市/全链路输入_v2_complete_20261113'


def main():
    B.M.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed snapshot')
    manifest = B.M.read(B.OUT / 'manifest.json')
    for rel, h in manifest['files'].items():
        assert B.M.F.sha(B.OUT / rel) == h
        dest = OUT / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(B.OUT / rel, dest)
    links = dict(manifest['inputs'])
    links.update({str(B.OUT / rel): h for rel, h in manifest['files'].items()})
    links[str(B.OUT / 'manifest.json')] = B.M.F.sha(B.OUT / 'manifest.json')
    audit = B.M.read(B.OUT / 'input_audit.json'); completion = {}
    for d in ['龙华区', '龙岗区']:
        ledger = B.M.read(OUT / d / 'evidence_ledger.json')
        missing = [(r['q'], r['school_index']) for r in ledger if r['positive'] and r['routes']['training'] and not r['reference_currently_present']]
        if not missing:
            completion[d] = dict(added_reference_pairs=0); continue
        x = B.M.pkl(OUT / d / 'routed.pkl')
        dist_path = B.LATE / d / 'district.pkl'; rule_path = B.LATE / d / 'rules.pkl'
        fact_path = B.LATE / d / 'textual_facts.json'
        dist = B.M.pkl(dist_path); rules = B.M.pkl(rule_path)
        known = {int(k): v for k, v in B.M.read(fact_path)['new_boundary_support'].items()}
        after = {k: v for k, v in known.items() if k not in {q for q, s in missing}}
        out, added, held = extend(x, dist, rules, missing, after, known)
        assert set(added) == set(missing) and not held
        states = np.load(OUT / d / 'whole_scope.npz')['states']
        out.pair[..., -3:] = (states[:, out.ref_schools, None] == np.array([-1, 0, 1])).astype(np.float32)
        # Atomic replacement: never truncate an inherited hard link in place.
        dest = OUT / d / 'routed.pkl'; temp = dest.with_suffix('.building')
        with temp.open('wb') as handle:
            pickle.dump(out, handle, protocol=5)
        temp.replace(dest)
        for r in ledger:
            if (r['q'], r['school_index']) in added and r['positive']:
                r['reference_currently_present'] = True
        B.M.write(OUT / d / 'evidence_ledger.json', ledger)
        completion[d] = dict(added_reference_pairs=len(added), pairs=added,
            labels_unchanged=True, school_anchor_features_unchanged=True,
            old_reference_pairs_preserved=True, boundary_support_removed=sorted(set(known)-set(after)))
        audit[d]['references'] = len(out.ref_sites)
        audit[d]['reference_schools'] = len(set(out.ref_schools))
        for p in [dist_path, rule_path, fact_path]:
            links[str(p)] = B.M.F.sha(p)
    for p in [Path(__file__), B.M.REPO / 'scripts/shenzhen_partial_reference_extension.py']:
        links[str(p)] = B.M.F.sha(p)
    B.M.write(OUT / 'input_audit.json', audit)
    B.M.write(OUT / 'reference_completion.json', completion)
    B.M.write(OUT / 'input_links.json', links)
    protocol = B.M.read(B.OUT / 'protocol.json')
    protocol['reference_completion'] = 'All admitted textual positives now represented in target reference bank; two unconditional named branches in mixed-household school added.'
    B.M.write(OUT / 'protocol.json', protocol)
    for rel, h in manifest['files'].items():
        assert B.M.F.sha(B.OUT / rel) == h, 'Parent snapshot mutated'
    B.M.write(OUT / 'manifest.json', dict(inputs=links, files={str(p.relative_to(OUT)): B.M.F.sha(p)
        for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('REFERENCE COMPLETION SEALED', completion, flush=True)


if __name__ == '__main__':
    main()
