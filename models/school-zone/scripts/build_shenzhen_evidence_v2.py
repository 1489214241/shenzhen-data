"""Build a routed, immutable input snapshot, without target scoring access."""
import collections
import copy
import pickle
from pathlib import Path

import numpy as np

import experiment_modernnca_shenzhen_multilabel as M
from shenzhen_reviewed_supplements import ReviewedSupplements
from zoning.scope_evidence_v2 import (append_scope_features, compile_base_branches,
                                    evidence_routes, exact_named_relations)

OUT = M.REPO / 'data/深圳市/全链路输入_v2_20261113'
LATE = M.REPO / 'data/深圳市/累计文字监督与参照_20261113'
SCOPE = M.REPO / 'data/深圳市/完整范围特征输入_20261112'
RAW = M.REPO / 'data/深圳市/边界合取单项修复输入_20261112'
FILES = {}


def read(p):
    FILES[str(p)] = M.F.sha(p)
    return M.read(p)


def pkl(p):
    FILES[str(p)] = M.F.sha(p)
    return M.pkl(p)


def npz(p):
    FILES[str(p)] = M.F.sha(p)
    return dict(np.load(p))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as handle:
        pickle.dump(value, handle, protocol=5)


def main():
    M.guard()
    if OUT.exists():
        raise FileExistsError('Preserve frozen input snapshot')
    c = M.verify()
    audit = {}
    for d in c['sources'] + c['targets']:
        old = pkl(M.CURRENT / 'inputs/current' / f'{d}.pkl' if d in c['targets'] else M.OLD / 'inputs' / f'{d}.pkl')
        original_labels = npz(M.OLD / 'inputs' / f'{d}_labels.npz')
        states = npz(SCOPE / d / 'composed_scope.npz')['states']
        y = copy.deepcopy(original_labels)
        if d in c['targets']:
            # This is a new controlled candidate, not adoption of the old failed retrain.
            x = pkl(LATE / d / 'reference_inputs.pkl')
            dist = pkl(LATE / d / 'district.pkl')
            meta = read(M.CURRENT / 'metadata' / f'{d}.json')
            reviewed = read(M.ROOT / 'scope_graph_refresh_review/graphs' / f'{d}_changes.json')['school_inventory']
            policy = read(M.ROOT / 'round08_reference_retraining/policies' / f'{d}.json')
            assert dist.answers == {} and dist.train_answers == {}
            assert dist.site_ids == [s['coordinate_group_id'] for s in meta['sites']]
            assert dist.school_ids == [s['school_id'] for s in meta['schools']]
            qi = {s: q for q, s in enumerate(dist.site_ids)}
            sj = {s: j for j, s in enumerate(dist.school_ids)}
            ledger = {}

            def add(q, j, kind, proof, positive=True):
                key = (q, j, positive)
                row = ledger.setdefault(key, dict(site_id=dist.site_ids[q], school_id=dist.school_ids[j],
                    q=q, school_index=j, year=2026, stage='小学', positive=positive,
                    relation='base_scope_membership', evidence=[]))
                row['evidence'].append(dict(kind=kind, proof=proof))

            for q, j in np.argwhere(original_labels['labels'] == 1):
                kind = 'legacy_named_match' if meta['known_kind'].get(str(q)) == 'named' else 'legacy_boundary'
                add(int(q), int(j), kind, dict(snapshot=str(M.OLD / 'inputs' / f'{d}.json'),
                    distinction='Prior rule estimate; do not silently treat as verified eligibility'))
            supp = ReviewedSupplements(d)
            FILES.update(supp.hashes)
            for site, groups in supp.positive.items():
                for kind, schools in groups.items():
                    path = supp.paths.get('explicit_scope' if kind == 'explicit_positive_repair' else kind)
                    if kind == 'reviewed_administrative_identity':
                        path = M.ROOT / 'administrative_identity_review_v1/school_facts' / f'{d}.json'
                    content = read(path)
                    facts = content.get('positive_repairs', []) if kind == 'explicit_positive_repair' else content.get('facts', [])
                    for school in schools:
                        proof = [f for f in facts if f['site_id'] == site and f['school_id'] == school]
                        assert proof, (site, school, kind)
                        add(qi[site], sj[school], kind, dict(source_file=str(path), source_sha256=FILES[str(path)], facts=proof))
            negative = read(supp.paths['explicit_scope'])['negative_facts']
            for fact in negative:
                assert not fact['conflicts_with_prior_positive']
                add(qi[fact['site_id']], sj[fact['school_id']], 'explicit_negative', fact, False)
            for fact in exact_named_relations(reviewed, meta['sites']):
                add(fact['q'], sj[fact['school_id']], fact['kind'], fact)
            y['labels'][:] = -1
            refs = set(zip(map(int, x.ref_sites), map(int, x.ref_schools)))
            constraints = {s: dict(positive=[], negative=[]) for s in dist.site_ids}
            for (q, j, positive), row in ledger.items():
                route = evidence_routes([e['kind'] for e in row['evidence']], positive)
                row['routes'] = route
                row['reference_currently_present'] = (q, j) in refs
                if route['training']:
                    value = int(positive)
                    assert y['labels'][q, j] in [-1, value], (d, row)
                    y['labels'][q, j] = value
                if route['hard_constraint']:
                    constraints[row['site_id']]['positive' if positive else 'negative'].append(row['school_id'])
            # Reference bank membership must have a provenance record, including weak records.
            missing = refs - {(q, j) for q, j, positive in ledger if positive}
            if missing:
                raise ValueError(f'Unrouted reference relations: {d}: {sorted(missing)}')
            for value in constraints.values():
                value['positive'] = sorted(set(value['positive']))
                value['negative'] = sorted(set(value['negative']))
                assert not set(value['positive']) & set(value['negative'])
            y['train_ids'] = np.flatnonzero((y['labels'] >= 0).any(1))
            assert not len(y['valid_ids']) and not (y['labels'] >= 0).all(1).any()
            M.write(OUT / d / 'evidence_ledger.json', list(ledger.values()))
            M.write(OUT / d / 'constraints.json', constraints)
            branches = compile_base_branches(reviewed, policy)
            M.write(OUT / d / 'base_branches.json', branches)
            M.write(OUT / d / 'full_scope_inventory.json', reviewed)
            M.write(OUT / d / 'metadata.json', meta)
            # Full formulas and observed clause states travel together; never
            # flatten an unresolved AND to a positive list or a known negative.
            for name in ['aligned_observations.json', 'policy_observations.json', 'reviewed_policy.json']:
                M.write(OUT / d / name, read(RAW / d / name))
            observations = npz(RAW / d / 'observation_arrays.npz')
            np.savez_compressed(OUT / d / 'observation_arrays.npz', **observations)
            audit[d] = dict(schools=x.school_n, sites=x.n, references=len(refs),
                reference_schools=len(set(x.ref_schools)), ledger_relations=len(ledger),
                strong_positive=int((y['labels'] == 1).sum()), strong_negative=int((y['labels'] == 0).sum()),
                weak_positive_relations=sum(r['positive'] and not r['routes']['training'] for r in ledger.values()),
                formerly_observed_now_soft_only=int(((original_labels['labels'] == 1) & (y['labels'] == -1)).sum()),
                supported_base_schools=len({b['school_id'] for b in branches}),
                university_base_schools_retained=len({b['school_id'] for b in branches if 'university' in b['relation_tags']}),
                target_complete_labels=0)
        else:
            x = old
            audit[d] = dict(sites=x.n, schools=x.school_n, source_labels_unchanged=True,
                complete_train=int((y['labels'][y['train_ids']] >= 0).all(1).sum()),
                partial_train=int((~(y['labels'][y['train_ids']] >= 0).all(1)).sum()))
        routed = append_scope_features(x, states)
        # Both arms have identical dimensions. Control has an absent block,
        # not a falsely known-outside observation. All native columns preserved.
        control = append_scope_features(old, np.full((old.n, old.school_n), -1, np.int8))
        control.pair[..., -3:] = 0; control.anchor[..., -3:] = 0
        for value in [control, routed]:
            np.testing.assert_array_equal(value.self_mask, value.groups[:, None] == value.groups[value.ref_sites][None])
            assert value.np_ == 44 and value.na == 30
            assert np.isfinite(value.pair).all() and np.isfinite(value.anchor).all()
        np.testing.assert_array_equal(control.pair[..., :41], old.pair)
        np.testing.assert_array_equal(control.anchor[..., :27], old.anchor)
        np.testing.assert_array_equal(routed.pair[..., :41], x.pair)
        np.testing.assert_array_equal(routed.anchor[..., :27], x.anchor)
        save(OUT / d / 'control.pkl', control)
        save(OUT / d / 'routed.pkl', routed)
        np.savez_compressed(OUT / d / 'control_labels.npz', **original_labels)
        np.savez_compressed(OUT / d / 'routed_labels.npz', **y)
        np.savez_compressed(OUT / d / 'whole_scope.npz', states=states)
        print('ROUTED', d, audit[d], flush=True)
    code = [Path(__file__), M.REPO / 'zoning/scope_evidence_v2.py', M.REPO / 'tests/test_scope_evidence_v2.py']
    for p in code:
        FILES[str(p)] = M.F.sha(p)
    for p, h in FILES.items():
        assert M.F.sha(p) == h, p
    M.write(OUT / 'protocol.json', dict(sources=c['sources'], targets=c['targets'],
        source_complete_and_partial_labels_preserved=True, new_evidence_acquired=False,
        control='Original inputs and labels with three zero channels; matched architecture.',
        routed='Archived reviewed native features/references plus uniform whole-scope observations; typed target partial supervision; no estimated labels as hard facts.',
        prediction='School-wise membership hypotheses, not a fixed finite whole-set catalogue; only verified textual constraints may be hard applied.',
        compound_input_intervention=True, causal_limit='Matched total pipeline comparison does not isolate each input component.',
        original_snapshots_unchanged=True, target_gold_read=False, target_polygon_read=False,
        independent_validation=False, adopted=False))
    M.write(OUT / 'input_audit.json', audit)
    M.write(OUT / 'input_links.json', FILES)
    M.write(OUT / 'manifest.json', dict(inputs=FILES, files={str(p.relative_to(OUT)): M.F.sha(p)
        for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('INPUT SNAPSHOT SEALED', OUT, flush=True)


if __name__ == '__main__':
    main()
