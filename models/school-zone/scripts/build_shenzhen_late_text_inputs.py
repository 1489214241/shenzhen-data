"""Stage textual-only supervision and its separately controlled reference bank."""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment_shenzhen_explicit_negative_training as E
import build_shenzhen_community_reference_inputs as OLD
from shenzhen_partial_reference_extension import extend

F, R = E.F, E.R
BASE = E.B.OUT
INVENTORY = R.ROOT/'late_supervision_inventory_20261113'
IDENTITIES = R.ROOT/'administrative_identity_review_v1'
OUT = F.REPO/'data/深圳市/累计文字监督与参照_20261113'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f: pickle.dump(value, f, protocol=5)


def collect(d, dist, y):
    pending = {}; geographic = []
    qi = {v: i for i, v in enumerate(dist.site_ids)}; sj = {v: i for i, v in enumerate(dist.school_ids)}
    for row in F.read(INVENTORY/d/'inventory.json')['missing_label_relations']:
        pair = (qi[row['site_id']], sj[row['school_id']])
        assert pair == (row['site_index'], row['school_index']) and y[pair] == -1
        if row['has_textual_proof']:
            assert any(s['evidence_type'] == 'textual_proof' for s in row['sources'])
            pending[pair] = dict(row, provenance='cumulative_textual_facts', complete_school_set=False)
        else: geographic.append(row)
    for fact in F.read(IDENTITIES/'school_facts'/f'{d}.json')['facts']:
        kinds = {w['kind'] for w in fact['proof']}
        assert {'official_community_membership', 'reviewed_administrative_identity'} <= kinds
        assert not fact['complete_school_set'] and not fact['enrollment_eligibility']
        q, s = qi[fact['site_id']], sj[fact['school_id']]
        assert y[q, s] != 0
        if y[q, s] == -1:
            assert (q, s) not in pending
            pending[q, s] = dict(site_id=fact['site_id'], school_id=fact['school_id'],
                site_index=q, school_index=s, building_name=dist.site_names[q], school_name=dist.school_names[s],
                provenance='reviewed_sole_committee_identity', evidence=fact, complete_school_set=False)
    return pending, geographic


def trusted_boundary_known(d, added):
    with (OLD.RAW/d/'rules.pkl').open('rb') as h: original = pickle.load(h)
    with (BASE/d/'district.pkl').open('rb') as h: dist = pickle.load(h)
    facts = F.read(OLD.FACTS/'facts'/f'{d}.json')['facts']
    _, _, _, touched = OLD.union_positive_references(original.known, original.kind, dist.site_ids, dist.school_ids, facts)
    before = {q: ss for q, ss in original.known.items() if q not in touched}
    changed = {q for q, s in added}
    after = {q: ss for q, ss in before.items() if q not in changed}
    return before, after


def main():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve textual input snapshot')
    links = {}
    for folder in [BASE, INVENTORY, IDENTITIES]:
        R.check(folder, 'manifest.json'); m = F.read(folder/'manifest.json')
        links[str(folder/'manifest.json')] = F.sha(folder/'manifest.json')
        links.update({str(folder/p): h for p, h in m['files'].items()})
        for p, h in m.get('sources', {}).items(): assert F.sha(p) == h; links[p] = h
        if (folder/'input_links.json').exists(): links.update(F.read(folder/'input_links.json')['files'])
    audit = {}
    for d in R.TARGETS:
        with (BASE/d/'district.pkl').open('rb') as h: dist = pickle.load(h)
        with (BASE/d/'rules.pkl').open('rb') as h: rules = pickle.load(h)
        with (BASE/d/'inputs.pkl').open('rb') as h: x = pickle.load(h)
        assert dist.answers == {} and dist.train_answers == {}
        old = dict(np.load(BASE/d/'labels.npz')); y = {k: v.copy() for k, v in old.items()}
        selected, geographic = collect(d, dist, y['labels'])
        for q, s in selected: y['labels'][q, s] = 1
        y['train_ids'] = np.flatnonzero((y['labels'] >= 0).any(1))
        assert {tuple(v) for v in np.argwhere(y['labels'] != old['labels'])} == set(selected)
        np.testing.assert_array_equal(y['labels'] == 0, old['labels'] == 0)
        assert not len(y['valid_ids'])
        before, after = trusted_boundary_known(d, selected)
        out, added, held = extend(x, dist, rules, list(selected), after, before)
        assert set(added) | set(held) == set(selected)
        np.testing.assert_array_equal(out.anchor, x.anchor)
        for key in x.pair_blocks:
            if key != 'learned_boundary':
                np.testing.assert_array_equal(out.pair[:, :len(x.ref_sites), x.pair_blocks[key]], x.pair[:, :, x.pair_blocks[key]])
        for key in ['groups', 'y', 'supervised']: np.testing.assert_array_equal(getattr(out, key), getattr(x, key))
        assert not len(out.supervised) and (out.y == -1).all()
        np.testing.assert_array_equal(out.self_mask, out.groups[:, None] == out.groups[out.ref_sites][None])
        dest = OUT/d; dest.mkdir(parents=True)
        shutil.copyfile(BASE/d/'inputs.pkl', dest/'control_inputs.pkl')
        for name in ['district.pkl', 'rules.pkl', 'model_metadata.json']: shutil.copyfile(BASE/d/name, dest/name)
        dump(dest/'reference_inputs.pkl', out)
        np.savez_compressed(dest/'labels.npz', **y)
        np.savez_compressed(dest/'control_labels.npz', **old)
        F.write(dest/'textual_facts.json', dict(facts=list(selected.values()), geographic_estimates_held=geographic,
            added_reference_pairs=added, excluded_reference_pairs=held,
            old_boundary_support={str(q): ss for q, ss in before.items()},
            new_boundary_support={str(q): ss for q, ss in after.items()}))
        newrows = sorted(set(y['train_ids'])-set(old['train_ids']))
        assert not set(x.groups[newrows]) & set(x.groups[old['train_ids']])
        audit[d] = dict(added_textual_labels=len(selected), geometric_labels_added=0, geographic_estimates_held=len(geographic),
            training_rows_before=len(old['train_ids']), training_rows_after=len(y['train_ids']), new_training_ids=list(map(int, newrows)),
            positives_before=int((old['labels'] == 1).sum()), positives_after=int((y['labels'] == 1).sum()),
            negatives_unchanged=True, reference_pairs_before=len(x.ref_sites), reference_pairs_after=len(out.ref_sites),
            added_references=len(added), new_reference_sites=len(set(out.ref_sites)-set(x.ref_sites)),
            schools_with_reference_before=len(set(x.ref_schools)), schools_with_reference_after=len(set(out.ref_schools)),
            held_reference_pairs=len(held), withdrawn_boundary_support_sites=sorted(set(before)-set(after)),
            changed_old_boundary_cells=int((out.pair[:, :len(x.ref_sites), x.pair_blocks['learned_boundary']] != x.pair[:, :, x.pair_blocks['learned_boundary']]).sum()),
            other_old_features_bitwise_equal=True, school_anchors_unchanged=True, same_property_mask=True)
        for p in [OLD.RAW/d/'rules.pkl', OLD.FACTS/'facts'/f'{d}.json']:
            links[str(p)] = F.sha(p)
        print('TEXTUAL_INPUTS', d, audit[d], flush=True)
    for p in [Path(__file__), F.REPO/'scripts/shenzhen_partial_reference_extension.py',
              F.REPO/'scripts/build_shenzhen_community_reference_inputs.py', F.REPO/'zoning/evidence.py']:
        links[str(p)] = F.sha(p); dest = OUT/'code'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    F.write(OUT/'input_audit.json', dict(districts=audit, target_gold_read=False, target_polygon_read=False,
        unknown_labels_not_negative=True, complete_school_sets_not_invented=True,
        model_training_performed=False, adopted=False))
    F.write(OUT/'input_links.json', dict(files=links))
    F.write(OUT/'manifest.json', dict(files={str(p.relative_to(OUT)): F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__ == '__main__': main()
