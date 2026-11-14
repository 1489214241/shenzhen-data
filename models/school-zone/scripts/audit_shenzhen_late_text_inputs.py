"""Independently recount labels, provenance and each added reference column."""
import pickle
import numpy as np
import torch
import build_shenzhen_late_text_inputs as B
from zoning import evidence as EV

F, R = B.F, B.R


def main():
    F.guard(); R.check(B.OUT, 'manifest.json')
    for p, h in F.read(B.OUT/'input_links.json')['files'].items(): assert F.sha(p) == h
    dest = R.ROOT/'late_text_inputs_independent_audit.json'
    if dest.exists(): raise FileExistsError('Preserve numerical input audit')
    checks = {}
    for d in R.TARGETS:
        with (B.BASE/d/'inputs.pkl').open('rb') as h: old = pickle.load(h)
        with (B.OUT/d/'reference_inputs.pkl').open('rb') as h: new = pickle.load(h)
        with (B.OUT/d/'district.pkl').open('rb') as h: dist = pickle.load(h)
        with (B.OUT/d/'rules.pkl').open('rb') as h: rules = pickle.load(h)
        oy = dict(np.load(B.BASE/d/'labels.npz')); ny = dict(np.load(B.OUT/d/'labels.npz'))
        assert F.sha(B.OUT/d/'control_inputs.pkl') == F.sha(B.BASE/d/'inputs.pkl')
        qi = {s: i for i, s in enumerate(dist.site_ids)}; sj = {s: i for i, s in enumerate(dist.school_ids)}
        expected = set()
        paths = [(R.OUT/'facts'/f'{d}.json', 'facts'),
                 (R.ROOT/'round20_residential_membership/facts'/f'{d}.json', 'facts'),
                 (R.ROOT/'named_region_facts_v1'/d/'facts.json', 'facts'),
                 (R.ROOT/'explicit_scope_facts_v1'/d/'facts.json', 'positive_repairs'),
                 (B.IDENTITIES/'school_facts'/f'{d}.json', 'facts')]
        for p, key in paths:
            for fact in F.read(p)[key]:
                q, s = qi[fact['site_id']], sj[fact['school_id']]
                assert oy['labels'][q, s] != 0
                if oy['labels'][q, s] == -1: expected.add((q, s))
        actual = {tuple(p) for p in np.argwhere(oy['labels'] != ny['labels'])}
        assert expected == actual
        for q, s in expected: assert ny['labels'][q, s] == 1
        np.testing.assert_array_equal(oy['labels'] == 0, ny['labels'] == 0)
        np.testing.assert_array_equal(ny['train_ids'], np.flatnonzero((ny['labels'] >= 0).any(1)))
        for k in ['valid_ids', 'valid_labels']: np.testing.assert_array_equal(oy[k], ny[k])
        facts = F.read(B.OUT/d/'textual_facts.json')
        before = {int(q): ss for q, ss in facts['old_boundary_support'].items()}
        after = {int(q): ss for q, ss in facts['new_boundary_support'].items()}
        assert after == {q: ss for q, ss in before.items() if q not in {q for q, s in expected}}
        for q, ss in after.items(): assert ss == before[q]
        nr = len(old.ref_sites); added = list(zip(map(int, new.ref_sites[nr:]), map(int, new.ref_schools[nr:])))
        assert set(added) == {p for p in expected if p[0] not in dist.not_reference}
        assert list(zip(new.ref_sites[:nr], new.ref_schools[:nr])) == list(zip(old.ref_sites, old.ref_schools))
        np.testing.assert_array_equal(old.anchor, new.anchor)
        for k in ['groups', 'y', 'supervised']: np.testing.assert_array_equal(getattr(old, k), getattr(new, k))
        for k, sl in old.pair_blocks.items():
            if k != 'learned_boundary': np.testing.assert_array_equal(old.pair[:, :, sl], new.pair[:, :nr, sl])
        lines = EV.lines(dist.lines_key or dist.city)
        border = EV.learned_boundary(lines, dist.coords, dist.groups, after, new.ref_sites)
        np.testing.assert_array_equal(border, new.pair[:, :, new.pair_blocks['learned_boundary']])
        for index, (q, school) in enumerate(added, nr):
            col = new.pair[:, index]; delta = dist.coords[q]-dist.coords
            distance = np.linalg.norm(delta, axis=1)
            geometry = np.stack([np.minimum(distance/300, 50), np.log1p(distance/100),
                np.clip(delta[:, 0]/1000, -5, 5), np.clip(delta[:, 1]/1000, -5, 5), np.exp(-distance/500)], axis=1).astype(np.float32)
            np.testing.assert_array_equal(col[:, new.pair_blocks['distance']], geometry)
            assert not col[:, new.pair_blocks['named_reference']].any()
            np.testing.assert_array_equal(col[:, new.pair_blocks['policy']], old.anchor[:, school, old.anchor_blocks['policy']])
            qs = rules.statuses[:, school]; rs = rules.statuses[q, school]
            for key, predicate in [('agreement', qs == rs), ('disagreement', qs != rs)]:
                np.testing.assert_array_equal(col[:, new.pair_blocks[key]], ((qs >= 0) & (rs >= 0) & predicate).astype(np.float32))
            same = (dist.community_name == dist.community_name[q]) & (dist.community_name != '')
            both = dist.community_strict & dist.community_strict[q]
            np.testing.assert_array_equal(col[:, new.pair_blocks['same_community']], np.stack([same & both, ~same & both], axis=1).astype(np.float32))
            assert new.self_mask[q, index]
        new_sites = sorted(set(new.ref_sites)-set(old.ref_sites))
        if new_sites:
            cross, within = EV.crossing(lines, dist.coords, dist.coords[new_sites])
            for q, i in zip(new_sites, range(len(new_sites))):
                for col in np.flatnonzero(new.ref_sites == q):
                    np.testing.assert_array_equal(new.pair[:, col, new.pair_blocks['crossing']], cross[:, i])
                    np.testing.assert_array_equal(new.pair[:, col, new.pair_blocks['within_2km']], within[:, i])
        np.testing.assert_array_equal(new.self_mask, new.groups[:, None] == new.groups[new.ref_sites][None])
        groups = new.groups[new.ref_sites]
        bias = np.asarray([-np.log(sum(groups == g)) for g in groups], np.float32)
        np.testing.assert_array_equal(new.group_bias, bias)
        expected_membership = np.zeros((len(new.ref_schools)+new.school_n, new.school_n), np.float32)
        for i, s in enumerate(list(new.ref_schools)+list(range(new.school_n))): expected_membership[i, s] = 1
        np.testing.assert_array_equal(new.membership.cpu().numpy(), expected_membership)
        conflicts = [dict(site_id=dist.site_ids[q], school_id=dist.school_ids[s]) for q, s in expected if rules.negative[q, s]]
        checks[d] = dict(textual_labels=len(expected), new_references=len(added), new_reference_sites=len(new_sites),
            old_negative_labels_unchanged=True, school_features_unchanged=True, source_facts_recounted=True,
            own_property_refs_masked=True, boundary_support_withdrawn=sorted(set(before)-set(after)),
            negative_feature_conflicts=conflicts, all_added_column_formulas_recomputed=True)
        print('TEXT_INPUTS_VERIFIED', d, checks[d], flush=True)
    assert sum(v['textual_labels'] for v in checks.values()) == 19
    assert not any(v['negative_feature_conflicts'] for v in checks.values())
    F.write(dest, dict(districts=checks, input_manifest_sha256=F.sha(B.OUT/'manifest.json'),
        target_gold_read=False, target_polygon_read=False, adopted=False))


if __name__ == '__main__': main()
