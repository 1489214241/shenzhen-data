import unittest
from types import SimpleNamespace

import numpy as np

from zoning.scope_evidence_v2 import (append_scope_features, apply_verified_constraints,
                                    compile_base_branches, evidence_routes)


class ScopeEvidenceTests(unittest.TestCase):
    def row(self, school, ast, status='individual_scope_requires_site_applicability'):
        return dict(school_id=school, scope_status=status, ast=ast, source_text='official', evidence=['official'])

    def policy(self):
        return dict(year=2026, stage='小学', relations=[dict(kind='university', school_ids=['A', 'B'])])

    def test_university_preserves_each_base_without_member_expansion(self):
        rr = compile_base_branches([self.row('A', dict(op='membership', text='甲社区')),
                                    self.row('B', dict(op='membership', text='乙社区'))], self.policy())
        self.assertEqual([(r['school_id'], r['expression']) for r in rr],
                         [('A', ('membership', '甲社区')), ('B', ('membership', '乙社区'))])
        self.assertTrue(all(not r['university_member_expansion'] for r in rr))

    def test_household_or_keeps_only_unconditional_branch(self):
        a = dict(op='named', text='无条件花园')
        b = dict(op='conditional', condition='hukou', children=[dict(op='named', text='有条件村')])
        rr = compile_base_branches([self.row('A', dict(op='or', children=[a, b]),
                                            'mixed_qualification_partial_scope')], self.policy())
        self.assertEqual([r['expression'] for r in rr], [('named', '无条件花园')])

    def test_and_unknown_and_exclusion_are_never_discarded(self):
        a = dict(op='membership', text='甲社区'); u = dict(op='unknown', text='未明条件')
        self.assertEqual(compile_base_branches([self.row('A', dict(op='and', children=[a, u]))], self.policy()), [])
        rr = compile_base_branches([self.row('A', dict(op='minus', children=[a, dict(op='named', text='排除花园')]))], self.policy())
        self.assertEqual(rr[0]['expression'][0], 'minus')

    def test_estimates_not_training_or_hard_constraints(self):
        route = evidence_routes(['legacy_boundary', 'road_corridor'])
        self.assertFalse(route['training']); self.assertFalse(route['hard_constraint'])
        self.assertTrue(route['reference'])
        self.assertTrue(evidence_routes(['community_round07'])['training'])
        self.assertFalse(evidence_routes(['community_round07'])['complete_school_set'])

    def test_unknown_state_and_scope_reference_alignment(self):
        x = SimpleNamespace(n=2, school_n=2, np_=1, na=1, pair=np.ones((2, 2, 1)),
                            anchor=np.ones((2, 2, 1)), ref_schools=np.array([1, 0]),
                            pair_blocks={'a': slice(0, 1)}, anchor_blocks={'a': slice(0, 1)})
        y = append_scope_features(x, np.array([[-1, 1], [0, -1]]))
        np.testing.assert_array_equal(y.pair[0, :, 1:], [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(y.anchor[1, :, 1:], [[0, 1, 0], [1, 0, 0]])
        self.assertEqual(x.pair.shape[-1], 1)

    def test_explicit_negative_blocks_positive_and_no_transitive_expansion(self):
        self.assertEqual(apply_verified_constraints(['A', 'B'], ['C'], ['B']), ['A', 'C'])
        self.assertEqual(apply_verified_constraints(['A'], [], ['A']), [])
        with self.assertRaises(ValueError):
            apply_verified_constraints(['A'], ['B'], ['B'])


if __name__ == '__main__':
    unittest.main()
