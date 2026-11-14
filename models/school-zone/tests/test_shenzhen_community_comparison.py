import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_community_comparison import nearest_separated, membership_terms, direct_estimates, identity


class CommunityComparisonTests(unittest.TestCase):
    def test_second_unit_ignores_duplicate_office(self):
        r = nearest_separated([[0, 0]], [[100, 0], [110, 0], [300, 0]], ['a', 'a', 'b'])
        self.assertTrue(r['strict'][0])
        self.assertEqual(r['second_distance'][0], 300)

    def test_partial_and_exclusions_are_not_whole_membership(self):
        ast = {'op':'minus','children':[{'op':'membership','text':'A'}, {'op':'membership','text':'B'}]}
        self.assertEqual(list(membership_terms(ast)), [('A', True)])
        self.assertEqual(list(membership_terms({'op':'and','children':[{'op':'membership','text':'C'}, {'op':'road'}]})), [('C', True)])

    def test_rule_respects_unknown_multischool_and_exclusion(self):
        cs = np.zeros((4, 2, 4)); cs[0, 0, 0]=1;cs[1, :, 0]=1;cs[2, 0, 0]=1;cs[3, 0, 0]=1
        negative=np.zeros((4,2),bool);negative[2,0]=True
        self.assertEqual(direct_estimates(cs, np.ones(4,bool), {3:[1]}, negative), {0:[0]})

    def test_parent_and_subcommittee_not_collapsed(self):
        p={'uid':'x','name':'丹湖社区居民委员会','policy_names':['丹湖社区','丹湖社区丹湖居委会']}
        self.assertEqual(identity(p),'ambiguous-poi:x')


if __name__ == '__main__':
    unittest.main()
