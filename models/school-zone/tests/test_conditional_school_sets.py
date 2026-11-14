import unittest
from zoning.conditional_school_sets import catalogue, site_candidates, select, named_sufficient


class SetTests(unittest.TestCase):
    def policy(self):
        return dict(school_ids=['A','B','C'], school_scope_inventory=[], relations=[],year=2026,stage='小学')

    def test_no_transitive_union(self):
        g={'edges':[dict(trigger_school_id='A',implied_school_id='B'),dict(trigger_school_id='B',implied_school_id='C')]}
        sets=[set(r['school_ids']) for r in catalogue(self.policy(),[g])]
        self.assertIn({'A','B'},sets);self.assertIn({'B','C'},sets);self.assertNotIn({'A','B','C'},sets)

    def test_university_not_expanded(self):
        p=self.policy();p['relations']=[dict(kind='university',school_ids=['A','B'])]
        self.assertTrue(all(len(r['school_ids'])==1 for r in catalogue(p,[])))

    def test_unknown_not_certified_and_exclusion_blocks_whole_set(self):
        r=dict(candidate_id='ab',school_ids=['A','B'],exact_site_names=['花园一期'])
        site=dict(building_name='花园二期')
        candidates,audit=site_candidates([r],site)
        self.assertEqual(candidates[0]['applicability'],'unknown')
        with self.assertRaises(ValueError):site_candidates([r],site,excluded=['B'])
        self.assertFalse(candidates[0]['complete_school_set'])

    def test_coordinate_alias_does_not_spread_identity(self):
        r=dict(candidate_id='ab',school_ids=['A','B'],exact_site_names=['花园一期'])
        c,_=site_candidates([r],dict(building_name='花园一期',coordinate_group_building_names=['花园一期','花园二期']))
        self.assertEqual(c[0]['applicability'],'unknown')

    def test_partial_positive_is_not_complete(self):
        c,_=site_candidates(catalogue(self.policy(),[]),dict(building_name='x'),lower_bound=['A'])
        self.assertTrue(all('A' in r['school_ids'] for r in c))
        self.assertTrue(all(not r['complete_school_set'] for r in c))

    def test_weak_extra_member_is_penalized(self):
        c=[dict(school_ids=['A']),dict(school_ids=['A','B'])]
        self.assertEqual(select([.99,.01],c,['A','B'],1)['school_ids'],['A'])
        self.assertEqual(select([.5,.5],c,['A','B'],1)['school_ids'],['A','B'])

    def test_conditions_not_stripped(self):
        leaf=dict(op='named',text='花园')
        for op in ['and','minus','conditional']:
            self.assertEqual(named_sufficient(dict(op=op,children=[leaf])),set())
        self.assertEqual(named_sufficient(dict(op='or',children=[leaf])),{'花园'})


if __name__=='__main__':unittest.main()
