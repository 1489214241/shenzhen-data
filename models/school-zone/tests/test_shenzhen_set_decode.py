import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_set_decode import decode


class SetDecodeTests(unittest.TestCase):
    def test_probability_split_among_identical_sets(self):
        best,mass,keys=decode([[.3,.3,.4]],[['A','B'],['B','A'],['C']])
        self.assertEqual(best,[('A','B')]);np.testing.assert_allclose(mass,[[.6,.4]])

    def test_overlap_does_not_imply_union_or_transitive_closure(self):
        best,mass,keys=decode([[.3,.3,.4]],[['A','B'],['B','C'],['D']])
        self.assertEqual(best,[('D',)]);self.assertEqual(len(keys),3)

    def test_singleton_exactly_matches_argmax(self):
        p=np.random.default_rng(12).dirichlet(np.ones(5),size=20)
        best,_,_=decode(p,[[s] for s in range(5)])
        self.assertEqual(best,[(int(s),) for s in p.argmax(1)])

    def test_tie_keeps_original(self):
        best,_,_=decode([[.25,.25,.5]],[['A','B'],['A','B'],['C']])
        self.assertEqual(best,[('C',)])

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):decode([[.2,.2]],[['A'],['B']])


if __name__=='__main__':unittest.main()
