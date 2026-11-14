"""Synthetic geographic cases: crossing gates and distinct-unit distance ratios."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import shapely

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_hangzhou_community_methods import road_aware
from shenzhen_community_comparison import nearest_separated


def lines(cls):
    return SimpleNamespace(tree=shapely.STRtree([shapely.LineString([(100,-50),(100,50)])] if cls else []),cls=[cls] if cls else [])


class GeographicCases(unittest.TestCase):
    def test_no_road_strict_and_outside_radius(self):
        a=road_aware(np.array([[0,0],[3000,0]]),np.array([[350,0]]),lines(None))
        self.assertEqual(a['strict'].tolist(),[True,False])
        self.assertEqual(a['point'].tolist(),[0,-1])

    def test_prefers_farther_unblocked_point(self):
        a=road_aware(np.array([[0,0]]),np.array([[200,0],[0,600]]),lines('major'))
        self.assertEqual(a['point'].tolist(),[1])
        self.assertEqual(a['strict'].tolist(),[False])
        self.assertEqual(a['loose'].tolist(),[True])

    def test_soft_and_hard_crossing(self):
        for cls,expected in [('local',True),('tertiary',True),('major',False),('secondary',False),('rail',False),('river',False)]:
            a=road_aware(np.array([[0,0]]),np.array([[300,0]]),lines(cls))
            self.assertFalse(a['strict'][0]);self.assertEqual(bool(a['loose'][0]),expected)

    def test_second_distinct_identity_and_ambiguity(self):
        pts=np.array([[100,0],[110,0],[250,0]])
        a=nearest_separated(np.array([[0,0]]),pts,['A','A','B'])
        self.assertTrue(a['strict'][0]);self.assertEqual(a['second_distance'][0],250)
        b=nearest_separated(np.array([[0,0]]),pts,['A','B','C'])
        self.assertFalse(b['strict'][0]);self.assertFalse(b['loose'][0])


if __name__=='__main__':unittest.main()
