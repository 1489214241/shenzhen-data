import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shenzhen_full_inputs as F
import build_shenzhen_model_inputs as B


class InputTests(unittest.TestCase):
    def test_no_south_prefix_identity(self):
        def response(name):
            return dict(term='大富社区',response=dict(results=[dict(name=name,uid='test',area='龙华区',location=dict(lng=114.05,lat=22.7))]))
        self.assertEqual(F.candidates([response('南大富社区居民委员会')],'龙华区','大富社区'),[])
        self.assertEqual(len(F.candidates([response('大富社区居民委员会')],'龙华区','大富社区')),1)
        self.assertEqual(len(F.candidates([response('观澜街道大富社区居民委员会')],'龙华区','大富社区')),1)

    def test_subcommittee_not_parent(self):
        self.assertEqual(F.leaf('高峰社区鹊山居委会'),'鹊山')
        self.assertEqual(F.leaf('同胜社区居委会'),'同胜')

    def test_direction_conjunction(self):
        p=B.parse_scope('贝尔路以南—五和大道以东—布龙路以北—坂雪岗大道以西')
        self.assertEqual(p['op'],'road')
        self.assertEqual(p['pairs'],[['贝尔路','南'],['五和大道','东'],['布龙路','北'],['坂雪岗大道','西']])

    def test_conditional_community_is_not_whole(self):
        p=B.parse_scope('水围社区居委会（华旺路以北部分）')
        self.assertEqual(p['op'],'and')
        self.assertEqual([c['op'] for c in p['children']],['membership','road'])

    def test_excluded_area_stays_unknown(self):
        p=B.parse_scope('平湖社区（熙璟城片区除外）')
        self.assertEqual(p['op'],'and')
        self.assertEqual(p['children'][1]['op'],'unknown')

    def test_native_cross_sections_disagreement(self):
        pts=np.array([[114.05,22.65],[114.15,22.65],[114.,22.65]])
        lines=[dict(type='LineString',coordinates=[[x,22.6],[x,22.7]]) for x in [114.,114.1]]
        r=dict(status='ready',predicate='line_side',inward='东',geometries=lines)
        np.testing.assert_array_equal(B.strict_line(r,pts),[-1,1,-1])

    def test_multiple_ranges_are_union(self):
        p=B.parse_scope('范围一：甲大道以东、乙路以北范围。\n范围二：丙路以西、丁街以南范围。')
        self.assertEqual(p['op'],'or')
        self.assertEqual(len(p['children']),2)


if __name__=='__main__':unittest.main()
