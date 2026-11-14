"""Scope-loss regression tests; no target evaluation labels."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shenzhen_aligned_policy as P
from build_shenzhen_model_inputs import parse_scope


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.parser=P.Parser([dict(school_name='甲实验学校'),dict(school_name='乙小学')],parse_scope)

    def test_simple_list_and_alias(self):
        a=self.parser.parse('万科城小区、中海日辉台小区、平安里家园（壹锦园）')
        self.assertEqual(a['op'],'or')
        self.assertTrue(all(c['op']=='named' for c in a['children']))
        r=P.NM.match_named_lists([dict(building_name='壹锦园')],[dict(school_name='甲',named_list_text=a['children'][2]['text'])])
        self.assertEqual(r['named'][0,0],1)

    def test_phase_not_erased(self):
        r=P.NM.match_named_lists([dict(building_name=n) for n in ['桂芳园一期','桂芳园七期','桂芳园']],
                                [dict(school_name='甲',named_list_text='桂芳园（1—6期）')])
        self.assertEqual(r['named'][:,0].tolist(),[1,-1,-1])

    def test_exclusion_stays_bound(self):
        a=self.parser.parse('贝尔路以南—坂雪岗大道以东—布龙路以北（不含中海日辉台及甲实验学校招生片区）')
        self.assertEqual(a['op'],'minus')
        self.assertEqual(a['children'][0]['op'],'road')
        # A housing name plus school range must not be mistaken for a bare name.
        self.assertNotEqual(a['children'][1]['op'],'named')

    def test_ring_is_not_direction_guess(self):
        a=self.parser.parse('吉祥路、龙平西路、龙城大道、龙福路围成区域。')
        self.assertEqual(a['op'],'ring');self.assertEqual(len(a['names']),4)

    def test_inclusion_does_not_destroy_tail_name(self):
        a=self.parser.parse('平大路、清平高速、平龙西路围合区域（包括融悦山居、金穗花园、南油花园等）')
        self.assertEqual(a['op'],'or');self.assertEqual(a['children'][0]['op'],'ring')
        self.assertEqual(a['children'][1]['children'][-1]['text'],'南油花园')

    def test_qualification_not_discarded(self):
        self.assertEqual(self.parser.parse('阳光花园，[国都花园、长龙花园]的非深户生')['op'],'conditional')
        a=self.parser.parse('龙岗社区（不含杨梅岗、龙园大观）、平南社区')
        self.assertEqual(a['op'],'or');self.assertEqual(a['children'][0]['op'],'minus')

    def test_admin_restriction_not_named(self):
        a=self.parser.parse('吉华街道水官高速以南、布龙路以北、广深铁路以西区域')
        self.assertEqual(a['op'],'and');self.assertEqual(a['children'][0]['op'],'admin')
        self.assertEqual(a['children'][1]['op'],'road')

    def test_school_reference(self):
        a=self.parser.parse('甲实验学校和乙小学的招生范围')
        self.assertEqual(a['op'],'or');self.assertEqual([c['school'] for c in a['children']],[0,1])

    def test_address_parity(self):
        a=self.parser.parse('新河路1号—407号（单号）')
        # Must reach the shared address grammar before generic parentheses.
        self.assertEqual(a['op'],'address');self.assertEqual(a['address'],['新河路',1,407,'odd'])

    def test_shared_sentence_preserves_base(self):
        a=self.parser.parse('龙城大道、回龙路、长兴北路、盐龙大道围成区域。该招生范围与兰著学校共享招生地段。')
        self.assertEqual(a['op'],'ring')

    def test_partial_admin_sides_are_conjunction(self):
        a=self.parser.parse('南至龙平西路、东至万科翰邻城、西和北至龙城街道行政区域界线。')
        self.assertEqual(a['op'],'and')
        self.assertEqual(a['children'][0]['op'],'road')

    def test_cardinal_scope_with_terminal_included_houses(self):
        a=self.parser.parse('北至甲路，西至乙路，东至丙大道，南至丁社区南端（含甲苑、乙苑）')
        self.assertEqual(a['op'],'or')
        boundary=a['children'][0]
        self.assertEqual(boundary['op'],'and')
        self.assertEqual(len(boundary['children']),4)
        self.assertEqual(a['children'][1]['op'],'or')
        self.assertTrue(all(c['op']=='named' for c in a['children'][1]['children']))

    def test_cardinal_scope_with_terminal_exclusion(self):
        a=self.parser.parse('北至甲路，西至乙路，东至丙大道，南至丁社区南端（不含甲苑）')
        self.assertEqual(a['op'],'minus')
        self.assertEqual(a['children'][0]['op'],'and')
        self.assertEqual(len(a['children'][0]['children']),4)

    def test_multiple_parentheses_do_not_drop_community(self):
        a=self.parser.parse('甲社区（龙观大道以东部分）、乙社区（龙观大道以西部分）')
        self.assertEqual(a['op'],'or')
        self.assertTrue(all(c['op']=='and' for c in a['children']))
        self.assertTrue(all(c['children'][1]['op']=='road' for c in a['children']))

    def test_mixed_include_exclude_not_bare_parent(self):
        a=self.parser.parse('慢城（包括一、二、三期，不含来座山）')
        self.assertEqual(a['op'],'unknown')

    def test_native_exclusion_overrides_geometric_inside(self):
        sites=[dict(building_name=n,coordinate_group_building_names=[n],coordinate_group_id=str(i),representative_point_wgs84=[i,0]) for i,n in enumerate(['甲花园','乙花园','丙花园'])]
        schools=[dict(school_id='s',school_name='试验小学')]
        inventory=[dict(school_name='试验小学',base_scope_text='甲路以东、乙路以北（不含乙花园）',status='individual_scope_requires_site_applicability',evidence=[])]
        ev=P.Evaluator(sites,schools,inventory,dict(district='龙岗区',relations=[]),{'甲路':[1],'乙路':[1]},parse_scope,
                       lambda name,di,alias=None:np.array([1,1,0]) if name=='甲路' else np.ones(3,dtype=np.int8),
                       lambda *args,**kw:np.full(3,-1,np.int8),lambda xy:np.asarray(xy),None)
        r=ev.build()
        self.assertEqual(r['known'],{0:[0]})
        self.assertEqual(r['negative'][:,0].tolist(),[False,True,True])

    def test_school_reference_cycle_is_unknown(self):
        sites=[dict(building_name='甲花园',coordinate_group_building_names=['甲花园'],coordinate_group_id='q',representative_point_wgs84=[0,0])]
        schools=[dict(school_id='a',school_name='甲小学'),dict(school_id='b',school_name='乙小学')]
        inventory=[dict(school_name=n,base_scope_text=t,status='individual_scope_requires_site_applicability',evidence=[]) for n,t in [('甲小学','乙小学的招生范围'),('乙小学','甲小学的招生范围')]]
        ev=P.Evaluator(sites,schools,inventory,dict(district='龙岗区',relations=[]),{},parse_scope,
                       lambda *a:np.array([-1]),lambda *a,**k:np.array([-1]),lambda xy:np.asarray(xy),None)
        np.testing.assert_array_equal(ev.evaluate(ev.asts[0],0),[-1])

    def test_closed_ring_uses_boundary_band(self):
        sites=[dict(building_name=str(i),coordinate_group_building_names=[str(i)],coordinate_group_id=str(i),representative_point_wgs84=xy) for i,xy in enumerate([[150,150],[450,150],[10,150]])]
        names=['甲路','乙路','丙路','丁路'];segments=[[[0,0],[300,0]],[[300,0],[300,300]],[[300,300],[0,300]],[[0,300],[0,0]]]
        by={n:[dict(properties=dict(osm_id=i),geometry=dict(type='LineString',coordinates=xy))] for i,(n,xy) in enumerate(zip(names,segments))}
        ev=P.Evaluator(sites,[],[],dict(district='龙岗区',relations=[]),by,parse_scope,None,None,lambda xy:np.asarray(xy),None)
        np.testing.assert_array_equal(ev.ring(names),[1,0,-1])

    def test_numbered_colon_list_preserves_first_name(self):
        a=self.parser.parse('甲路口往西—乙路片区：①甲花园、乙花园；②丙花园、丁花园')
        terms=[v['text'] for v in P.walk(a) if v['op']=='named']
        self.assertEqual(terms,['甲花园','乙花园','丙花园','丁花园'])
        a=self.parser.parse('甲社区：含甲花园、乙花园')
        self.assertEqual([v['text'] for v in P.walk(a) if v['op']=='named'],['甲花园','乙花园'])
        self.assertFalse(any(v['op']=='membership' for v in P.walk(a)))

    def test_conjoined_sides_and_turning_route(self):
        a=self.parser.parse('龙岗大道以南与同心路以东')
        self.assertEqual(a['op'],'road')
        self.assertEqual(a['pairs'],[['龙岗大道','南'],['同心路','东']])
        a=self.parser.parse('东西干道—布龙路—铁路桥往南沿铁路往南—中兴桥往东—中兴路以北区域')
        self.assertEqual(a['op'],'unknown')

    def test_mixed_parenthetical_exclusion_list(self):
        a=self.parser.parse('甲社区[甲路以北区域（乙花园、丙花园）除外；含：丁花园、戊花园]')
        self.assertEqual(a['op'],'or')
        self.assertEqual(a['children'][0]['op'],'minus')
        self.assertEqual([v['text'] for v in P.walk(a['children'][1]) if v['op']=='named'],['丁花园','戊花园'])

    def test_qualified_ring_not_mined_as_road_names(self):
        a=self.parser.parse('甲社区（由铁路、甲路、乙路、丙路围合的乙花园及工业区等片区除外）')
        self.assertEqual(a['op'],'minus')
        self.assertEqual(a['children'][1]['op'],'unknown')
        a=self.parser.parse('甲社区（由铁路、甲路、乙路、丙路围合的乙花园及工业区等片区除外）、九号院馆、坤宜福苑')
        self.assertEqual(a['op'],'or')
        self.assertEqual([v['text'] for v in a['children'][1:]],['九号院馆','坤宜福苑'])

if __name__=='__main__':unittest.main()
