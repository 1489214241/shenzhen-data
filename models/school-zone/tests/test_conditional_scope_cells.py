import unittest
from zoning.conditional_scope_cells import build_cells,sufficient_branches


class CellTests(unittest.TestCase):
    def row(self,sid,ast):
        return dict(school_id=sid,school_name=sid,ast=ast,source_text='source',evidence=['source'],scope_status='individual_scope_requires_site_applicability')

    def test_common_subscope_compiles_joint_set_without_school_chain(self):
        x=dict(op='membership',text='新和社区');y=dict(op='membership',text='丹湖居委会');z=dict(op='membership',text='别的社区')
        rows=[self.row('A',dict(op='or',children=[x,y])),self.row('B',dict(op='or',children=[x,z]))]
        recs,_=build_cells(rows,dict(relations=[],year=2026,stage='小学'))
        match=[r for r in recs if r['condition']==('membership','新和社区')]
        self.assertEqual(match[0]['school_ids'],['A','B'])
        self.assertEqual(match[0]['root_school_ids'],[])
        self.assertFalse(match[0]['complete_school_set'])

    def test_exclusion_not_erased(self):
        x=dict(op='membership',text='社区');ex=dict(op='named',text='排除小区')
        expr=sufficient_branches(dict(op='minus',children=[x,ex]))
        self.assertEqual(expr[0][0],'minus')
        self.assertNotIn(('membership','社区'),expr)

    def test_unknown_and_qualifier_not_dropped(self):
        x=dict(op='named',text='花园');u=dict(op='conditional',children=[x])
        self.assertEqual(sufficient_branches(dict(op='and',children=[x,u])),[])
        self.assertEqual(sufficient_branches(dict(op='or',children=[x,u])),[('named','花园')])

    def test_university_member_not_added_by_common_scope(self):
        x=dict(op='membership',text='社区');rows=[self.row('A',x),self.row('B',x)]
        recs,_=build_cells(rows,dict(relations=[dict(kind='university',school_ids=['B'])],year=2026,stage='小学'))
        self.assertEqual(recs,[])


if __name__=='__main__':unittest.main()
