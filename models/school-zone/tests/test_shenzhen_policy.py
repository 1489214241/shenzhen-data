"""政策反例回归：局部共享、大学区、未决条件、稳定ID及输入/Gold隔离。"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zoning import shenzhen_policy as sp, shenzhen_policy_evaluate as ev, predict, rules, shenzhen_sources


def site(name='甲住宅'):
    return dict(coordinate_group_id='q', building_name=name, coordinate_group_building_names=[name])


def package(relations):
    return dict(version='test', year=2026, school_ids=['A','B','C','D'], relations=relations)


def rel(key, kind, ids, triggers=(), names=(), **kw):
    return dict(rule_id=key, kind=kind, school_ids=ids, trigger_school_ids=list(triggers), exact_site_names=list(names), reason=key, **kw)


def fact(rule, status='satisfied', **kw):
    return dict(site_id='q', rule_id=rule, status=status, year=2026, stage='小学',
                evidence_kind='manual_source_review', evidence='独立原文和住宅身份核查', **kw)


class SemanticTests(unittest.TestCase):
    def test_university_does_not_expand(self):
        p = package([rel('u','university',['A','B','C'])])
        self.assertEqual(sp.apply(p,site(),['B'])['predicted_school_ids'], ['B'])

    def test_university_does_not_arbitrarily_collapse(self):
        p = package([rel('u','university',['A','B'])])
        r = sp.apply(p,site(),['A','B'])
        self.assertEqual(r['predicted_school_ids'], ['A','B'])
        self.assertTrue(r['pending'])
        self.assertEqual(r['policy_supported_school_ids'], [])

    def test_verified_original_base_controls_university(self):
        p = package([rel('u','university',['A','B'])])
        self.assertEqual(sp.apply(p,site(),['A','B'],[fact('u',base_school_id='B')])['predicted_school_ids'], ['B'])

    def test_shared_not_destroyed_by_university(self):
        p = package([rel('u','university',['A','B']),rel('s','shared',['A','B'],names=['甲住宅'])])
        self.assertEqual(sp.apply(p,site(),['A'],[fact('u',base_school_id='A')])['predicted_school_ids'], ['A','B'])

    def test_local_shared_direction_and_no_transitive_closure(self):
        p = package([rel('ab','shared',['A','B'],['A']),rel('bc','shared',['B','C'],['B'])])
        self.assertEqual(sp.apply(p,site(),['A'])['predicted_school_ids'], ['A','B'])
        self.assertEqual(sp.apply(p,site(),['C'])['predicted_school_ids'], ['C'])

    def test_unknown_fact_does_not_trigger_model_expansion(self):
        p = package([rel('ab','shared',['A','B'],['A'])])
        self.assertEqual(sp.apply(p,site(),['A'],[fact('ab','unknown')])['predicted_school_ids'], ['A'])

    def test_phase_prefix_not_an_identity(self):
        p = package([rel('ab','shared',['A','B'],names=['甲住宅'])])
        self.assertEqual(sp.apply(p,site('甲住宅二期'))['predicted_school_ids'], [])
        self.assertEqual(sp.apply(p,site())['predicted_school_ids'], ['A','B'])

    def test_coordinate_collision_does_not_propagate_name(self):
        p = package([rel('ab','shared',['A','B'],names=['甲住宅'])])
        s = site(); s['coordinate_group_building_names'].append('乙商业')
        self.assertEqual(sp.apply(p,s)['predicted_school_ids'], [])

    def test_eligibility_is_not_base_membership(self):
        p = package([rel('lottery','eligibility',['C'],names=['甲住宅'])])
        self.assertEqual(sp.apply(p,site(),['A'])['predicted_school_ids'], ['A'])

    def test_missing_base_is_not_negative(self):
        p = package([rel('base','base_scope',['A'],complete_base_scope=True)])
        self.assertEqual(sp.apply(p,site(),['A'])['predicted_school_ids'], ['A'])
        self.assertEqual(sp.apply(p,site(),['A'],[fact('base','not_satisfied')])['predicted_school_ids'], [])

    def test_gold_evidence_rejected(self):
        p = package([rel('s','shared',['A','B'])]); f=fact('s');f['evidence_kind']='official_map_point_membership'
        with self.assertRaises(ValueError):sp.apply(p,site(),[],[f])

    def test_identity_and_year_rejected(self):
        p = package([rel('s','shared',['A','B'])])
        for field, value in [('site_id','other'),('year',2025),('stage','初中')]:
            f=fact('s');f[field]=value
            with self.assertRaises(ValueError):sp.apply(p,site(),[],[f])
        with self.assertRaises(ValueError):sp.apply(p,site(),['old_school'])

    def test_model_shared_output_not_training_fact(self):
        r=sp.apply(package([rel('s','shared',['A','B'],['A'])]),site(),['A'])
        self.assertEqual(r['predicted_school_ids'],['A','B'])
        self.assertEqual(r['policy_supported_school_ids'],[])

    def test_independent_base_fact_keeps_direct_shared_without_closure(self):
        p=package([rel('base','base_scope',['A']),rel('ab','shared',['A','B'],['A']),rel('bc','shared',['B','C'],['B'])])
        r=sp.apply(p,site(),['D'],[fact('base')])
        self.assertEqual(r['predicted_school_ids'],['A','B'])
        self.assertEqual(r['policy_supported_school_ids'],['A','B'])

    def test_pending_labels_preserved_and_full_denominator(self):
        p=package([rel('mixed','base_scope',['A'])]);p['school_scope_inventory']=[dict(school_id=s,status='individual') for s in p['school_ids']]
        g=[dict(coordinate_group_id='q',building_name='甲住宅',point_school_ids=['A','B'])]
        before=copy.deepcopy(g); audit=ev.audit_gold(p,g)
        r=sp.apply(p,site(),['A']); result=ev.score([r],g,audit)
        self.assertEqual(g,before);self.assertEqual(result['n'],1)
        self.assertEqual(result['semantic_pending'],1);self.assertIsNone(result['base_scope_accuracy'])
        with self.assertRaises(ValueError):ev.score([],g,audit)


class CurrentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lh=sp.load_policy('龙华区');cls.lg=sp.load_policy('龙岗区')

    def rule(self,p,key):
        return next(r for r in p['relations'] if r['rule_id']==key)

    def test_longgang_partial_shared_counterexamples(self):
        r=self.rule(self.lg,'LG03'); narrow=r['trigger_school_ids'][0]; broad=next(s for s in r['school_ids'] if s!=narrow)
        self.assertEqual(set(sp.apply(self.lg,site(),[narrow])['predicted_school_ids']),set(r['school_ids']))
        self.assertNotIn(narrow,sp.apply(self.lg,site(),[broad])['predicted_school_ids'])
        # 南湾学校 -> 康桥（全范围同文），但绝不跨南湾实验传播给石芽岭。
        nanwan='sz:longgang:school:496FFFE16CFC5558E053BC09A8C0A5ED'
        shiyaling='sz:longgang:school:496FFFE16CF65558E053BC09A8C0A5ED'
        out=sp.apply(self.lg,site(),[nanwan])['predicted_school_ids']
        self.assertNotIn(shiyaling,out);self.assertEqual(len(out),2)

    def test_real_named_three_school_rule(self):
        r=sp.apply(self.lg,site('和通花园'))
        self.assertEqual(set(r['policy_supported_school_ids']),set(self.rule(self.lg,'LG06')['school_ids']))

    def test_priority_does_not_split_shared_and_population_stays_pending(self):
        xianyi='sz:longgang:school:496FFFE16D2C5558E053BC09A8C0A5ED'
        xinyi='sz:longgang:school:496FFFE16D0A5558E053BC09A8C0A5ED'
        self.assertIn(xinyi,sp.apply(self.lg,site(),[xianyi])['predicted_school_ids'])
        longling='sz:longgang:school:496FFFE16CE15558E053BC09A8C0A5ED'
        out=sp.apply(self.lg,site('龙岭山庄'),[longling])
        self.assertEqual(out['predicted_school_ids'],[longling])
        self.assertTrue(out['pending']);self.assertFalse(out['policy_supported_school_ids'])

    def test_separate_campuses_and_university(self):
        guanhu=next(s for s in self.lh['school_ids'] if s.endswith(':guanhu'))
        gaoxin=next(s for s in self.lh['school_ids'] if s.endswith(':gaoxin'))
        self.assertIn(gaoxin,sp.apply(self.lh,site(),[guanhu])['predicted_school_ids'])
        u=self.rule(self.lh,'LH-UNIVERSITY')['school_ids']
        for s in u:self.assertEqual(set(sp.apply(self.lh,site(),[s])['predicted_school_ids']) & set(u),{s})

    def test_policy_execution_never_loads_gold(self):
        with patch.object(shenzhen_sources,'load_point_gold',side_effect=AssertionError('Gold leaked')):
            rows=shenzhen_sources.predict_with_policy('龙岗区')
        self.assertEqual(len(rows),1174)

    def test_main_prediction_hook_and_non_shenzhen_noop(self):
        import numpy as np
        current=shenzhen_sources._load('龙岗区','小区数据/坐标组.json')
        s=next(s for s in current if s['building_name']=='和通花园')
        schools=shenzhen_sources.load_schools('龙岗区')
        d=SimpleNamespace(city='深圳',name='龙岗区',n=1,school_n=len(schools),site_names=[s['building_name']],
                          site_ids=[s['coordinate_group_id']],school_ids=[x['school_id'] for x in schools],school_names=[x['school_name'] for x in schools])
        ro=rules.run(d,'unified_20261106')
        self.assertEqual(len(ro.known[0]),3)
        out=predict.answers(d,ro,np.zeros((1,len(schools))))
        self.assertEqual(len(out[0]['schools']),3)
        row=[{'x':1}]
        self.assertIs(sp.annotate_output(SimpleNamespace(city='杭州'),row),row)


if __name__=='__main__':unittest.main()
