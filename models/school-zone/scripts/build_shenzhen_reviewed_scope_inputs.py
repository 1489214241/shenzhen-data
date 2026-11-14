"""Round04 input snapshot: locally qualified clauses and reviewed street scopes.

The four source reviews activate geography, not household eligibility. Frozen
inputs and live policy packages are never edited. New positive evidence is a
lower bound: literal unconditional names or an independent administrative face.
"""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import openpyxl
import shenzhen_full_inputs as F
import build_shenzhen_aligned_inputs as B
import shenzhen_aligned_policy as AP
from shenzhen_qualified_scope import QualifiedParser
from zoning import shenzhen_policy as SP

OUT=F.REPO/'data/深圳市/局部条件修复输入_20261112'
OLD=F.REPO/'data/深圳市/修复实验输入_20261112'
REVIEW=F.REPO/'outputs/shenzhen_resolution_20261112/policy_review_sources'
BROAD={'龙华区鹭湖实验学校（小学部）':13,'龙华区观湖外国语学校（小学部）':14}
LOCAL={'水径小学','阳光小学'}


def exact_unconditional_names(ast):
    if ast['op']=='named':return {SP.normalize_name(ast['text'])}
    if ast['op']=='or':return set().union(*(exact_unconditional_names(c) for c in ast['children']))
    # In particular, never descend a conditional, exclusion or conjunction.
    return set()


def literal_group_hit(site,terms):
    names={SP.normalize_name(v) for v in site['coordinate_group_building_names'] if v}
    return bool(names) and names <= terms


def reviewed_policy(policy,local_review,workbook_rows):
    p=deepcopy(policy);changes=[]
    for row in p['school_scope_inventory']:
        name=row['school_name'];before=deepcopy(row)
        if name in LOCAL:
            rec=next(r for r in local_review['changes'] if r['school_id']==row['school_id'])
            assert rec['source_text']==row['base_scope_text']
            assert row['status']=='population_condition_pending'
            row['status']='mixed_qualification_partial_scope'
            row['review_basis']='Explicit top-level bracket binds household condition locally; all household conditions remain unknown.'
        elif name in BROAD:
            record=workbook_rows[BROAD[name]]
            assert row['base_scope_text']==record['values'][2]=='面向观湖街道招生。'
            assert record['values'][3]=='1-6类' and row['status']=='broad_scope_pending'
            row['status']='reviewed_street_geographic_scope'
            row['review_basis']='2026 official range table and 2024 catchment plan call this the school scope; only geographical membership is estimated.'
        if row!=before:
            changes.append(dict(school_id=row['school_id'],school_name=name,before=before,after=deepcopy(row)))
    # Keep eligibility/pending relations visible. This is not a new automatic
    # whole-school expansion edge, and never introduces university expansion.
    assert p['relations']==policy['relations']
    return p,changes


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve snapshot')
    for rel,h in F.read(OLD/'manifest.json')['files'].items():assert F.sha(OLD/rel)==h,rel
    book=next((F.SZ.DATA/'原始资料/年度政策').glob('*76160b725e.xlsx'))
    wb=openpyxl.load_workbook(book,read_only=True,data_only=True)
    sheet=wb['Sheet1'];rows={i:dict(row=i,values=list(next(sheet.iter_rows(min_row=i,max_row=i,values_only=True)))) for i in [2,3,13,14,108]}
    wb.close()
    local_review=F.read(REVIEW/'qualified_scope_review.json')
    F.write(OUT/'source_review.json',dict(workbook=str(book),workbook_sha256=F.sha(book),sheet='Sheet1',rows=rows,
        current_table_url='https://www.szlhq.gov.cn/attachment/1/1720/1720520/12826919.xlsx',
        official_html_url='https://www.szlhq.gov.cn/ywjyzszt/zszxdh/content/post_12833739.html',
        prior_plan_url='https://www.szlhq.gov.cn/attachment/1/1450/1450146/11317906.pdf',prior_plan_page=1,
        prior_plan_quote='（一）小学部学区 学区范围：面向全观湖街道招生。',
        prior_plan_verification='Official text cross-check; previous PDF screenshot/download failed; no new visual claim.',
        local_qualified_review=local_review,target_gold_read=False,enrollment_eligibility=False))
    policies={};all_changes={}
    for d in F.SZ.DISTRICTS:
        policies[d],all_changes[d]=reviewed_policy(SP.load_policy(d),local_review,rows)
        F.write(OUT/d/'reviewed_policy.json',policies[d])
    assert sum(map(len,all_changes.values()))==4
    F.write(OUT/'policy_changes.json',all_changes)
    captured={}
    original_parser,original_eval,original_load,original_out=AP.Parser,AP.Evaluator,SP.load_policy,B.OUT

    class CaptureEvaluator(original_eval):
        def build(self):
            result=super().build();facts=[];scope_audit=[]
            for j,row in enumerate(self.inventory):
                name=row['school_name']
                if name not in BROAD and name not in LOCAL:continue
                ast=self.asts[j]
                if name in BROAD:
                    assert ast==dict(op='admin',text='观湖街道')
                    vals=self.admin(ast['text']);hit=vals==1;kind='independent_admin_geography'
                    sources=[r for r in self.admin_audit if r['policy_name']==ast['text']]
                    assert len(sources)==1 and sources[0]['status']=='independent_osm_boundary'
                else:
                    terms=exact_unconditional_names(ast)
                    hit=np.array([literal_group_hit(s,terms) for s in self.sites]);kind='literal_unconditional_name'
                    sources=sorted(terms)
                scope_audit.append(dict(school_id=row['school_id'],school_name=name,ast=ast,evidence_kind=kind,
                    evidence_detail=sources,positive_sites=int(hit.sum()),unresolved_household_conditions=True if name in LOCAL else False,
                    source_text=row['base_scope_text'],sources=row['evidence']))
                for q in np.flatnonzero(hit):
                    facts.append(dict(site_id=self.sites[q]['coordinate_group_id'],school_id=row['school_id'],school_name=name,
                        building_names=self.sites[q]['coordinate_group_building_names'],evidence_kind=kind,
                        source_text=row['base_scope_text'],sources=row['evidence'],enrollment_eligibility=False))
            captured[self.policy['district']]=dict(facts=facts,scopes=scope_audit,complete_school_set=False,
                office_proxy_used_for_new_facts=False,school_polygon_used=False,target_gold_read=False)
            return result

    try:
        AP.Parser=QualifiedParser;AP.Evaluator=CaptureEvaluator
        SP.load_policy=lambda d:deepcopy(policies[d]);B.OUT=OUT
        B.main()
    finally:
        AP.Parser,AP.Evaluator,SP.load_policy,B.OUT=original_parser,original_eval,original_load,original_out
    audit={}
    for d in F.SZ.DISTRICTS:
        F.write(OUT/d/'reviewed_positive_facts.json',captured[d])
        with (OLD/d/'district.pkl').open('rb') as f:old=pickle.load(f)
        with (OUT/d/'district.pkl').open('rb') as f:new=pickle.load(f)
        assert old.site_ids==new.site_ids and old.school_ids==new.school_ids
        np.testing.assert_array_equal(old.coords,new.coords);np.testing.assert_array_equal(old.school_xy,new.school_xy)
        assert new.answers=={} and new.train_answers=={}
        a=np.load(OLD/d/'model_arrays.npz');b=np.load(OUT/d/'model_arrays.npz')
        before=F.read(OLD/d/'model_metadata.json');after=F.read(OUT/d/'model_metadata.json')
        oldobs=F.read(OLD/d/'policy_observations.json');newobs=F.read(OUT/d/'policy_observations.json')
        changed=[dict(school_id=x['school_id'],school_name=x['school_name'],before=x,after=y) for x,y in zip(oldobs,newobs) if x!=y]
        expected={r['school_id'] for r in all_changes[d]}
        assert {r['school_id'] for r in changed}==expected
        F.write(OUT/d/'school_observation_changes.json',changed)
        oldpairs={(int(q),s) for q,ss in before['known'].items() for s in ss}
        newpairs={(int(q),s) for q,ss in after['known'].items() for s in ss}
        audit[d]=dict(array_changed_cells={k:int(np.count_nonzero(a[k]!=b[k])) for k in a.files},
            changed_schools=len(changed),old_known_positions=len(before['known']),new_known_positions=len(after['known']),
            rule_pairs_added=len(newpairs-oldpairs),rule_pairs_removed=len(oldpairs-newpairs),
            independent_new_evidence_pairs=len(captured[d]['facts']),
            caution='Rebuilt native known sets are not complete and may displace previous positives. Round04 uses only new independent facts, preserving old predicted sets.',
            identities_and_coordinates_unchanged=True,source_labels_empty=True)
    F.write(OUT/'与修复实验输入差异.json',audit)
    shutil.copyfile(Path(__file__),OUT/'code'/Path(__file__).name)
    shutil.copyfile(F.REPO/'scripts/shenzhen_qualified_scope.py',OUT/'code/shenzhen_qualified_scope.py')
    manifest=F.read(OUT/'manifest.json')
    manifest.update(schema='shenzhen_reviewed_partial_scope_v1',based_on=str(OLD),all_school_inventory=202,
        files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    F.write(OUT/'manifest.json',manifest)
    print(audit,flush=True)


if __name__=='__main__':main()
