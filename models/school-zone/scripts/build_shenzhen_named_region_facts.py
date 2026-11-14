"""Infer site-scope facts from explicit named members of conjunctive regions."""
from pathlib import Path
from collections import defaultdict
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_named_region_scope import NamedRegionParser,positive_witnesses
from shenzhen_aligned_policy import NM
from shenzhen_scope_implication import canonical,implies
from shenzhen_partial_scope import partial,sat_proof

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'scope_graph_refresh_review/graphs'
INPUT=F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
OUT=ROOT/'named_region_facts_v1'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source proof')
    paths=[Path(__file__),F.REPO/'scripts/shenzhen_named_region_scope.py',
        F.REPO/'scripts/shenzhen_partial_scope.py',F.REPO/'scripts/shenzhen_scope_implication.py',
        F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py',F.REPO/'scripts/shenzhen_aligned_policy.py']
    sources={str(p):F.sha(p) for p in paths}
    for p,h in F.read(SOURCE/'manifest.json')['sources'].items():
        assert F.sha(p)==h,p
        sources[p]=h
    # Additional reviewed original-PDF pages, alongside page9 in parent manifest.
    for rel in ['partial_scope_review/longgang_page-10.png','partial_scope_review/longgang_page-11.png',
                'mixed_range_review/longgang_page-12.png','mixed_range_review/longgang_page-13.png','mixed_range_review/longgang_page-14.png']:
        sources[str(ROOT/rel)]=F.sha(ROOT/rel)
    summary={}
    for d in F.SZ.DISTRICTS:
        ip=SOURCE/f'{d}_changes.json';gp=SOURCE/f'{d}.json';mp=INPUT/d/'model_metadata.json'
        for p in [ip,gp,mp]:sources[str(p)]=F.sha(p)
        inv=F.read(ip)['school_inventory'];parser=NamedRegionParser(inv,parse_scope)
        asts=[parser.parse(r['source_text']) for r in inv]
        restricted={r['school_id'] for r in F.read(gp)['inventory'] if r['skip_reason']=='university_eligibility_or_pending_member'}
        active={r['school_id']:partial(a,r['school_id']) for r,a in zip(inv,asts)
                if r['school_id'] not in restricted and r['scope_status']=='individual_scope_requires_site_applicability'}
        sites=F.read(mp)['sites'];rows=[];to_site=[]
        for q,s in enumerate(sites):
            for name in sorted(set(s['coordinate_group_building_names'])):
                rows.append(dict(building_id=s['coordinate_group_id'],building_name=name,source_district=d));to_site.append(q)
        witnesses=[];facts=defaultdict(list);byid={r['school_id']:r for r in inv}
        for source,a in zip(inv,asts):
            sid=source['school_id']
            if sid not in active:continue
            for w in positive_witnesses(a):
                region=canonical(w['region_ast'])
                assert implies(region,active[sid]),'Witness region must lie in its source school scope'
                match=NM.match_named_lists(rows,[dict(school_name='source region',named_list_text='、'.join(w['included_terms']))],district=d)
                hits=defaultdict(list);ambiguous=set(match['review_rows'])
                for i,q in enumerate(to_site):
                    if i not in ambiguous and not match['phase_limits'][i,0] and match['named'][i,0]==1:
                        hits[q].append(rows[i]['building_name'])
                targets=[]
                for dest,expr in active.items():
                    if implies(region,expr):targets.append(dict(school_id=dest,proof=sat_proof(region,expr)))
                receipt=dict(w,source_school_id=sid,source_school_name=source['school_name'],source_evidence=source['evidence'],
                    expression=region,matched_sites=[dict(site_id=sites[q]['coordinate_group_id'],names=names) for q,names in hits.items()],
                    name_match_events=match['events'],review_rows=sorted(ambiguous),targets=targets)
                ix=len(witnesses);witnesses.append(receipt)
                for q in hits:
                    for target in targets:facts[(q,target['school_id'])].append(ix)
        derived=[]
        for (q,sid),evidence in sorted(facts.items()):
            derived.append(dict(district=d,site_id=sites[q]['coordinate_group_id'],building_names=sites[q]['coordinate_group_building_names'],
                school_id=sid,school_name=byid[sid]['school_name'],witness_indices=evidence,
                school_scope=byid[sid]['source_text'],school_evidence=byid[sid]['evidence'],
                source='official_explicit_included_names_and_direct_region_implication',
                prediction_trigger_used=False,complete_school_set=False,enrollment_eligibility=False))
        changes=[dict(school_id=r['school_id'],school_name=r['school_name'],before=r['ast'],after=a)
                 for r,a in zip(inv,asts) if r['ast']!=a]
        F.write(OUT/d/'facts.json',dict(facts=derived,witnesses=witnesses,school_asts=[dict(school_id=r['school_id'],ast=a) for r,a in zip(inv,asts)],ast_changes=changes))
        summary[d]=dict(schools=len(inv),witness_clauses=len(witnesses),matched_sites=len({r['site_id'] for r in derived}),school_relations=len(derived),changed_asts=len(changes))
        print(d,summary[d],flush=True)
        for r in derived:
            if any(witnesses[i]['source_school_id']!=r['school_id'] for i in r['witness_indices']):
                print('CROSS_SCOPE',r['building_names'],r['school_name'],flush=True)
    F.write(OUT/'summary.json',dict(districts=summary,target_gold_read=False,target_polygon_read=False,
        model_predictions_read=False,road_aliases_substituted=False,unknown_negative_labels=False,
        official_pdf_pages_reviewed=[9,10,11,12,13,14],independent_validation=False,adopted=False))
    F.write(OUT/'manifest.json',dict(sources=sources,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*.json')}))


if __name__=='__main__':main()
