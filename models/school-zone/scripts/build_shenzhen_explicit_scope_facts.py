"""Full202-school source audit of local exclusions and restored named positives."""
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import shenzhen_full_inputs as F
from shenzhen_aligned_policy import NM,walk,SOURCE_FILES
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_named_region_scope import NamedRegionParser
from shenzhen_scoped_parenthesis import ScopedParenthesisParser
from shenzhen_explicit_scope_facts import evaluate,prove

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'scope_graph_refresh_review/graphs'
INPUT=F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
OUT=ROOT/'explicit_scope_facts_v1'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source audit')
    paths=[Path(__file__),F.REPO/'scripts/shenzhen_explicit_scope_facts.py',F.REPO/'scripts/shenzhen_scoped_parenthesis.py',
           F.REPO/'scripts/shenzhen_named_region_scope.py',ROOT/'partial_scope_review/longgang_page-10.png']+SOURCE_FILES
    sources={str(p):F.sha(p) for p in paths}
    for p,h in F.read(SOURCE/'manifest.json')['sources'].items():assert F.sha(p)==h,p;sources[p]=h
    summary={}
    for d in F.SZ.DISTRICTS:
        gp=SOURCE/f'{d}.json';ip=SOURCE/f'{d}_changes.json';mp=INPUT/d/'model_metadata.json'
        lp=F.REPO/'data/深圳市/边界合取修复模型输入_20261112'/d/'labels.npz'
        npth=ROOT/'named_region_facts_v1'/d/'facts.json'
        for p in [gp,ip,mp,lp,npth]:sources[str(p)]=F.sha(p)
        inv=F.read(ip)['school_inventory'];sites=F.read(mp)['sites'];meta_schools=F.read(mp)['schools']
        assert [s['school_id'] for s in meta_schools]==[s['school_id'] for s in inv]
        pold=NamedRegionParser(inv,parse_scope);pnew=ScopedParenthesisParser(inv,parse_scope)
        old=[pold.parse(s['source_text']) for s in inv];new=[pnew.parse(s['source_text']) for s in inv]
        blocked={r['school_id'] for r in F.read(gp)['inventory'] if r['skip_reason']=='university_eligibility_or_pending_member'}
        eligible=[r['school_id'] not in blocked and r['scope_status']=='individual_scope_requires_site_applicability' for r in inv]
        rows=[];to_site=[]
        for q,s in enumerate(sites):
            for name in sorted(set(s['coordinate_group_building_names'])):
                rows.append(dict(building_id=s['coordinate_group_id'],building_name=name,source_district=d));to_site.append(q)
        terms=sorted({n['text'] for a in old+new for n in walk(a) if n['op']=='named'})
        values={};receipts={}
        for term in terms:
            m=NM.match_named_lists(rows,[dict(school_name='explicit source term',named_list_text=term)],district=d)
            hit=defaultdict(list);ambiguous={to_site[i] for i in m['review_rows']}
            for i,q in enumerate(to_site):
                if m['phase_limits'][i,0]:ambiguous.add(q)
                elif m['named'][i,0]==1:hit[q].append(rows[i]['building_name'])
            # A coordinate group with a conflicting/ambiguous alias abstains.
            hit={q:n for q,n in hit.items() if q not in ambiguous}
            v=np.full(len(sites),-1,np.int8)
            for q in hit:v[q]=1
            values[term]=v
            receipts[term]=dict(hits=[dict(site_id=sites[q]['coordinate_group_id'],matched_names=n) for q,n in hit.items()],
                                ambiguous_site_ids=sorted(sites[q]['coordinate_group_id'] for q in ambiguous),events=m['events'])
        original_positive=set()
        labels=dict(np.load(lp))['labels']
        for q,j in np.argwhere(labels==1):original_positive.add((sites[q]['coordinate_group_id'],inv[j]['school_id']))
        for f in F.read(npth)['facts']:original_positive.add((f['site_id'],f['school_id']))
        facts=[];negative=[];pending=[];inventory=[];changes=[]
        for j,(row,a,b) in enumerate(zip(inv,old,new)):
            sid=row['school_id'];before=evaluate(a,values,len(sites));after=evaluate(b,values,len(sites))
            has_exclusion=bool(re.search('除外|不含|不包括|不包含|以外|剔除',row['source_text']))
            if a!=b:changes.append(dict(school_id=sid,school_name=row['school_name'],source_text=row['source_text'],before=a,after=b,evidence=row['evidence']))
            hit_exclusion=set()
            for n in walk(b):
                if n['op']=='minus':hit_exclusion.update(np.flatnonzero(evaluate(n['children'][1],values,len(sites))==1).tolist())
            inventory.append(dict(**row,after_ast=b,eligible=eligible[j],has_exclusion=has_exclusion,
                                  name_proven_inside=int((after==1).sum()),name_proven_outside=int((after==0).sum()),
                                  matched_local_exclusion_sites=len(hit_exclusion)))
            for q in sorted(hit_exclusion):
                if after[q]!=0:pending.append(dict(site_id=sites[q]['coordinate_group_id'],school_id=sid,
                    building_names=sites[q]['coordinate_group_building_names'],school_name=row['school_name'],
                    reason='local_exclusion_does_not_exclude_all_school_branches',whole_scope_state=int(after[q])))
            if not eligible[j]:continue
            wanted={q:0 for q in np.flatnonzero(after==0)}
            if a!=b:wanted.update({q:1 for q in np.flatnonzero((after==1)&(before!=1))})
            for q,w in sorted(wanted.items()):
                true_terms={n['text'] for n in walk(b) if n['op']=='named' and values[n['text']][q]==1}
                proof=prove(b,true_terms,w)
                f=dict(district=d,site_id=sites[q]['coordinate_group_id'],building_names=sites[q]['coordinate_group_building_names'],
                    school_id=sid,school_name=row['school_name'],source_text=row['source_text'],evidence=row['evidence'],
                    proof=proof,complete_school_set=False,prediction_trigger_used=False,
                    conflicts_with_prior_positive=(sites[q]['coordinate_group_id'],sid) in original_positive if w==0 else False)
                (facts if w==1 else negative).append(f)
        F.write(OUT/d/'facts.json',dict(positive_repairs=facts,negative_facts=negative,pending_local_exclusions=pending,
                                      parser_changes=changes,school_inventory=inventory,name_receipts=receipts))
        summary[d]=dict(schools=len(inv),exclusion_schools=sum(r['has_exclusion'] for r in inventory),
            parser_changed_schools=len(changes),positive_repairs=len(facts),negative_relations=len(negative),
            negative_positive_conflicts=sum(f['conflicts_with_prior_positive'] for f in negative),pending_local_exclusions=len(pending))
        print('AUDIT',d,summary[d],flush=True)
        for f in facts:print('POSITIVE_REPAIR',f['building_names'],f['school_name'],flush=True)
        for f in negative:print('NEGATIVE',f['building_names'],f['school_name'],f['conflicts_with_prior_positive'],flush=True)
    F.write(OUT/'summary.json',dict(districts=summary,target_gold_read=False,target_polygon_read=False,
        model_predictions_read=False,unmatched_names_are_unknown=True,negative_labels_not_integrated=True,
        geometry_used=False,official_page10_visually_reviewed=True,adopted=False))
    F.write(OUT/'manifest.json',dict(sources=sources,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*.json')}))


if __name__=='__main__':main()
