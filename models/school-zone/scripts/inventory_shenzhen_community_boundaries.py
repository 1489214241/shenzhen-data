"""Source-only census of explicit community four-side descriptions.

This creates a review queue, never community memberships or school labels.
Every profile remains in the audit, including missing and ambiguous boundaries.
"""
from pathlib import Path
from collections import defaultdict,Counter
import re
import shenzhen_full_inputs as F

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'community_membership_sources'
OUT=ROOT/'community_boundary_text_inventory'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve complete source census')
    profiles=F.read(SOURCE/'profile_inventory.json')['profiles']
    roads=F.read(F.OUT/'road_geometry.json')['features'];names=defaultdict(set)
    for f in roads:
        for k in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
            for name in str(f['properties'].get(k,'')).split(';'):
                if name:names[name].add(f['properties']['osm_id'])
    ledger=[];candidates=[];raw_hashes={}
    for index,p in enumerate(profiles):
        path=SOURCE/p['source']['raw_file'];assert F.sha(path)==p['source']['sha256'];raw_hashes[str(path)]=F.sha(path)
        record=dict(profile_index=index,district=p['district'],street=p['street'],community=p['community'],
            source=p['source'],boundary_paragraphs=[])
        for j,quote in enumerate(p['paragraphs']):
            matches=list(re.finditer(r'(?<![东南西北])([东南西北])(?:至|起|临|接|靠)([^，。；、\s]+)',quote))
            if len({m[1] for m in matches})<3:continue
            sides=defaultdict(list)
            for m in matches:sides[m[1]].append(m[2])
            # A queue candidate must preserve four unique directions and four
            # complete linear names. Mixed entities remain visibly unresolved.
            simple=all(len(sides[z])==1 for z in '东南西北') and len(matches)==4
            linear=simple and all(re.fullmatch(r'[\u4e00-\u9fff0-9]+(?:路|大道|河|高速|高速公路|铁路)',sides[z][0]) for z in '东南西北')
            row=dict(profile_index=index,paragraph_index=j,district=p['district'],street=p['street'],community=p['community'],
                quote=quote,source=p['source'],sides=dict(sides),simple_four_sides=bool(simple),linear_name_candidate=bool(linear),
                exact_geometry={z:sorted(names[sides[z][0]]) for z in '东南西北'} if linear else {},
                review_status='not_reviewed_not_membership_not_gold')
            record['boundary_paragraphs'].append(row)
            if linear:candidates.append(row)
        ledger.append(record)
    summary=dict(profiles=len(profiles),profiles_with_three_or_more_directional_mentions=sum(bool(r['boundary_paragraphs']) for r in ledger),
        linear_four_side_candidates=len(candidates),all_four_exact_named_geometry=sum(all(r['exact_geometry'].values()) for r in candidates),
        candidate_communities=len({(r['district'],r['street'],r['community']) for r in candidates}),
        target_labels_read=False,target_polygons_read=False,school_relations_generated=0,
        caution='Descriptions may be approximate, outdated, or include disconnected/ambiguous road segments. Exact name matching is a proposal, never geometry validation.')
    F.write(OUT/'all_profile_audit.json',ledger);F.write(OUT/'linear_review_queue.json',candidates);F.write(OUT/'summary.json',summary)
    rows=['官方社区四至文字：完整来源人口审计，尚未产生任何新归属。','']
    for r in candidates:
        rows += [f"{r['profile_index']} {r['district']}/{r['street']}/{r['community']}",
                 r['source']['url'],str(r['source']['page_dates']),r['quote'],
                 '各方向独立同名道路数量：'+str({z:len(v) for z,v in r['exact_geometry'].items()}),'']
    (OUT/'待核对原文.txt').write_text('\n'.join(rows),encoding='utf-8')
    sources={str(p):F.sha(p) for p in [Path(__file__),SOURCE/'profile_inventory.json',F.OUT/'road_geometry.json']}
    sources.update(raw_hashes)
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},sources=sources))
    print(summary,flush=True)
    for r in candidates:print(r['profile_index'],r['community'],r['sides'],{z:len(v) for z,v in r['exact_geometry'].items()},flush=True)


if __name__=='__main__':main()
