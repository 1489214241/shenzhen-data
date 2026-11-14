"""Post-score consistency audit. Reads target map labels; NEVER training input.

Compare source-proved complete-scope implications against the original map
sets, without changing either. Priority clauses and unresolved source semantics
remain visible. A contradiction does not, by itself, identify which source errs.
"""
from pathlib import Path
import collections
import numpy as np
from shapely.geometry import Polygon,shape
from shapely.ops import transform
from pyproj import Transformer
import pdfplumber
import shenzhen_full_inputs as F
from zoning import shenzhen_policy as SP

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OUT=ROOT/'scope_map_consistency_audit'
BASE=F.REPO/'outputs/shenzhen_aligned_model_20261111'


def main():
    if OUT.exists():raise FileExistsError('Preserve diagnostic snapshot')
    base=F.read(BASE/'results.json')
    for d,h in base['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold=F.read(BASE/'scored_cases.json');reports={};cases=[]
    for d in ['龙华区','龙岗区']:
        graph_path=ROOT/'round02_scope_inclusion_v2/source_graphs'/f'{d}.json'
        graph=F.read(graph_path);policy=SP.load_policy(d)
        by={r['school_id']:r for r in policy['school_scope_inventory']}
        rows=[r for r in gold if r['district']==d and r['has_map_gold']];edges=[];affected=set()
        for e in graph['edges']:
            a,b=e['trigger_school_id'],e['implied_school_id']
            bad=[r for r in rows if a in r['gold'] and b not in r['gold']]
            affected.update(r['site_id'] for r in bad)
            entry=dict(trigger=by[a],implied=by[b],violating_sites=len(bad),
                original_text_literal_equal=by[a]['original_scope_text']==by[b]['original_scope_text'],
                processed_base_literal_equal=by[a]['base_scope_text']==by[b]['base_scope_text'],
                extra_clauses_present=bool(by[a]['extra_clauses'] or by[b]['extra_clauses']),proof=e['proof'])
            edges.append(entry)
            for r in bad:cases.append(dict(district=d,site_id=r['site_id'],building_name=r['building_name'],gold=r['gold'],
                trigger_school_id=a,implied_school_id=b,semantic_pending=r['semantic_pending'],
                diagnostic_only=True,target_for_training=False))
        reports[d]=dict(schools_reviewed=len(by),edges_checked=len(edges),scored_sites=len(rows),
            edges_with_violations=sum(e['violating_sites']>0 for e in edges),unique_violating_sites=len(affected),edges=edges,
            graph_sha256=F.sha(graph_path),policy_sha256=F.sha(F.SZ.DATA/d/'政策规则/规则包.json'))
    # Compare two independently named full-street scopes and their native map
    # polygons in one metric CRS. This data is quarantined from preparation.
    adminpath=F.REPO/'data/深圳市/规则对齐独立资料_20261111/admin_geometry.json'
    admin=next(f for f in F.read(adminpath)['features'] if f['properties'].get('name')=='观湖街道')
    tr=Transformer.from_crs(4326,32650,always_xy=True).transform
    area=transform(tr,shape(admin['geometry']));geometry=[]
    for d,terms in [('龙华区',['鹭湖实验学校','观湖外国语学校']),('龙岗区',['贤义外国语学校','信义实验小学'])]:
        p=F.SZ.DATA/d/'评价数据/官方学区范围.json';zones=F.read(p);selected=[]
        for term in terms:
            matches=[z for z in zones if term in z['school_name']];assert len(matches)==1
            z=matches[0]
            assert z['crs'] in ['BD-09_native_BMap_contract','BD-09_expected_BMap_contract']
            g=Polygon(F.project([F.CO.bd_to_wgs(pt) for pt in z['vertices']])) if z['geometry_valid'] else None
            if g is not None:assert g.is_valid
            hits={r['site_id'] for r in gold if r['district']==d and z['school_id'] in r['gold']}
            selected.append((z,g,hits))
        a,ga,ha=selected[0];b,gb,hb=selected[1]
        comparison=dict(district=d,school_a=a['school_name'],school_b=b['school_name'],
            source_url=a['map_source_url'],snapshot_a=a['snapshot_time'],snapshot_b=b['snapshot_time'],
            vertices_a=len(a['vertices']),vertices_b=len(b['vertices']),
            polygon_iou=ga.intersection(gb).area/ga.union(gb).area if ga is not None and gb is not None else None,
            geometry_valid_a=a['geometry_valid'],geometry_valid_b=b['geometry_valid'],
            geometry_issue_a=a['geometry_issue'],geometry_issue_b=b['geometry_issue'],
            invalid_ring_handling='No repair, area or IoU for invalid rings. Original official-algorithm point labels retained.',
            map_hit_a=len(ha),map_hit_b=len(hb),map_hit_both=len(ha&hb),only_a=len(ha-hb),only_b=len(hb-ha),
            native_polygon_sha256=F.sha(p),coordinate_conversion='BD09->iterativeWGS84->EPSG32650',
            native_point_labels_preserved=True)
        if d=='龙华区':
            comparison['independent_street_geometry']=dict(properties=admin['properties'],sha256=F.sha(adminpath),
                area_km2=area.area/1e6,source_is_official=False)
            comparison['school_vs_street']=[dict(school=z['school_name'],school_area_km2=g.area/1e6,
                intersection_over_street=g.intersection(area).area/area.area,intersection_over_school=g.intersection(area).area/g.area,
                iou=g.intersection(area).area/g.union(area).area) for z,g,h in selected]
        geometry.append(comparison)
    pdf=next((F.SZ.DATA/'原始资料/年度政策').glob('*dd40f1d9af.pdf'))
    pages=[]
    with pdfplumber.open(pdf) as reader:
        for n in [3,9]:pages.append(dict(page=n,text=reader.pages[n-1].extract_text()))
    source=dict(path=str(pdf),sha256=F.sha(pdf),url='https://www.lg.gov.cn/attachment/1/1722/1722313/12839768.pdf',pages=pages,
        corroborating_official_url='https://www.lg.gov.cn/attachment/1/1722/1722313/12840154.pdf',
        explanation='The official priority clause explicitly calls Xianyi/Xinyi a shared catchment. Priority adds2.5points within that shared catchment; do not relabel it a university zone to fit map outputs.')
    F.write(OUT/'report.json',dict(districts=reports,geometry=geometry,priority_source=source,
        target_gold_read=True,target_gold_changed=False,training_or_prediction_input=False,
        limitation='Violations of source-parsed implications against map-hit labels. Requires source semantics/geometry audit; not proof that every map label is wrong. No Gold rewriting or revised score.'))
    F.write(OUT/'cases_not_training.json',cases)
    lines=['官方范围关系与原地图答案一致性审计（仅评分后诊断，禁止训练使用）','',
        '未改原图答案、样本分母或政策。核对已有76条直接文字范围包含关系，涉及全部202校的准入清单。',
        '不一致表示文字解析/政策关系与地图标签不能同时成立，不能仅凭此认定地图错误。','']
    for d,r in reports.items():
        lines.append(f"{d}: {r['edges_checked']}条关系，{r['edges_with_violations']}条存在不一致，影响{r['unique_violating_sites']}个不同小区。")
        for e in sorted(r['edges'],key=lambda e:-e['violating_sites']):
            if e['violating_sites']:lines.append(f"  {e['trigger']['school_name']} -> {e['implied']['school_name']}: {e['violating_sites']}例；原文附加条款={e['extra_clauses_present']}")
    lines += ['',source['url'],'官方指南第3页：贤义外国语学校--信义实验小学明确属于共享学区；优享只关系第一志愿加分。',
        '这不能按大学区单校规则处理。需继续核对地图是否展示优享范围、范围是否过时或采集字段含义不一致。','']
    for g in geometry:lines.append(str(g))
    (OUT/'文字与地图不一致审计.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print({d:{k:v for k,v in r.items() if k!='edges'} for d,r in reports.items()},flush=True)
    print(geometry,flush=True)


if __name__=='__main__':main()
