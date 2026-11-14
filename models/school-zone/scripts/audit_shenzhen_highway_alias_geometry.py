"""Diagnostic geometry after two reviewed road-name identity proposals.

No target properties, school polygons or Gold. Full G15 is a superset of Jihe,
so its geometry is only a candidate for a locally bounded four-road ring.
"""
from pathlib import Path
from collections import defaultdict
from shapely.geometry import LineString,mapping
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shenzhen_full_inputs as F
from audit_shenzhen_road_topology import corridor_candidates

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'community_boundary_geometry_audit'
OUT=ROOT/'highway_alias_community_geometry'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve independent geometry audit')
    manifest=F.read(SOURCE/'manifest.json')
    for rel,h in manifest['files'].items():assert F.sha(SOURCE/rel)==h,rel
    for p,h in manifest['sources'].items():assert F.sha(p)==h,p
    evidence=[dict(road='机荷高速',route_code='G15',kind='subroute_not_global_equivalence',
        url='https://jtys.sz.gov.cn/zwgk/ggqsydw/gsgl/',
        quote='机荷高速（路线编码G15）总里程42.4公里，包括机荷高速东段（清湖至荷坳）和机荷高速西段（机场至清湖），为国家高速公路网沈阳—海口高速公路组成部分',
        retrieval='Official-domain search-index excerpt read; direct HTML retrieval failed TLS and web open timed out. Not a locally archived original.',
        allowed_use='Independent geometry feasibility only; require local route-segment verification before a membership estimate.'),
        dict(road='水官高速公路',candidate='水官高速',route_code='S28',kind='reviewed_highway_suffix_abbreviation',
        url='https://jtys.sz.gov.cn/zwgk/ggqsydw/gsgl/',
        quote='水官高速（路线编号S28）呈东西走向，里程16.003公里，路基宽50米，双向10车道',
        retrieval='Official-domain search-index excerpt; exact full-road identity still requires provenance review before prediction.',
        allowed_use='Independent geometry feasibility only. No arbitrary road suffix normalization.'),
        dict(road='龙观东路',candidate='龙观大道',kind='rejected_unproven_equivalence',
        url='https://jtys.sz.gov.cn/attachment/1/1518/1518354/11856071.pdf',page=7,
        quote='龙观大道 主干路 双向六车道 布龙路 龙观东路 4.4',
        retrieval='PDF text read through web tool, page7: endpoint naming does not establish equivalence.')]
    F.write(OUT/'identity_review.json',dict(evidence=evidence,generated_memberships=0,used_for_model=False))
    roads=F.read(F.OUT/'road_geometry.json')['features'];byid={f['properties']['osm_id']:f for f in roads}
    cases=F.read(SOURCE/'report.json')['cases'];selected=[c for c in cases if c['case_id'] in [1,9]]
    # Resolve by semantic identity as well, so inventory changes cannot silently
    # apply a proposal to a different community.
    assert {c['community'] for c in selected}=={'润城社区','怡锦社区'}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    reports=[]
    for case in selected:
        code='G15' if case['community']=='润城社区' else 'S28'
        side='南' if code=='G15' else '北';resolved={k:list(v) for k,v in case['source_osm_ids'].items()}
        resolved[side]=[f['properties']['osm_id'] for f in roads if code in str(f['properties'].get('ref','')).split(';') and f['properties'].get('highway')=='motorway']
        assert all(resolved.values())
        lines=[unary_union([LineString(F.project(byid[i]['geometry']['coordinates'])) for i in resolved[z]]) for z in '东南西北']
        record=dict(case_id=case['case_id'],community=case['community'],names=case['names'],sources=case['sources'],
            resolved_osm_ids=resolved,route_code=code,changed_side=side,radii={})
        geometries={}
        for radius in [0,10,25,50]:
            faces,supported=corridor_candidates(lines,radius);geometries[radius]=supported
            record['radii'][str(radius)]=dict(all_faces=len(faces),supported=len(supported),
                candidates=[dict(area_km2=p.area/1e6,bounds=list(p.bounds),geometry=mapping(p)) for p in supported])
        reports.append(record)
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        for ax in axes:
            for i,(z,g) in enumerate(zip('东南西北',lines)):
                segments=[g] if g.geom_type=='LineString' else list(g.geoms)
                for j,seg in enumerate(segments):
                    xx,yy=seg.xy;ax.plot(xx,yy,color=plt.get_cmap('tab10').colors[i],lw=1,label=z+'界 '+case['names'][z] if j==0 else None)
            for p in geometries[25]:
                xx,yy=p.exterior.xy;ax.fill(xx,yy,color='purple',alpha=.2)
            ax.set_aspect('equal');ax.legend(fontsize=8);ax.ticklabel_format(useOffset=False,style='plain')
        if geometries[25]:
            b=unary_union(geometries[25]).bounds;pad=max(b[2]-b[0],b[3]-b[1])*.1+100
            axes[1].set_xlim(b[0]-pad,b[2]+pad);axes[1].set_ylim(b[1]-pad,b[3]+pad)
        fig.suptitle(case['community']+'：独立道路名称候选核查，未作为归属输入');fig.tight_layout()
        dest=OUT/'plots'/f"{case['case_id']:02d}.png";dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=140);plt.close(fig)
        print(case['community'],{k:[v['supported'],[round(p['area_km2'],3) for p in v['candidates']]] for k,v in record['radii'].items()},flush=True)
    F.write(OUT/'report.json',dict(cases=reports,target_gold_read=False,target_polygon_read=False,target_sites_used=False,generated_memberships=0,adopted=False))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        sources={str(SOURCE/'manifest.json'):F.sha(SOURCE/'manifest.json'),str(Path(__file__)):F.sha(__file__),
                 str(F.OUT/'road_geometry.json'):F.sha(F.OUT/'road_geometry.json')}))


if __name__=='__main__':main()
