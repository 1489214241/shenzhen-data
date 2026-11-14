"""Diagnose independent road-ring failures without target map labels.

Exact line polygonization is compared to holes left by small road corridors.
Corridor candidates are diagnostic only until geometric and source review.
No synthetic gap segment, school-centred crop or target-based face selection.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import LineString,Polygon,mapping
from shapely.ops import unary_union,polygonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shenzhen_full_inputs as F
from shenzhen_resident_group_scope import ResidentGroupParser
from build_shenzhen_aligned_inputs import parse_scope

OUT=F.REPO/'outputs/shenzhen_resolution_20261112/road_ring_topology_audit'
INPUT=F.REPO/'data/深圳市/居民小组约束输入_20261112'


def walk(ast):
    if ast['op']=='ring':yield ast
    for child in ast.get('children',[]):yield from walk(child)


def polygons(geometry):
    return [geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)


def corridor_candidates(lines,radius):
    if radius==0:
        faces=list(polygonize(unary_union(lines)))
        supported=[p for p in faces if p.is_valid and all(p.boundary.intersection(g).length>=50 for g in lines)]
    else:
        barrier=unary_union([g.buffer(radius) for g in lines])
        faces=[Polygon(ring) for p in polygons(barrier) for ring in p.interiors]
        supported=[p for p in faces if p.is_valid and all(p.boundary.intersection(g.buffer(radius+.01)).length>=50 for g in lines)]
    return faces,supported


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve topology audit')
    road_path=F.OUT/'road_geometry.json'
    assert F.sha(road_path)==F.read(F.OUT/'manifest.json')['files']['road_geometry.json']
    roads=F.read(road_path)['features'];byname=defaultdict(dict)
    for f in roads:
        for key in ['name','name:zh','name:zh-Hans','official_name','alt_name','old_name']:
            for name in str(f['properties'].get(key,'')).split(';'):
                if name:byname[name][str(f['properties']['osm_id'])]=f
    cases={}
    for district in F.SZ.DISTRICTS:
        policy=F.read(INPUT/district/'reviewed_policy.json');inventory=policy['school_scope_inventory']
        parser=ResidentGroupParser(inventory,parse_scope)
        for school in inventory:
            for ast in walk(parser.parse(school['base_scope_text'])):
                key=tuple(sorted(ast['names']))
                if key not in cases:cases[key]=dict(names=ast['names'],schools=[])
                cases[key]['schools'].append(dict(district=district,school_id=school['school_id'],
                    school_name=school['school_name'],source_text=school['base_scope_text'],evidence=school.get('evidence',[])))
    results=[];plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    for index,case in enumerate(cases.values()):
        names=case['names'];missing=[n for n in names if n not in byname]
        row=dict(case_id=index,**case,missing=missing)
        if missing:results.append(row);continue
        lines=[unary_union([LineString(F.project(f['geometry']['coordinates'])) for f in byname[n].values()]) for n in names]
        row['source_osm_ids']={n:sorted(byname[n]) for n in names}
        row['pairwise_min_distance_m']=[dict(a=names[i],b=names[j],distance=float(lines[i].distance(lines[j]))) for i in range(len(names)) for j in range(i)]
        row['radii']={};candidates={}
        for radius in [0,10,25,50]:
            faces,supported=corridor_candidates(lines,radius);candidates[radius]=supported
            row['radii'][str(radius)]=dict(faces=len(faces),supported=len(supported),
                candidates=[dict(area_m2=float(p.area),bounds=list(p.bounds),geometry=mapping(p)) for p in supported])
        results.append(row)
        fig,axes=plt.subplots(1,2,figsize=(15,7));colors=plt.get_cmap('tab10').colors
        for ax in axes:
            for j,(n,g) in enumerate(zip(names,lines)):
                segments=[g] if g.geom_type=='LineString' else list(g.geoms)
                for k,seg in enumerate(segments):
                    xx,yy=seg.xy;ax.plot(xx,yy,color=colors[j%10],lw=1,label=n if k==0 else None)
            for p in candidates[25]:
                xx,yy=p.exterior.xy;ax.fill(xx,yy,color='purple',alpha=.2)
            ax.set_aspect('equal');ax.legend(fontsize=8);ax.ticklabel_format(useOffset=False,style='plain')
        axes[0].set_title('完整独立道路线；紫色为25米廊道诊断候选')
        if candidates[25]:
            bounds=unary_union(candidates[25]).bounds;pad=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*.1+100
            axes[1].set_xlim(bounds[0]-pad,bounds[2]+pad);axes[1].set_ylim(bounds[1]-pad,bounds[3]+pad)
        axes[1].set_title('按全部候选范围显示；没有学校或答案选面')
        fig.suptitle(' / '.join(names)+'\n'+case['schools'][0]['school_name']);fig.tight_layout()
        dest=OUT/'plots'/f'{index:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest,dpi=140);plt.close(fig)
        print(index,names,{k:(v['faces'],v['supported']) for k,v in row['radii'].items()},flush=True)
    F.write(OUT/'report.json',dict(cases=results,target_gold_read=False,target_polygon_read=False,
        used_for_model=False,radii_m=[0,10,25,50],criterion='Every named boundary supports at least50m of face boundary; keep all qualifying faces.',
        warning='Corridor holes are independent geometric candidates, not verified school polygons or enrollment facts.'))
    links={str(road_path):F.sha(road_path)}
    for d in F.SZ.DISTRICTS:
        p=INPUT/d/'reviewed_policy.json';links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_scoped_ring.py',F.REPO/'scripts/shenzhen_resident_group_scope.py']:
        links[str(p)]=F.sha(p)
    F.write(OUT/'manifest.json',dict(sources=links,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
