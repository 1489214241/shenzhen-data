"""Independent geometry feasibility for the complete community boundary queue."""
from pathlib import Path
import re
from collections import defaultdict
from shapely.geometry import LineString,mapping
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shenzhen_full_inputs as F
from audit_shenzhen_road_topology import corridor_candidates

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
SOURCE=ROOT/'community_boundary_text_inventory'
OUT=ROOT/'community_boundary_geometry_audit'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve geometry feasibility audit')
    manifest=F.read(SOURCE/'manifest.json')
    for rel,h in manifest['files'].items():assert F.sha(SOURCE/rel)==h,rel
    for p,h in manifest['sources'].items():assert F.sha(p)==h,p
    queue=F.read(SOURCE/'linear_review_queue.json');cases={}
    for r in queue:
        key=(r['district'],r['street'],r['community'],tuple(r['sides'][z][0] for z in '东南西北'))
        if key not in cases:cases[key]=dict(district=r['district'],street=r['street'],community=r['community'],
            names={z:r['sides'][z][0] for z in '东南西北'},source_osm_ids=r['exact_geometry'],sources=[])
        cases[key]['sources'].append(dict(source=r['source'],quote=r['quote']))
    roads={f['properties']['osm_id']:f for f in F.read(F.OUT/'road_geometry.json')['features']}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    rows=[]
    for index,case in enumerate(cases.values()):
        row=dict(case_id=index,**case,missing_names=[case['names'][z] for z,v in case['source_osm_ids'].items() if not v])
        if row['missing_names']:rows.append(row);continue
        lines=[unary_union([LineString(F.project(roads[j]['geometry']['coordinates'])) for j in case['source_osm_ids'][z]]) for z in '东南西北']
        areas=re.findall(r'(?:面积(?:约|为|共)?|总面积(?:约|为)?)([0-9.]+)平方公里',case['sources'][0]['quote'])
        row['declared_area_km2']=[float(a) for a in areas];row['radii']={};shapes={}
        for radius in [0,10,25,50]:
            faces,supported=corridor_candidates(lines,radius);shapes[radius]=supported
            row['radii'][str(radius)]=dict(total_faces=len(faces),supported_faces=len(supported),
                candidates=[dict(area_km2=p.area/1e6,bounds=list(p.bounds),geometry=mapping(p)) for p in supported])
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        for ax in axes:
            for i,(z,g) in enumerate(zip('东南西北',lines)):
                segments=[g] if g.geom_type=='LineString' else list(g.geoms)
                for j,seg in enumerate(segments):
                    xx,yy=seg.xy;ax.plot(xx,yy,color=plt.get_cmap('tab10').colors[i],lw=1,label=z+'界 '+case['names'][z] if j==0 else None)
            for p in shapes[25]:
                xx,yy=p.exterior.xy;ax.fill(xx,yy,color='purple',alpha=.2)
            ax.set_aspect('equal');ax.legend(fontsize=8);ax.ticklabel_format(useOffset=False,style='plain')
        # Zoom from boundary geometry only, never school sites or target answers.
        if shapes[25]:
            b=unary_union(shapes[25]).bounds;pad=max(b[2]-b[0],b[3]-b[1])*.1+100
            axes[1].set_xlim(b[0]-pad,b[2]+pad);axes[1].set_ylim(b[1]-pad,b[3]+pad)
        axes[0].set_title('完整独立道路；全部满足四边支持的候选')
        axes[1].set_title('25米廊道候选；尚未认定社区边界')
        fig.suptitle(case['district']+'/'+case['street']+'/'+case['community']);fig.tight_layout()
        dest=OUT/'plots'/f'{index:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=140);plt.close(fig)
        rows.append(row)
        print(index,case['community'],{k:[v['supported_faces'],[round(p['area_km2'],3) for p in v['candidates']]] for k,v in row['radii'].items()},'official_area',areas,flush=True)
    F.write(OUT/'report.json',dict(cases=rows,all_candidate_boundary_sets=len(rows),exact_name_complete=sum(not r['missing_names'] for r in rows),
        target_gold_read=False,target_polygon_read=False,target_sites_used=False,school_relations_generated=0,
        caution='Closed shape is not proof of official community geometry; original directional semantics, dates, changes, and approximate area still require review.'))
    sources={str(SOURCE/'manifest.json'):F.sha(SOURCE/'manifest.json'),str(Path(__file__)):F.sha(__file__),
             str(F.REPO/'scripts/audit_shenzhen_road_topology.py'):F.sha(F.REPO/'scripts/audit_shenzhen_road_topology.py')}
    sources.update(manifest['sources'])
    F.write(OUT/'manifest.json',dict(sources=sources,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
