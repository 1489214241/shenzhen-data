"""Enumerate every school changed by incomplete-boundary grouping repair."""
import shenzhen_full_inputs as F
from shenzhen_resident_group_scope import ResidentGroupParser
from shenzhen_boundary_conjunction import BoundaryConjunctionParser
from shenzhen_aligned_policy import walk
from build_shenzhen_aligned_inputs import parse_scope

INPUT=F.REPO/'data/深圳市/居民小组约束输入_20261112'
OUT=F.REPO/'outputs/shenzhen_resolution_20261112/boundary_conjunction_audit'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve full population audit')
    for rel,h in F.read(INPUT/'manifest.json')['files'].items():assert F.sha(INPUT/rel)==h,rel
    changes=[];summary={}
    for d in F.SZ.DISTRICTS:
        inv=F.read(INPUT/d/'reviewed_policy.json')['school_scope_inventory']
        old=ResidentGroupParser(inv,parse_scope);new=BoundaryConjunctionParser(inv,parse_scope)
        local=[]
        for row in inv:
            a=old.parse(row['base_scope_text']);b=new.parse(row['base_scope_text'])
            if a!=b:
                r=dict(district=d,school_name=row['school_name'],school_id=row['school_id'],source=row['base_scope_text'],
                    evidence=row['evidence'],before=a,after=b,
                    old_named=sorted(v['text'] for v in walk(a) if v['op']=='named'),new_named=sorted(v['text'] for v in walk(b) if v['op']=='named'))
                local.append(r);changes.append(r)
        summary[d]=dict(schools=len(inv),changes=len(local))
        print(d,summary[d],flush=True)
        for r in local:print(r['school_name'],r['source'],r['before']['op'],'->',r['after']['op'],flush=True)
    F.write(OUT/'report.json',dict(summary=summary,changes=changes,target_gold_read=False,target_polygon_read=False,adopted=False))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        input_manifest=F.sha(INPUT/'manifest.json'),parser=F.sha(F.REPO/'scripts/shenzhen_boundary_conjunction.py'),code=F.sha(__file__)))


if __name__=='__main__':main()
