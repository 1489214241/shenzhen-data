"""Freeze population audit and rebuild only the three fallback-OR defects."""
from pathlib import Path
import shutil
import audit_shenzhen_boundary_conjunction as A
import build_shenzhen_resident_group_inputs as B
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2


def main():
    A.OUT=A.F.REPO/'outputs/shenzhen_resolution_20261112/boundary_conjunction_audit_v2'
    A.BoundaryConjunctionParser=BoundaryConjunctionParserV2
    A.main()
    # The reused audit's parser hash refers to its original implementation;
    # record the actual staged wrapper and parser separately before freezing.
    manifest=A.F.read(A.OUT/'manifest.json')
    manifest['parser']=A.F.sha(A.F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py')
    manifest['wrapper']=A.F.sha(__file__);A.F.write(A.OUT/'manifest.json',manifest)
    B.OLD=B.F.REPO/'data/深圳市/居民小组约束输入_20261112'
    B.OUT=B.F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
    B.ResidentGroupParser=BoundaryConjunctionParserV2;B.main()
    for p in [Path(__file__),B.F.REPO/'scripts/shenzhen_boundary_conjunction.py',
              B.F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py',B.F.REPO/'tests/test_shenzhen_boundary_conjunction_v2.py']:
        dest=B.OUT/'code'/p.relative_to(B.F.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    diff=B.F.read(B.OUT/'input_difference_audit.json')
    diff['limitation']='Three fallback-OR grouping repairs only. Existing admin intersections, all coordinates and geometry preserved.'
    B.F.write(B.OUT/'input_difference_audit.json',diff)
    manifest=B.F.read(B.OUT/'manifest.json');manifest['schema']='shenzhen_incomplete_boundary_conjunction_v2'
    manifest['files']={str(p.relative_to(B.OUT)):B.F.sha(p) for p in B.OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    B.F.write(B.OUT/'manifest.json',manifest)


if __name__=='__main__':main()
