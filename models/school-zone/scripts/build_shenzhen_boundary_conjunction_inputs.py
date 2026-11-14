"""Isolated raw rebuild for reviewed incomplete-boundary conjunctions."""
from pathlib import Path
import shutil
import build_shenzhen_resident_group_inputs as B
from shenzhen_boundary_conjunction import BoundaryConjunctionParser


def main():
    B.OLD=B.F.REPO/'data/深圳市/居民小组约束输入_20261112'
    B.OUT=B.F.REPO/'data/深圳市/边界合取修复输入_20261112'
    B.ResidentGroupParser=BoundaryConjunctionParser
    B.main()
    for p in [Path(__file__),B.F.REPO/'scripts/shenzhen_boundary_conjunction.py',B.F.REPO/'tests/test_shenzhen_boundary_conjunction.py']:
        dest=B.OUT/'code'/p.relative_to(B.F.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    diff=B.F.read(B.OUT/'input_difference_audit.json')
    diff['limitation']='Only incomplete-boundary grouping and directed admin-border guard changed relative to resident+ring raw snapshot; no geometry, coordinates or policies added.'
    B.F.write(B.OUT/'input_difference_audit.json',diff)
    manifest=B.F.read(B.OUT/'manifest.json');manifest['schema']='shenzhen_incomplete_boundary_conjunction_v1'
    manifest['files']={str(p.relative_to(B.OUT)):B.F.sha(p) for p in B.OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    B.F.write(B.OUT/'manifest.json',manifest)


if __name__=='__main__':main()
