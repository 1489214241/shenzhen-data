"""Isolate the resident-group fix after the combined rebuild exposed rule conflicts."""
from pathlib import Path
import shutil
import build_shenzhen_resident_group_inputs as B
from shenzhen_resident_group_scope_isolated import QualifiedResidentGroupParser


def main():
    B.OUT=B.F.REPO/'data/深圳市/居民小组单项修复输入_20261112'
    B.ResidentGroupParser=QualifiedResidentGroupParser
    B.main()
    for p in [Path(__file__),B.F.REPO/'scripts/shenzhen_resident_group_scope_isolated.py']:
        dest=B.OUT/'code'/p.relative_to(B.F.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    diff=B.F.read(B.OUT/'input_difference_audit.json')
    diff['limitation']='Only resident-group qualification changed; no scoped-ring repair. Source estimates are not complete school sets.'
    B.F.write(B.OUT/'input_difference_audit.json',diff)
    manifest=B.F.read(B.OUT/'manifest.json');manifest['schema']='shenzhen_isolated_resident_group_scope_v1'
    manifest['files']={str(p.relative_to(B.OUT)):B.F.sha(p) for p in B.OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    B.F.write(B.OUT/'manifest.json',manifest)


if __name__=='__main__':main()
