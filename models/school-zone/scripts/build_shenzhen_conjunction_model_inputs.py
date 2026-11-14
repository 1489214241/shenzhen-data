"""Carry only the reviewed conjunction-state delta into current model inputs."""
from pathlib import Path
import shutil
import build_shenzhen_resident_feature_inputs as B


def main():
    B.BASE=B.F.REPO/'data/深圳市/居民小组修复模型输入_20261112'
    B.BEFORE=B.F.REPO/'data/深圳市/居民小组约束输入_20261112'
    B.AFTER=B.F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
    B.OUT=B.F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
    B.main()
    audit=B.F.read(B.OUT/'input_audit.json');audit['scope']='Only three-school incomplete-boundary conjunction state delta; resident fix inherited; no new geometry, references or labels.'
    B.F.write(B.OUT/'input_audit.json',audit)
    for d in B.F.SZ.DISTRICTS:
        meta=B.F.read(B.OUT/d/'model_metadata.json');meta['incomplete_boundary_conjunction_repair']=True
        B.F.write(B.OUT/d/'model_metadata.json',meta)
    for p in [Path(__file__),B.F.REPO/'scripts/shenzhen_boundary_conjunction_v2.py']:
        dest=B.OUT/'code'/p.relative_to(B.F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    manifest=B.F.read(B.OUT/'manifest.json')
    manifest['files']={str(p.relative_to(B.OUT)):B.F.sha(p) for p in B.OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    B.F.write(B.OUT/'manifest.json',manifest)


if __name__=='__main__':main()
