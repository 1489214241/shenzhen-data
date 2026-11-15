"""Verify published Gold against the unchanged source and captured point kernel."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'数据清单/光明坪山Gold_20261115'
def read(p): return json.loads(p.read_text(encoding='utf8'))
for row in read(AUDIT/'发布校验清单.json')['files']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
result={}
for d,n in [('光明区',167),('坪山区',107)]:
    zones=read(ROOT/d/'polygons.json')
    groups=read(ROOT/d/'小区数据/坐标组.json')
    schools=read(ROOT/d/'schools.json')
    rows=read(ROOT/d/'Gold/小区_学校Gold.json')
    req=read(AUDIT/f'{d}_点落面请求.json')
    assert req['zones']==[z['vertices'] for z in zones]
    assert req['points']==[g['representative_point_bd09'] for g in groups]
    source=read(ROOT/'来源存档/深圳市/光明坪山准入核查_20261115/current_sources/guangming_map_browser.json')
    # The saved request includes the captured function, whose hash is recorded.
    assert hashlib.sha256(req['function'].encode()).hexdigest()==read(AUDIT/'Gold验收.json')['point_kernel_sha256']
    assert req['function']==source['point_predicate']
    raw=subprocess.check_output(['node',str(AUDIT/'点落面核验.cjs'),str(AUDIT/f'{d}_点落面请求.json')])
    hits=np.array(json.loads(raw),dtype=np.int8)
    with np.load(ROOT/d/'Gold/评价标签.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(hits,z['path_membership'])
        assert list(z['site_ids'])==[g['coordinate_group_id'] for g in groups]
        assert list(z['school_ids'])==[s['school_id'] for s in schools]
        assert list(z['zone_ids'])==[p['zone_id'] for p in zones]
        byid={r['coordinate_group_id']:r for r in rows}
        assert len(byid)==len(rows)==n
        counts=[]
        for i,g in enumerate(groups):
            expected={zones[j]['school_id'] for j in np.flatnonzero(hits[i])}
            actual={str(z['school_ids'][j]) for j in np.flatnonzero(z['membership'][i])}
            assert actual==expected==set(byid[g['coordinate_group_id']]['point_school_ids'])
            assert bool(z['evaluation_mask'][i])==bool(expected)
            assert byid[g['coordinate_group_id']]['representative_point_bd09']==g['representative_point_bd09']
            counts.append(len(expected))
    result[d]=dict(gold_groups=n,single_school=sum(k==1 for k in counts),multi_school=sum(k>1 for k in counts),point_path_pairs=int(hits.size))
print(json.dumps(dict(passed=True,districts=result,source_paths_unchanged=True),ensure_ascii=False,indent=2))
