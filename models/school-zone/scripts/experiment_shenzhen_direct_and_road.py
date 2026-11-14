"""Frozen Modern: direct scope implications and an UNVERIFIED road hypothesis.

Separate geometry, prepare, infer, score, audit processes. Never read school
polygons. Road aliases are explicitly unresolved and cannot enter production.
"""
import argparse
import collections
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import unary_union
import experiment_shenzhen_full_prior as E
from audit_shenzhen_road_topology import corridor_candidates
from zoning.full_policy_prior import one_pass

F = E.M.F
REPO = E.REPO
INPUT = E.INPUT
OUT = REPO / 'outputs/shenzhen_road_policy_trial_20261114'
PARENT = REPO / 'outputs/modern_shenzhen_full_prior_repaired_20261114'
read, write, sha = E.read, E.write, E.sha
DISTRICTS = ['龙华区', '龙岗区']
ARMS = ['direct_model_roots', 'direct_model_and_text_roots',
        'road_alias_hypothesis', 'direct_plus_road_hypothesis']


def verify(name):
    for p, h in read(OUT / name).items():
        assert sha(p) == h, p


def geometry():
    E.M.guard()
    if (OUT / 'inputs_sha256.json').exists():
        raise FileExistsError('Preserve sealed geometry audit')
    roadpath = F.OUT / 'road_geometry.json'
    assert sha(roadpath) == read(F.OUT / 'manifest.json')['files']['road_geometry.json']
    byname = collections.defaultdict(list)
    for f in read(roadpath)['features']:
        byname[f['properties'].get('name', '')].append(f)
    # Only the northern corridor needed for this preselected shared-policy case.
    # Bridge remains a separate explicit hypothesis, not a fuzzy global alias.
    alias = ['龙城北路', '龙城北桥']
    byname['龙城大道'] = [f for n in alias for f in byname[n]]
    names = ['龙城大道', '回龙路', '长兴北路', '盐龙大道']
    lines = [unary_union([LineString(F.project(f['geometry']['coordinates']))
                         for f in byname[n]]) for n in names]
    candidates, radii = {}, {}
    for radius in [0, 10, 25, 50]:
        faces, supported = corridor_candidates(lines, radius)
        candidates[radius] = supported
        radii[str(radius)] = dict(faces=len(faces), supported=len(supported),
            areas_m2=[p.area for p in supported])
    policy = read(INPUT / '龙岗区/reviewed_policy.json')
    rule = next(r for r in policy['relations'] if r.get('rule_id') == 'LG03'
                or (r['kind'] == 'shared' and any('悦澜山' in str(x) for x in r.values())))
    # Fixed before looking at any new prediction or scoring result.
    selected = candidates[50]
    accepted = len(selected) == 1
    output = dict(status='unverified_alias_and_road_corridor_hypothesis',
        alias=alias, official_road='龙城大道', names=names,
        radius_m=50, radius_reason='Pre-score geometric sensitivity audit; conservative interior outside 50m road corridors. This width is a hypothesis, not a measured cadastral boundary.',
        all_radii=radii, unique_candidate=accepted,
        geometry=mapping(selected[0]) if accepted else None, crs='EPSG:32650',
        source_osm_ids={n:[f['properties']['osm_id'] for f in byname[n]] for n in names},
        pairwise_distance_m=[dict(a=names[i], b=names[j], distance=lines[i].distance(lines[j]))
                             for i in range(len(names)) for j in range(i)],
        policy_rule=rule, target_gold_read=False, target_polygon_read=False,
        alias_verified=False, adopted=False)
    write(OUT / 'geometry.json', output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    for color_index, (n, line) in enumerate(zip(names, lines)):
        segs = [line] if line.geom_type == 'LineString' else line.geoms
        color = plt.get_cmap('tab10').colors[color_index]
        for i, seg in enumerate(segs):
            ax.plot(*seg.xy, color=color, lw=1.4, label=n if i == 0 else None)
    if accepted:
        p = selected[0]
        ax.fill(*p.exterior.xy, alpha=.22, color='purple', label='50米道路走廊留下的候选内部')
        x0, y0, x1, y1 = p.bounds
        ax.set_xlim(x0-250, x1+250); ax.set_ylim(y0-250, y1+250)
    ax.set_aspect('equal'); ax.legend()
    ax.set_title('悦澜山局部共享范围：独立道路试算\n龙城大道道路别名尚未核实；不含学区地图或答案')
    fig.tight_layout(); fig.savefig(OUT / 'road_candidate.png', dpi=160); plt.close(fig)
    print('GEOMETRY', radii, 'accepted_hypothesis', accepted, flush=True)


def prepare():
    E.M.guard()
    if (OUT / 'inputs_sha256.json').exists():
        raise FileExistsError('Preserve frozen inputs')
    # Verify old evidence and predictions without loading old scores.
    for manifest in ['inputs_sha256.json', 'prediction_freeze.json']:
        for p, h in read(PARENT / manifest).items():
            assert sha(p) == h, p
    old = read(PARENT / 'protocol.json')
    protocol = dict(targets=DISTRICTS, seeds=old['seeds'], arms=ARMS,
        primary='direct_model_roots vs baseline_current', training_steps=0,
        frozen_model='Modern routed_bce_legacy; identical predictions and source training',
        direct='Original decision.raw_school_ids trigger all direct Boolean full-scope implication edges and reviewed shared rules ONCE. Added schools never become roots.',
        direct_text='Secondary: independently verified constraints.positive may also be roots. Existing final predictions and newly added predictions are not roots.',
        road='Exploratory UNVERIFIED alias 龙城大道 = 龙城北路 plus 龙城北桥; unique interior outside 50m OSM road corridors; inside adds LG03 schools only. Outside and boundary remain unknown; no deletion.',
        road_limits='Not independently verified road naming or cadastral boundary. Geometry parameter chosen using label-free topology. Cannot claim deployable improvement even if scores improve.',
        university='No university member-list expansion. Graph already excludes university/eligibility/pending school scopes.',
        exclusions='Never add explicitly excluded school; record blocked additions; preserve baseline.',
        population='1802 output groups,1761 scored,41 output-only,242 semantic-pending retained',
        scoring='Full exact-set equality; same frozen Gold, precision/recall and 1km spatial bootstrap3000',
        prior_results_seen=True, independent_validation=False, adopted=False,
        target_gold_read=False, target_polygon_read=False)
    write(OUT / 'protocol.json', protocol)
    paths = [Path(__file__).resolve(), OUT/'protocol.json', OUT/'geometry.json',
             REPO/'zoning/full_policy_prior.py', REPO/'scripts/audit_shenzhen_road_topology.py',
             F.OUT/'road_geometry.json', PARENT/'prediction_freeze.json', PARENT/'inputs_sha256.json']
    paths += list((OUT/'sources').glob('*.json'))
    for d in DISTRICTS:
        paths += [INPUT/d/f for f in ['metadata.json','constraints.json','reviewed_policy.json']]
        paths.append(PARENT/'inventory'/f'{d}.json')
        for label in ['ensemble']+list(map(str,old['seeds'])):
            for arm in ['baseline_current','full_scope_minimum']:
                paths.append(PARENT/'predictions'/arm/label/f'{d}.json')
    write(OUT/'inputs_sha256.json', {str(p):sha(p) for p in paths})
    print('INPUTS AND PROTOCOL SEALED', flush=True)


def infer():
    E.M.guard(); verify('inputs_sha256.json')
    if (OUT/'prediction_freeze.json').exists():
        raise FileExistsError('Preserve predictions')
    cfg=read(OUT/'protocol.json'); geo=read(OUT/'geometry.json')
    polygon=shape(geo['geometry']) if geo['unique_candidate'] else None
    coverage={}; evidence=[]
    for d in DISTRICTS:
        meta=read(INPUT/d/'metadata.json'); cons=read(INPUT/d/'constraints.json')
        policy=read(INPUT/d/'reviewed_policy.json'); graph=read(PARENT/'inventory'/f'{d}.json')['graph']
        sites={s['coordinate_group_id']:s for s in meta['sites']}
        xy=F.project([s['representative_point_wgs84'] for s in meta['sites']])
        inside={s['coordinate_group_id'] for s,p in zip(meta['sites'],xy)
                if d=='龙岗区' and polygon is not None and polygon.contains(Point(p))}
        coverage[d]=dict(road_inside_sites=len(inside),direct_edges=len(graph['edges']),labels={})
        for label in ['ensemble']+list(map(str,cfg['seeds'])):
            base=read(PARENT/'predictions/baseline_current'/label/f'{d}.json')
            arms={a:[] for a in ARMS}; changes=collections.Counter()
            for row in base:
                sid=row['site_id']; old=set(row['after']); con=cons[sid]
                roots=set(row['decision']['raw_school_ids'])
                negative=set(con['negative']); positive=set(con['positive'])
                direct=one_pass(roots,policy,[graph])-roots
                withtext=one_pass(roots|positive,policy,[graph])
                road=set(geo['policy_rule']['school_ids']) if sid in inside else set()
                additions=dict(direct_model_roots=direct,
                    direct_model_and_text_roots=withtext,
                    road_alias_hypothesis=road,
                    direct_plus_road_hypothesis=withtext|road)
                for arm, add in additions.items():
                    after=old|(add-negative)
                    decision=dict(original_model_roots=sorted(roots), independent_text_roots=sorted(positive),
                        added=sorted(after-old), excluded_additions=sorted(add&negative),
                        road_inside=sid in inside, road_alias_verified=False,
                        transitive_expansion=False, complete_school_set=False)
                    arms[arm].append(dict(row,after=sorted(after),trial_decision=decision))
                    changes[arm]+=int(after!=old)
                if label=='ensemble':
                    evidence.append(dict(district=d,site_id=sid,building_name=sites[sid]['building_name'],
                        original_model_roots=sorted(roots),text_roots=sorted(positive),road_inside=sid in inside,
                        source_edges=[e for e in graph['edges'] if e['trigger_school_id'] in roots|positive],
                        directly_triggered_shared=[r for r in policy['relations'] if r['kind']=='shared'
                            and set(r.get('trigger_school_ids',[]))&(roots|positive)]))
            for arm,rows in arms.items():write(OUT/'predictions'/arm/label/f'{d}.json',rows)
            write(OUT/'predictions/baseline_current'/label/f'{d}.json',base)
            write(OUT/'predictions/previous_scope'/label/f'{d}.json',read(PARENT/'predictions/full_scope_minimum'/label/f'{d}.json'))
            coverage[d]['labels'][label]=dict(changes)
    write(OUT/'coverage.json',coverage); write(OUT/'input_evidence.json',evidence)
    files=list((OUT/'predictions').rglob('*.json'))+[OUT/'coverage.json',OUT/'input_evidence.json']
    write(OUT/'prediction_freeze.json',{str(p):sha(p) for p in files})
    print('PREDICTIONS SEALED',coverage,flush=True)


def score():
    verify('inputs_sha256.json');verify('prediction_freeze.json')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve scores')
    cfg=read(OUT/'protocol.json')
    truthpath=REPO/'outputs/modernnca_partial_sets_20261113/scored_rows/categorical_policy.json'
    truth={(r['d'],r['site_id']):r for r in read(truthpath)}
    hashes=read(REPO/'outputs/modernnca_partial_sets_20261113/results.json')['gold_sha256']
    for d,h in hashes.items():assert sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    filters={'all':lambda r:True,'single':lambda r:r['size']==1,'multi':lambda r:r['size']>1,
        'three_plus':lambda r:r['size']>=3,'semantic_pending':lambda r:r['semantic_pending'],
        'not_pending':lambda r:not r['semantic_pending'],
        **{d:lambda r,d=d:r['d']==d for d in DISTRICTS}}
    metrics={}; seeds={}; rows_by={}
    for arm in ['baseline_current','previous_scope']+ARMS:
        seeds[arm]={}
        for label in ['ensemble']+list(map(str,cfg['seeds'])):
            rows=[]; outside=0
            for d in DISTRICTS:
                preds=read(OUT/'predictions'/arm/label/f'{d}.json')
                for p in preds:
                    key=d,p['site_id']
                    if key not in truth:outside+=1;continue
                    g=truth[key]; a=set(p['after']);t=set(g['truth'])
                    rows.append(dict(g,prediction=sorted(a),correct=a==t,tp=len(a&t),fp=len(a-t),fn=len(t-a)))
            assert len(rows)==1761 and outside==41 and sum(r['semantic_pending'] for r in rows)==242
            mm={k:E.M.metric([r for r in rows if f(r)]) for k,f in filters.items()}
            seeds[arm][label]=mm
            if label=='ensemble':
                metrics[arm]=mm;rows_by[arm]=rows;write(OUT/'scored_rows'/f'{arm}.json',rows)
    comparisons={}; changes={}
    for comparator in ['baseline_current','previous_scope']:
        before={(r['d'],r['site_id']):r for r in rows_by[comparator]}
        comparisons[comparator]={}
        for arm in ARMS:
            comparisons[comparator][arm]={}
            for k,f in filters.items():
                rows=[r for r in rows_by[arm] if f(r)]
                delta=[int(r['correct'])-int(before[r['d'],r['site_id']]['correct']) for r in rows]
                comparisons[comparator][arm][k]=dict(corrected=delta.count(1),harmed=delta.count(-1),
                    delta_pp=100*float(np.mean(delta)),ci95=E.M.evaluate.bootstrap(rows,delta,20261104,3000))
            if comparator=='baseline_current':
                changes[arm]=[dict(r,before=before[r['d'],r['site_id']]['prediction'],
                    before_correct=before[r['d'],r['site_id']]['correct']) for r in rows_by[arm]
                    if set(r['prediction'])!=set(before[r['d'],r['site_id']]['prediction'])]
    write(OUT/'changes_diagnostic_only.json',changes)
    write(OUT/'results.json',dict(metrics=metrics,seeds=seeds,comparisons=comparisons,
        gold_sha256=hashes,scoring_rows_sha256=sha(truthpath),primary=cfg['primary'],
        adopted=False,independent_validation=False,road_alias_verified=False))
    print('RESULTS', {a:m['all'] for a,m in metrics.items()},flush=True)


def audit():
    verify('inputs_sha256.json');verify('prediction_freeze.json')
    cfg=read(OUT/'protocol.json');r=read(OUT/'results.json');n=0
    for d in DISTRICTS:
        cons=read(INPUT/d/'constraints.json'); meta=read(INPUT/d/'metadata.json')
        for label in ['ensemble']+list(map(str,cfg['seeds'])):
            base=read(OUT/'predictions/baseline_current'/label/f'{d}.json')
            assert base==read(PARENT/'predictions/baseline_current'/label/f'{d}.json')
            for arm in ARMS:
                rows=read(OUT/'predictions'/arm/label/f'{d}.json')
                assert len(rows)==len(base)==len(meta['sites'])
                for a,b in zip(base,rows):
                    assert a['site_id']==b['site_id']
                    assert set(a['after'])<=set(b['after'])
                    assert not set(b['after'])&set(cons[b['site_id']]['negative'])
                    assert not b['trial_decision']['transitive_expansion']
                    n+=1
    before={(x['d'],x['site_id']):x for x in read(OUT/'scored_rows/baseline_current.json')}
    for arm in ARMS:
        rows=read(OUT/'scored_rows'/f'{arm}.json');assert len(rows)==1761
        assert sum(set(x['prediction'])==set(x['truth']) for x in rows)==r['metrics'][arm]['all']['correct']
        delta=[int(x['correct'])-int(before[x['d'],x['site_id']]['correct']) for x in rows]
        c=r['comparisons']['baseline_current'][arm]['all']
        assert c['corrected']==delta.count(1) and c['harmed']==delta.count(-1)
    for d,h in r['gold_sha256'].items():assert sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    write(OUT/'output_audit.json',dict(passed=True,prediction_rows_checked=n,
        baseline_identical=True,gold_unchanged=True,exact_set_recount=True,
        road_alias_verified=False,adopted=False))
    print('AUDIT PASSED',n,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['geometry','prepare','infer','score','audit'])
    globals()[ap.parse_args().phase]()
