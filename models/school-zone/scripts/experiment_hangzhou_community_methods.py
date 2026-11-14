"""Frozen-weight community-estimator comparison on Hangzhou; never changes defaults.

python scripts/experiment_hangzhou_community_methods.py prepare|train|infer|score
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict, Counter
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import shapely
import torch
from pyproj import Transformer
from zoning import data, rules, evidence, train, evaluate, DEVICE
from zoning.model import LinearRetrieval, RetrievalScorer
from shenzhen_community_comparison import nearest_separated

OUT = REPO / 'outputs/hangzhou_community_methods_20261113'
ROOT = REPO.parent
HZ = data.CITIES['杭州']
ARMS = ['hangzhou_E', 'changsha_E', 'hangzhou_ER', 'changsha_ER', 'changsha_keep_current_R']
MCA = ROOT / '科研推进/社区坐标新用法_20261104/raw/mca_hz'
UNITS = ROOT / '科研推进/杭州社区村条款_20261107/units_v2.json'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        pickle.dump(value, f, protocol=5)


def load(p):
    with p.open('rb') as f:
        return pickle.load(f)


def input_only(d):
    """Construct District without opening answers or reading scoring scope."""
    f = data.DATA / d
    meta, exc = read(f/'district.json'), read(f/'exceptions.json')
    with np.load(f/'arrays.npz') as a, np.load(f/'extra_evidence.npz') as e:
        x = data.District(name=d, city=meta['city'], crs=meta['crs'],
            site_names=meta['site_names'], school_names=meta['school_names'],
            orig=np.array(meta['orig']) if meta['orig'] is not None else None,
            coords=a['coords'], groups=a['groups'].astype(str), school_xy=a['school_xy'],
            answers={}, eval_rows=[], train_answers={}, scope=None,
            not_reference=set(exc['not_reference']), not_training=set(exc['not_training']),
            community_name=e['community_name'].astype(str), community_strict=e['community_strict'],
            community_school=e['community_school'], name_match=e['name_match'], lines_key=meta.get('lines'))
    return data.with_mca_community(x)


def road_aware(coords, points, ls):
    """Same nearest-eight crossing gates as build_shenzhen_aligned_inputs.infer_communities.

    Points are already in the target city's metric CRS (no Shenzhen projection).
    """
    dist = np.linalg.norm(coords[:, None]-points[None], axis=-1)
    idx = np.full(len(coords), -1, int)
    strict = np.zeros(len(coords), bool); loose = strict.copy()
    gates = np.full(len(coords), 3, int); ds = np.full(len(coords), np.inf)
    for q in range(len(coords)):
        cand = [j for j in np.argsort(dist[q]) if dist[q, j] <= 1500][:8]
        if not cand:
            continue
        segments = shapely.linestrings(np.stack([np.repeat(coords[q:q+1], len(cand), axis=0), points[cand]], 1))
        qi, li = ls.tree.query(segments, predicate='crosses')
        crossed = [set() for _ in cand]
        for i, j in zip(qi, li):
            crossed[i].add(ls.cls[j])
        free = [i for i, c in enumerate(crossed) if not c]
        soft = [i for i, c in enumerate(crossed) if not c & {'major','secondary','rail','river'}]
        k = (free or soft or [0])[0]
        idx[q] = cand[k]; gates[q] = 0 if free else 1 if soft else 2
        ds[q] = dist[q, idx[q]]
        strict[q] = gates[q] == 0 and ds[q] <= 400
        loose[q] = gates[q] <= 1 and ds[q] <= 800 and not strict[q]
    return dict(point=idx, strict=strict, loose=loose, distance=ds, gate=gates)


def inventory(d, crs):
    assert crs == 32651
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    grouped = defaultdict(list)
    for r in read(MCA/f'{d}.json'):
        if r.get('place_type') not in ('社区','行政村'):
            continue
        g = r.get('gdm') or {}
        if g.get('type') not in ('point','multipoint'):
            continue
        c = g['coordinates'][0] if g['type'] == 'multipoint' else g['coordinates']
        grouped[r['standard_name']].append(tr.transform(*c))
    records = [(name, xy) for name, pts in grouped.items() for xy in pts]
    return np.array([r[1] for r in records]), np.array([r[0] for r in records])


def features(x, names, est, units):
    assert len(set(x.school_names)) == x.school_n
    schools = {s:i for i,s in enumerate(x.school_names)}
    whole, part = defaultdict(set), defaultdict(set)
    for r in units:
        if r['district'] == x.name:
            (whole if r['whole'] else part)[r['unit']].add(schools[r['school']])
    cn = np.full(x.n, '', dtype=object)
    cs = np.zeros((x.n,x.school_n,4), np.float32)
    for q in range(x.n):
        if not (est['strict'][q] or est['loose'][q]):
            continue
        name = names[est['point'][q]]; cn[q] = name
        w, p = whole.get(name,set()), part.get(name,set())
        if w:
            cs[q, sorted(w), 0 if est['strict'][q] else 1] = 1
            cs[q, [s for s in range(x.school_n) if s not in w], 3] = 1
        if p:
            cs[q, sorted(p), 2] = 1
    return replace(x, community_name=cn.astype(str), community_strict=est['strict'], community_school=cs)


def rule(x, cfg, community):
    r = cfg['rules']
    return rules.run(x,r['preset'],r['hangzhou_names'],r['hangzhou_boundary'],community)


def replace_community(x, dist):
    """For a fixed rule/reference bank only these six feature columns may change."""
    pair=x.pair.copy();anchor=x.anchor.copy();ri=x.ref_sites
    cn,strict=dist.community_name,dist.community_strict
    same=(cn[:,None]==cn[ri][None]) & (cn[:,None]!='')
    both=strict[:,None] & strict[ri][None]
    pair[...,x.pair_blocks['same_community']]=np.stack([same&both,(~same)&both],-1)
    anchor[...,x.anchor_blocks['community_clause']]=dist.community_school
    return replace(x,pair=pair,anchor=anchor)


def prepare():
    cfg = read(REPO/'configs/default.json')
    assert cfg['train']['seeds'] == list(range(20261131,20261141))
    assert cfg['rules']['hangzhou_community_rule']
    protocol = dict(config=cfg, arms=ARMS, primary='Full predicted-school set equals frozen map+list Gold; no scope intersection.',
        secondary='Historical scoped strict and any-hit are reported separately.',
        weights='Reconstruct current Hangzhou ER target-specific native models using original ten seeds; verify archived probabilities. Freeze weights for all arms.',
        E='Common policy-only reference bank, no community direct rules, estimator affects community features only.',
        ER='Same common policy bank plus each estimator own singleton whole-community direct rules; rebuild references and learned-boundary features.',
        keep_current_R='Diagnostic: replace only community features, retain current Hangzhou ER references and direct decisions.',
        supervision='Original eight-district training; target Gold is not loaded for target fit. Other districts official singleton answers are original legal source supervision.',
        thresholds='Changsha 8 points within 1500m; no-cross <=400 strict, no-hard-cross <=800 loose. Hangzhou nearest distinct-name ratio:1500m/2 strict,3000m/1.2 loose.',
        adoption='Exploratory local fixed-weight comparison; no default change, no independent holdout claim.',
        spatial_ci='Paired district-stratified 1km bootstrap3000,seed20261104.',
        crs='All Hangzhou sites, school points, MCA points and road lines use EPSG:32651.')
    write(OUT/'protocol.json',protocol)
    source_files = [REPO/'configs/default.json',UNITS,Path(__file__),REPO/'scripts/shenzhen_community_comparison.py']
    source_files += sorted((REPO/'zoning').glob('*.py'))
    source_files += list((data.DATA/'lines').glob('*.pkl'))
    D = {d:input_only(d) for d in cfg['districts']}
    for d in D:
        source_files += [p for p in (data.DATA/d).rglob('*') if p.is_file() and p.suffix in ('.json','.npz')]
    source_files += [MCA/f'{d}.json' for d in HZ]
    write(OUT/'input_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in source_files})
    # Hashing answers freezes their identity, but their contents are never parsed here.
    units = read(UNITS); audit = {}
    for d,x in D.items():
        r = rule(x,cfg,True)
        print('building current inputs',d,flush=True)
        if d in data.CITIES['长沙']:
            cache=REPO/'outputs/shenzhen_full_source_cache_20261111'
            manifest=read(cache/'manifest.json')
            assert sha(cache/f'{d}.pkl')==manifest['files'][f'{d}.pkl']
            for key in ['rules','evidence','model','train','data']:
                assert cfg[key]==manifest['config'][key]
            base_x=load(cache/f'{d}.pkl')[0]
            assert np.array_equal(base_x.y,np.array([r.known[q][0] if q in r.known else -1 for q in range(x.n)]))
            assert np.array_equal(base_x.groups,x.groups)
            assert np.array_equal(base_x.anchor[...,base_x.anchor_blocks['community_clause']],x.community_school)
            # Fresh first-district build from the interrupted preparation verifies cache compatibility.
            if (OUT/'current_inputs'/f'{d}.pkl').exists():
                prior=load(OUT/'current_inputs'/f'{d}.pkl')[2]
                for key in ['pair','anchor','ref_sites','ref_schools','self_mask','y','supervised']:
                    assert np.array_equal(getattr(prior,key),getattr(base_x,key)),(d,key)
        else:
            base_x = evidence.build(x,r,cfg['evidence'])
        save(OUT/'current_inputs'/f'{d}.pkl',(x,r,base_x))
        print('current inputs',d,base_x.pair.shape,flush=True)
        if d not in HZ:
            continue
        pts,names = inventory(d,x.crs)
        hz = nearest_separated(x.coords,pts,names)
        assert not np.any((hz['distance']==0)&(hz['second_distance']==0))
        cs = road_aware(x.coords,pts,evidence.lines(x.lines_key or x.city))
        dh,dc = features(x,names,hz,units),features(x,names,cs,units)
        for key in ('community_name','community_strict','community_school'):
            assert np.array_equal(getattr(dh,key),getattr(x,key)), (d,key)
        rb = rule(x,cfg,False); rc = rule(dc,cfg,True)
        audit[d] = dict(sites=x.n,schools=x.school_n,inventory_points=len(pts),inventory_identities=len(set(names)),base_known=len(rb.known),
            hangzhou_strict=int(hz['strict'].sum()),hangzhou_loose=int(hz['loose'].sum()),
            changsha_strict=int(cs['strict'].sum()),changsha_loose=int(cs['loose'].sum()),
            hangzhou_new_rules=len(r.known)-len(rb.known),changsha_new_rules=len(rc.known)-len(rb.known),
            original_hangzhou_features_exact=True,road_classes=dict(Counter(evidence.lines(x.city).cls)))
        for method,est in [('hangzhou',hz),('changsha',cs)]:
            p=OUT/'estimates'/f'{d}_{method}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,**est,inventory_xy=pts,inventory_names=names)
        xb=evidence.build(dh,rb,cfg['evidence'])
        xc=evidence.build(dc,rc,cfg['evidence'])
        for arm,dx,rx in [('hangzhou_E',dh,rb),('changsha_E',dc,rb),('hangzhou_ER',dh,r),('changsha_ER',dc,rc),('changsha_keep_current_R',dc,r)]:
            xx={'hangzhou_ER':base_x,'hangzhou_E':xb,'changsha_ER':xc}.get(arm)
            if xx is None:
                xx=replace_community(xb if arm=='changsha_E' else base_x,dc)
            save(OUT/'arms'/arm/f'{d}.pkl',(dx,rx,xx))
        print('prepared',d,audit[d],flush=True)
    write(OUT/'input_audit.json',audit)
    write(OUT/'prepared_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('**/*.pkl')})


def verify_hashes(file, base):
    for name,value in read(file).items():
        assert sha(base/name)==value,name


def train_current():
    verify_hashes(OUT/'prepared_hashes.json',OUT)
    cfg=read(OUT/'protocol.json')['config'];seeds=cfg['train']['seeds']
    current={d:load(OUT/'current_inputs'/f'{d}.pkl') for d in cfg['districts']}
    X={d:v[2] for d,v in current.items()}; checks={}
    for d in HZ:
        # Only source districts' answer dictionaries are read for this target fit.
        D={o:(current[o][0] if o==d else data.with_list_gold(data.load(o))) for o in current}
        assert D[d].answers=={} and D[d].train_answers=={}
        for seed in seeds:
            p=OUT/'weights'/f'{d}_{seed}.pt';p.parent.mkdir(parents=True,exist_ok=True)
            prob=OUT/'reconstructed'/f'{d}_{seed}.npy';prob.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and prob.exists():
                continue
            t=time.time();res=train.fit(d,X,D,cfg,[seed])[seed]
            torch.save({k:v.detach().cpu() for k,v in res[0].items()},p)
            np.save(prob,res[1]);print('trained',d,seed,round(time.time()-t,1),'seconds',flush=True)
        pp=np.stack([np.load(OUT/'reconstructed'/f'{d}_{s}.npy') for s in seeds])
        old=np.load(REPO/'outputs/hz_comm_ER/probs'/f'{d}.npz')['probs']
        checks[d]=dict(bitwise_equal=bool(np.array_equal(pp,old)),max_abs_difference=float(np.abs(pp-old).max()),
                       seed_argmax_disagreements=int((pp.argmax(-1)!=old.argmax(-1)).sum()),ensemble_argmax_disagreements=int((pp.mean(0).argmax(-1)!=old.mean(0).argmax(-1)).sum()))
        write(OUT/'baseline_reconstruction.json',checks)
        assert checks[d]['ensemble_argmax_disagreements']==0,checks[d]
        assert np.allclose(pp,old,atol=2e-5,rtol=2e-5),checks[d]
    write(OUT/'weights_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'weights').glob('*.pt')})


def infer():
    verify_hashes(OUT/'prepared_hashes.json',OUT);verify_hashes(OUT/'weights_hashes.json',OUT)
    cfg=read(OUT/'protocol.json')['config'];seeds=cfg['train']['seeds'];replay={}
    for d in HZ:
        for arm in ARMS:
            dx,rx,x=load(OUT/'arms'/arm/f'{d}.pkl');ps=[]
            for seed in seeds:
                model=RetrievalScorer(x.np_,x.na,LinearRetrieval(x),seed,cfg['model']['width'])
                model.load_state_dict(torch.load(OUT/'weights'/f'{d}_{seed}.pt',map_location='cpu',weights_only=True))
                model=model.to(DEVICE).eval();p=train.predict_proba(model,x)
                repeat=train.predict_proba(model,x)
                assert np.array_equal(p,repeat),(d,arm,seed)
                if arm=='hangzhou_ER':
                    assert np.array_equal(p,np.load(OUT/'reconstructed'/f'{d}_{seed}.npy'))
                ps.append(p)
            P=np.stack(ps);folder=OUT/'predictions'/arm;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{d}.npz',probs=P,seeds=seeds)
            rows=[]
            for k,p in [('ensemble',P.mean(0))]+[(str(s),P[i]) for i,s in enumerate(seeds)]:
                for q in range(dx.n):
                    rows.append(dict(seed=k,q=q,name=dx.site_names[q],prediction=rx.known.get(q,[int(p[q].argmax())]),
                                     by=rx.kind.get(q,'model')))
            write(folder/f'{d}.json',rows)
            replay[f'{d}/{arm}']=dict(models=len(seeds),all_inferences_bitwise_replayed=True,rows=len(rows))
            print('frozen predictions',d,arm,flush=True)
    write(OUT/'inference_audit.json',replay)
    write(OUT/'prediction_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'predictions').rglob('*') if p.is_file()})


def score():
    verify_hashes(OUT/'prediction_hashes.json',OUT)
    cfg=read(OUT/'protocol.json')['config'];seeds=['ensemble']+[str(s) for s in cfg['train']['seeds']]
    D={d:data.with_list_gold(data.load(d)) for d in HZ}
    rows={arm:[] for arm in ARMS};per_seed={arm:{} for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            rr=[]
            for d,x in D.items():
                frozen=read(OUT/'predictions'/arm/f'{d}.json')
                src=x.answer_source or {}
                for r in frozen:
                    q=r['q']
                    if r['seed']!=seed or q not in x.answers:
                        continue
                    truth=set(x.answers[q]);pred=set(r['prediction']);source=src.get(q,'map')
                    scoped=pred & (x.scope|truth if source=='list' else x.scope)
                    exact=pred==truth;strict=scoped==truth
                    legacy=bool(pred&truth) if r['by']!='model' or source=='list' else strict
                    rr.append(dict(d=d,city='杭州',q=q,name=r['name'],truth=sorted(truth),prediction=sorted(pred),
                        source=source,multi=len(truth)>1,by=r['by'],block=evaluate.block(x,q),ok=exact,strict=strict,legacy=legacy))
            assert len(rr)==2272,len(rr)
            per_seed[arm][seed]=dict(correct=sum(r['ok'] for r in rr),n=len(rr))
            if seed=='ensemble':rows[arm]=rr
    def summary(rr):
        return dict(n=len(rr),correct=sum(r['ok'] for r in rr),accuracy=100*np.mean([r['ok'] for r in rr]),
            historical_scoped_strict=100*np.mean([r['strict'] for r in rr]),historical_any_hit=100*np.mean([r['legacy'] for r in rr]))
    groups={'杭州':lambda r:True,**{d:(lambda r,d=d:r['d']==d) for d in HZ},
            '地图':lambda r:r['source']=='map','名单':lambda r:r['source']=='list',
            '单校':lambda r:not r['multi'],'多校':lambda r:r['multi']}
    metrics={arm:{g:summary([r for r in rr if fn(r)]) for g,fn in groups.items()} for arm,rr in rows.items()}
    comparisons={}
    for a,b in [('changsha_E','hangzhou_E'),('changsha_ER','hangzhou_ER'),('changsha_keep_current_R','hangzhou_ER'),('hangzhou_ER','hangzhou_E'),('changsha_ER','changsha_E')]:
        comparisons[f'{a} minus {b}']={}
        ra,rb=rows[a],rows[b]
        assert [(r['d'],r['q'],r['truth']) for r in ra]==[(r['d'],r['q'],r['truth']) for r in rb]
        for g,fn in groups.items():
            ids=[i for i,r in enumerate(ra) if fn(r)]
            delta=np.array([int(ra[i]['ok'])-int(rb[i]['ok']) for i in ids])
            comparisons[f'{a} minus {b}'][g]=dict(delta_pp=100*float(delta.mean()),corrected=int((delta>0).sum()),harmed=int((delta<0).sum()),
                ci95=evaluate.bootstrap([ra[i] for i in ids],delta,20261104,3000))
    direct={}
    for arm in ['hangzhou_ER','changsha_ER']:
        direct[arm]={}
        for d,x in D.items():
            _,r,_=load(OUT/'arms'/arm/f'{d}.pkl');qs=[q for q in r.known if r.kind[q]=='community']
            scored=[q for q in qs if q in x.answers]
            direct[arm][d]=dict(all_rules=len(qs),scored=len(scored),consistent=sum(set(r.known[q])==set(x.answers[q]) for q in scored))
    result=dict(metrics=metrics,comparisons=comparisons,per_seed=per_seed,direct_rule_postscore_audit=direct)
    write(OUT/'results.json',result)
    for arm,rr in rows.items():write(OUT/'scored_rows'/f'{arm}.json',rr)
    print(json.dumps(result,ensure_ascii=False,indent=1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','infer','score']);a=ap.parse_args()
    {'prepare':prepare,'train':train_current,'infer':infer,'score':score}[a.phase]()
