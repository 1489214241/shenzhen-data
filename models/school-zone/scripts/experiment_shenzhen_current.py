"""Frozen external test of the current RetrievalScorer, with incomplete Shenzhen inputs.

Run prepare, train, score in separate processes. Target map labels are blocked until score.
This is an input ablation, not a complete-input benchmark or an adopted model change.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zoning import data, rules, evidence as E, model as M, train as T, DEVICE
from zoning import shenzhen_sources as SZ, shenzhen_policy as SP, shenzhen_policy_evaluate as SE

OUT = REPO / 'outputs/shenzhen_current_model_20261111'
TARGETS = ['龙华区', '龙岗区']
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
START = time.time()


def log(*args):
    print(round(time.time()-START, 1), *args, flush=True)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard():
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = str(os.fsdecode(args[0])).replace('\\', '/')
        if '/深圳市/' in p and any(x in p for x in ['/Gold/', '/评价数据/', '/polygons.json', '点落面']):
            raise PermissionError('Target evaluation data cannot enter preparation/training: '+p)
    sys.addaudithook(audit)


def build_common(dist, rule):
    """Native feature formulas; omit unavailable blocks in BOTH sources and targets.

    No empty road file is substituted. A missing school marker has per-query median
    distance features imputed, not an invented coordinate or a copied campus marker.
    """
    known = rule.known
    refs = sorted((q, s, rule.kind[q]) for q, ss in known.items()
                  if q not in dist.not_reference for s in ss)
    ri = np.array([r[0] for r in refs], int)
    rs = np.array([r[1] for r in refs], int)
    named = np.array([r[2] in ('named', 'policy_base') for r in refs], bool)
    groups = dist.groups
    _, inv, cnt = np.unique(groups[ri], return_inverse=True, return_counts=True)
    st, neg = rule.statuses, rule.negative
    qst, rst = st[:, rs], st[ri, rs]
    delta = dist.coords[ri][None] - dist.coords[:, None]
    dd = np.linalg.norm(delta, axis=-1)
    P = dict(distance=E.geometry(delta, 300),
             named_reference=np.broadcast_to(named[None, :, None], (dist.n, len(ri), 1)).astype('float32'),
             policy=E.policy(qst, neg[:, rs]),
             agreement=((qst >= 0) & (rst[None] >= 0) & (qst == rst[None])).astype('float32'),
             disagreement=((qst >= 0) & (rst[None] >= 0) & (qst != rst[None])).astype('float32'),
             within_2km=((dd <= 2000) & (dd > 1e-6))[..., None].astype('float32'))
    valid = np.isfinite(dist.school_xy).all(1)
    safe = dist.school_xy.copy()
    safe[~valid] = 0  # temporary only; every derived feature replaced below
    delta = safe[None] - dist.coords[:, None]
    ad = E.geometry(delta, 1000)
    dd = np.linalg.norm(delta, axis=-1)
    aw = ((dd <= 2000) & (dd > 1e-6))[..., None].astype('float32')
    if not valid.all():
        assert valid.any()
        ad[:, ~valid] = np.median(ad[:, valid], axis=1)[:, None]
        aw[:, ~valid] = np.median(aw[:, valid], axis=1)[:, None]
    A = dict(distance=ad, policy=E.policy(st, neg), within_2km=aw)
    pair, pb = E._stack(P, list(P))
    anchor, ab = E._stack(A, list(A))
    sup = np.array(sorted(q for q, ss in known.items() if len(ss) == 1 and q not in dist.not_training), int)
    x = E.Inputs(dist.name, pair, anchor, pb, ab, ri, rs,
                 -np.log(cnt[inv]).astype('float32'), groups[:, None] == groups[ri][None], groups,
                 np.array([known[q][0] if q in known else -1 for q in range(dist.n)]), sup, dist.school_n)
    x.membership = torch.nn.functional.one_hot(torch.tensor(np.r_[rs, np.arange(dist.school_n)], dtype=torch.long), dist.school_n).float()
    assert np.isfinite(pair).all() and np.isfinite(anchor).all()
    return x


def target_input(d):
    sites = SZ._load(d, '小区数据/坐标组.json')
    schools = SZ.load_schools(d)
    sid = [s['school_id'] for s in schools]
    trans = Transformer.from_crs(4326, 32650, always_xy=True)
    def project(points):
        a = np.array([p if p is not None else [np.nan, np.nan] for p in points])
        return np.stack(trans.transform(a[:, 0], a[:, 1]), axis=1)
    coords = project([s['representative_point_wgs84'] for s in sites])
    school_xy = project([s['school_point_wgs84'] for s in schools])
    decisions = SP.predict_district(d)
    known = {i: [sid.index(s) for s in r['policy_supported_school_ids']]
             for i, r in enumerate(decisions) if r['policy_supported_school_ids']}
    assert [r['coordinate_group_id'] for r in decisions] == [s['coordinate_group_id'] for s in sites]
    dist = SimpleNamespace(name=d, n=len(sites), school_n=len(schools), coords=coords,
                           school_xy=school_xy, groups=np.array([s['coordinate_group_id'] for s in sites]),
                           not_reference=set(), not_training=set())
    rule = SimpleNamespace(known=known, kind={q: 'policy_base' for q in known},
                           statuses=np.full((len(sites), len(schools), 4), -1, np.int8),
                           negative=np.zeros((len(sites), len(schools)), bool))
    meta = dict(sites=sites, schools=schools, known={str(q): ss for q, ss in known.items()},
                missing_school_marker_ids=[s['school_id'] for s in schools if s['school_point_wgs84'] is None],
                projected_crs=32650, policy_reference_sites=len(known),
                singleton_training_sites=sum(len(v) == 1 for v in known.values()))
    return build_common(dist, rule), meta


def prepare():
    guard()
    if OUT.exists():
        raise FileExistsError('Preserve frozen runs; output already exists')
    cfg = read(REPO/'configs/default.json')
    protocol = dict(status='frozen_before_training_and_target_scoring', model='zoning.model.RetrievalScorer',
        targets=TARGETS, sources=cfg['districts'], seeds=cfg['train']['seeds'][:3], steps=40,
        lr=.003, regularization=.01, width=32, warm_start_steps=100, chunk=64,
        source_config=cfg, additional_target_gold_references=0,
        pair_blocks=['distance', 'named_reference', 'policy', 'agreement', 'disagreement', 'within_2km'],
        school_blocks=['distance', 'policy', 'within_2km'],
        omitted_in_both_sources_and_targets=['crossing', 'learned_boundary', 'same_community', 'community_clause', 'name'],
        target_directional_states='unknown; no independent road/admin observations',
        missing_school_marker='per-query median of other school distance features; not a verified location',
        baseline='current model argmax except independently text-known sites keep their complete policy set',
        treatment='same frozen baseline predictions passed through shenzhen_policy.predict_district',
        ensemble='mean of 3 seed probabilities, no score-based selection',
        metric='exact school set vs preserved official_map_point_membership; not certified basic catchment accuracy',
        bootstrap=dict(seed=20261111, draws=3000, block_metres=2000, paired=True),
        adoption='development external input-ablation test only; no model adoption')
    write(OUT/'protocol.json', protocol)
    code = [Path(__file__)] + sorted((REPO/'zoning').glob('*.py'))
    for p in code:
        dest = OUT/'code'/p.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    populations, manifest, stats = {}, {}, {}
    for d in protocol['sources']:
        dist = data.with_mca_community(data.with_list_gold(data.load(d)))
        rc = cfg['rules']
        r = rules.run(dist, rc['preset'], rc['hangzhou_names'], rc['hangzhou_boundary'], rc['hangzhou_community_rule'])
        x = build_common(dist, r)
        lab = T.uncovered_answers(dist, x)
        ids = sorted(set(x.supervised.tolist()) | set(lab))
        y = x.y.copy()
        for q, s in lab.items():
            y[q] = s
        populations[d] = dict(ids=ids, labels=y[ids].tolist())
        stats[d] = dict(sites=x.n, schools=x.school_n, references=len(x.ref_sites), training=len(ids))
        save_x(d, x)
        # Freeze actual independent source observations and labels by hashes.
        for p in (data.DATA/d).rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.npz'):
                manifest[str(p.relative_to(REPO))] = sha(p)
        log('SOURCE', d, stats[d])
    for d in TARGETS:
        x, meta = target_input(d)
        save_x(d, x)
        write(OUT/'inputs'/f'{d}.json', meta)
        populations[d] = dict(ids=x.supervised.tolist(), labels=x.y[x.supervised].tolist())
        stats[d] = dict(sites=x.n, schools=x.school_n, references=len(x.ref_sites),
                        reference_sites=meta['policy_reference_sites'], training=len(x.supervised),
                        missing_marker=meta['missing_school_marker_ids'])
        for rel in ['schools.json', 'texts.json', '小区数据/坐标组.json', '政策规则/规则包.json']:
            p = SZ.DATA/d/rel
            manifest[str(p.relative_to(REPO))] = sha(p)
        log('TARGET', d, stats[d])
    write(OUT/'training_population.json', populations)
    write(OUT/'input_audit.json', stats)
    frozen = [OUT/'protocol.json', OUT/'training_population.json', OUT/'input_audit.json']
    frozen += sorted((OUT/'inputs').glob('*')) + sorted((OUT/'code').rglob('*.py'))
    write(OUT/'input_manifest.json', dict(original_source_sha256=manifest,
          frozen_sha256={str(p.relative_to(OUT)): sha(p) for p in frozen},
          target_gold_read=False, target_polygons_read=False))


def save_x(d, x):
    p = OUT/'inputs'/f'{d}.pkl'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        pickle.dump(x, f, protocol=5)


def load_x(d):
    with (OUT/'inputs'/f'{d}.pkl').open('rb') as f:
        return pickle.load(f)


def move(t, dev):
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in t.items()}


def check_frozen():
    m = read(OUT/'input_manifest.json')
    for rel, h in m['frozen_sha256'].items():
        assert sha(OUT/rel) == h, rel
    # Current model implementation must equal saved implementation.
    for p in (OUT/'code/zoning').glob('*.py'):
        assert sha(REPO/'zoning'/p.name) == sha(p), p.name


def frames_for(names, xs, pop, chunk):
    active = [d for d in names if pop[d]['ids']]
    ng = {d: len(set(xs[d].groups[pop[d]['ids']])) for d in active}
    total = sum(ng.values())
    frames = []
    for d in active:
        x = xs[d]
        ids = np.array(pop[d]['ids'])
        labels = dict(zip(ids, pop[d]['labels']))
        _, inv, count = np.unique(x.groups[ids], return_inverse=True, return_counts=True)
        # Exactly the default full-batch group-balanced objective; no chunk renormalization.
        # Native training_batches includes the target batch even if it is empty.
        weight = len(names) / total / count[inv]
        for lo in range(0, len(ids), chunk):
            qq = ids[lo:lo+chunk]
            t = M.batch(x, qq, labels)
            t['weight'] = torch.tensor(weight[lo:lo+chunk], dtype=torch.float32, device=DEVICE)
            frames.append(move(t, 'cpu'))
    return frames, dict(active_batches=active, total_groups=total, chunks=len(frames))


def verify_chunk_gradient(x):
    ids = x.supervised[:23]
    start = T.warm_start(x, steps=0)
    m = M.RetrievalScorer(x.np_, x.na, start, 20261111, 32).to(DEVICE)
    full = M.batch(x, ids)
    p = m(full)
    loss = -(p[torch.arange(len(ids), device=DEVICE), full['y']].log()*full['weight']).sum()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()
    m.zero_grad()
    for lo in range(0, len(ids), 7):
        t = M.batch(x, ids[lo:lo+7])
        t['weight'] = full['weight'][lo:lo+7]
        p = m(t)
        (-(p[torch.arange(len(t['ids']), device=DEVICE), t['y']].log()*t['weight']).sum()).backward()
    split = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    diff = float((grad-split).abs().max())
    assert torch.allclose(grad, split, atol=2e-6, rtol=2e-5), diff
    return dict(max_gradient_difference=diff, passed=True)


def train():
    guard()
    check_frozen()
    c = read(OUT/'protocol.json')
    pop = read(OUT/'training_population.json')
    xs = {d: load_x(d) for d in c['sources']+TARGETS}
    write(OUT/'chunk_verification.json', verify_chunk_gradient(xs[c['sources'][0]]))
    history = {}
    for d in TARGETS:
        x = xs[d]
        frames, traininfo = frames_for([d]+c['sources'], xs, pop, c['chunk'])
        # Empty supervision leaves the native distance prior unchanged; no pseudo-labels.
        start = T.warm_start(x) if len(x.supervised) else M.LinearRetrieval(x)
        log('TRAIN_READY', d, traininfo)
        all_probs = []
        for seed in c['seeds']:
            ck = OUT/'checkpoints'/f'{d}_{seed}.pt'
            ck.parent.mkdir(parents=True, exist_ok=True)
            m = M.RetrievalScorer(x.np_, x.na, start, seed, c['width']).to(DEVICE)
            optimizer = torch.optim.Adam(m.parameters(), lr=c['lr'])
            losses = []
            for step in range(c['steps']):
                optimizer.zero_grad()
                reg = m.regularization(c['regularization'])
                reg.backward()
                objective = float(reg.detach())
                for cpu in frames:
                    t = move(cpu, DEVICE)
                    p = m(t)
                    loss = -(p[torch.arange(len(t['ids']), device=DEVICE), t['y']].clamp_min(1e-30).log()*t['weight']).sum()
                    assert torch.isfinite(loss)
                    loss.backward()
                    objective += float(loss.detach())
                    del t, p, loss
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
                optimizer.step()
                losses.append(objective)
                if (step+1) % 10 == 0:
                    log('TRAIN', d, seed, step+1, round(objective, 5))
            torch.save(m.state_dict(), ck)
            probs = T.predict_proba(m, x, chunk=64)
            all_probs.append(probs)
            history[f'{d}/{seed}'] = dict(loss=losses, training=traininfo, checkpoint_sha256=sha(ck))
            write(OUT/'training_history.json', history)
        p = OUT/'probabilities'/f'{d}.npz'
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, probs=np.stack(all_probs), seeds=c['seeds'])
        del frames
    # Freeze all districts and seeds before loading evaluation labels in another process.
    prediction_hashes = {}
    for d in TARGETS:
        meta = read(OUT/'inputs'/f'{d}.json')
        probs = np.load(OUT/'probabilities'/f'{d}.npz')['probs']
        sid = [s['school_id'] for s in meta['schools']]
        for label, p in [('ensemble', probs.mean(0))]+[(str(s), p) for s, p in zip(c['seeds'], probs)]:
            raw = {s['coordinate_group_id']: [sid[j]] for s, j in zip(meta['sites'], p.argmax(1))}
            before = dict(raw)
            for q, schools in meta['known'].items():
                before[meta['sites'][int(q)]['coordinate_group_id']] = [sid[j] for j in schools]
            decisions = SP.predict_district(d, before)
            out = OUT/'predictions'/label/f'{d}.json'
            write(out, dict(raw_argmax=raw, before=before, decisions=decisions))
            prediction_hashes[str(out.relative_to(OUT))] = sha(out)
    write(OUT/'prediction_freeze.json', dict(prediction_sha256=prediction_hashes,
          probability_sha256={d: sha(OUT/'probabilities'/f'{d}.npz') for d in TARGETS},
          protocol_sha256=sha(OUT/'protocol.json'), target_gold_read=False, target_polygons_read=False,
          device=str(DEVICE), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    log('ALL_PREDICTIONS_FROZEN')


def metrics(rows, key):
    n = len(rows)
    tp = sum(len(set(r[key]) & set(r['gold'])) for r in rows)
    fp = sum(len(set(r[key]) - set(r['gold'])) for r in rows)
    fn = sum(len(set(r['gold']) - set(r[key])) for r in rows)
    correct = sum(set(r[key]) == set(r['gold']) for r in rows)
    return dict(n=n, correct=correct, exact_accuracy=correct/n if n else None,
                false_positive_school_assignments=fp, false_negative_school_assignments=fn,
                overreported_sites=sum(bool(set(r[key])-set(r['gold'])) for r in rows),
                omitted_sites=sum(bool(set(r['gold'])-set(r[key])) for r in rows),
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)


def compare(rows):
    b = np.array([set(r['before']) == set(r['gold']) for r in rows])
    a = np.array([set(r['after']) == set(r['gold']) for r in rows])
    return dict(raw_argmax=metrics(rows, 'raw_argmax'), before=metrics(rows, 'before'), after=metrics(rows, 'after'),
        corrected=int((a & ~b).sum()), harmed=int((b & ~a).sum()),
        changed_sites=sum(set(r['before']) != set(r['after']) for r in rows),
        wrongly_added_sites=sum(bool((set(r['after'])-set(r['before']))-set(r['gold'])) for r in rows))


def interval(rows, c):
    sums = {}
    for r in rows:
        block = r['block']
        delta = int(set(r['after']) == set(r['gold'])) - int(set(r['before']) == set(r['gold']))
        v = sums.setdefault(block, [0, 0])
        v[0] += delta
        v[1] += 1
    a = np.array(list(sums.values()))
    rng = np.random.default_rng(c['seed'])
    draws = []
    for _ in range(c['draws']):
        v = a[rng.integers(0, len(a), len(a))].sum(0)
        draws.append(v[0]/v[1])
    return dict(blocks=len(a), delta_95_percentile=np.quantile(draws, [.025, .975]).tolist(),
                describes='paired 2km spatial block resampling; not independent city validation')


def score():
    check_frozen()
    freeze = read(OUT/'prediction_freeze.json')
    for rel, h in freeze['prediction_sha256'].items():
        assert sha(OUT/rel) == h
    c = read(OUT/'protocol.json')
    reports, all_rows, per_seed = {}, [], {}
    gold_hashes = {}
    for label in ['ensemble']+[str(s) for s in c['seeds']]:
        label_rows = []
        for d in TARGETS:
            f = read(OUT/'predictions'/label/f'{d}.json')
            decisions = {r['coordinate_group_id']: r for r in f['decisions']}
            meta = read(OUT/'inputs'/f'{d}.json')
            gold = SZ.load_point_gold(d)
            audit = {r['coordinate_group_id']: r for r in SE.audit_gold(SP.load_policy(d), gold)}
            g = {r['coordinate_group_id']: r['point_school_ids'] for r in gold}
            assert set(g) <= set(decisions)
            trans = Transformer.from_crs(4326, 32650, always_xy=True)
            rows = []
            for s in meta['sites']:
                key = s['coordinate_group_id']
                xy = trans.transform(*s['representative_point_wgs84'])
                r = dict(district=d, site_id=key, building_name=s['building_name'],
                         raw_argmax=f['raw_argmax'][key], before=f['before'][key],
                         after=decisions[key]['predicted_school_ids'],
                         gold=g.get(key, []), has_map_gold=key in g,
                         semantic_pending=bool(audit.get(key, {}).get('unresolved_reasons')),
                         block=f'{d}/{int(xy[0]//2000)}/{int(xy[1]//2000)}',
                         applied_rules=decisions[key]['applied_rules'])
                rows.append(r)
            scored = [r for r in rows if r['has_map_gold']]
            label_rows += scored
            if label == 'ensemble':
                all_rows += rows
                reports[d] = dict(candidate_sites=len(rows), outside_map_unscored=len(rows)-len(scored),
                    all_map=compare(scored), single=compare([r for r in scored if len(r['gold']) == 1]),
                    multi=compare([r for r in scored if len(r['gold']) > 1]),
                    semantic_pending=compare([r for r in scored if r['semantic_pending']]),
                    semantic_not_flagged=compare([r for r in scored if not r['semantic_pending']]),
                    paired_interval=interval(scored, c['bootstrap']))
                gold_hashes[d] = sha(SZ.DATA/d/'Gold/小区_学校Gold.json')
        per_seed[label] = compare(label_rows)
    scored = [r for r in all_rows if r['has_map_gold']]
    report = dict(status='completed_development_input_ablation_not_adopted', districts=reports,
                  overall=compare(scored), per_seed=per_seed,
                  paired_interval=interval(scored, c['bootstrap']), gold_sha256=gold_hashes,
                  candidate_sites=len(all_rows), map_gold_sites=len(scored),
                  base_catchment_accuracy=None, protocol_sha256=sha(OUT/'protocol.json'),
                  prediction_freeze_sha256=sha(OUT/'prediction_freeze.json'))
    write(OUT/'scored_cases.json', all_rows)
    write(OUT/'results.json', report)
    lines = ['当前 RetrievalScorer 深圳外测（2026-11-11）', '',
        '实际重新训练：长沙、杭州八区为来源；龙华、龙岗为目标；3个固定随机种子，每个40步。',
        'K=0：没有增加任何深圳地图Gold参照；深圳Gold及polygon不进入训练和预测。',
        '这是缺失输入下的外测：所有区同时移除跨路、学出边界、社区和名称相似度特征。',
        '深圳四至状态全部为未知；高新缺独立校址，其距离特征逐小区以其他学校中位数填补，未伪造校址。',
        '规则前后使用完全相同的模型概率，前项已保留独立文字直接确认的32个小区集合。',
        '分母为原始地图有命中的1761个坐标组；全量1802组预测保留，41个图外组不当作可靠负例。', '',
        '地区\t规则前完整集合准确率\t规则后完整集合准确率\t修正\t损害']
    for d, item in list(reports.items())+[('合计', {'all_map': report['overall']})]:
        r = item['all_map']
        lines.append(f"{d}\t{r['before']['correct']}/{r['before']['n']} ({r['before']['exact_accuracy']:.2%})\t{r['after']['correct']}/{r['after']['n']} ({r['after']['exact_accuracy']:.2%})\t{r['corrected']}\t{r['harmed']}")
    ci = report['paired_interval']['delta_95_percentile']
    lines += ['', f'合计配对2公里空间块bootstrap增量95%区间：{ci[0]*100:.2f} 至 {ci[1]*100:.2f} 个百分点。',
        '该区间仅描述本次外测，不等同于独立城市或未来年度验证。',
        '原图标签含大学区/派位范围等语义待核项，results.json单独分组；不能把原图分数当作基础学区真值准确率。',
        '完整结果：results.json；逐小区：scored_cases.json；输入限制：input_audit.json；冻结方案：protocol.json；权重：checkpoints/。']
    (OUT/'实验结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    log('RESULT', report['overall'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare', 'train', 'score'])
    args = ap.parse_args()
    {'prepare': prepare, 'train': train, 'score': score}[args.phase]()
