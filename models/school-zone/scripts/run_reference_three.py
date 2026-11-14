"""Run the frozen Changsha pilot. Separate selection, inference, and scoring phases.

python scripts/run_reference_three.py --phase all
python scripts/run_reference_three.py --phase report
Resume skips completed, hashed artifacts. Does not modify the default pipeline.
"""
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
import numpy as np
import torch
from zoning import evaluate, rules, references
from zoning import matching as M
from zoning import reference_trials as T
from zoning.baselines import constrain
from zoning.data import CITIES
from run_matching import setup

ROOT = REPO / 'outputs/reference_three_changsha_20261111'
DOC = REPO.parent / '科研推进/三个参照方法长沙试验_20261111'
OLD = REPO / 'outputs/main_pre_v1'
METHODS = ['egnn', 'cnaps', 'feat']
SEEDS = [20261201, 20261202, 20261203]
DISTRICTS = CITIES['长沙']
BATCHES = [(0, 1), (1, 1)]


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(event):
    msg = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), **event)
    print(json.dumps(msg, ensure_ascii=False), flush=True)
    ROOT.mkdir(exist_ok=True, parents=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(msg, ensure_ascii=False) + '\n')


def source_manifest():
    paths = [REPO/'zoning/reference_trials.py', REPO/'scripts/run_reference_three.py',
             REPO/'scripts/test_reference_trials.py', DOC/'试验方案.txt',
             REPO/'configs/default.json']
    paths += sorted((REPO/'third_party/reference_cores').glob('*.py'))
    paths += [REPO/f'zoning/{name}.py' for name in ['matching','data','rules','references','evaluate','baselines']]
    return {str(p.relative_to(REPO.parent)): filehash(p) for p in paths}


def audit_data(k, seed):
    c, D, R = setup(k, seed)
    batch = f'k{k}_s{seed}'
    old_eval = json.loads((OLD/batch/'eval.json').read_text(encoding='utf-8'))
    assert sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in DISTRICTS) == old_eval['summary']['长沙']['n']
    if k:
        r0 = {d: rules.run(D[d], c['rules']['preset'], c['rules'].get('hangzhou_names', 'all'),
              c['rules'].get('hangzhou_boundary', 'v1'), c['rules'].get('hangzhou_community_rule', False)) for d in D}
        _, picked = references.apply(c['references'], D, r0)
        prev = json.loads((OLD/batch/'references.json').read_text(encoding='utf-8'))
        assert json.loads(json.dumps(picked)) == prev, 'Historical reference draw mismatch'
    audit = {}
    for d in D:
        rows = evaluate.eval_rows(D[d], R[d].known)
        audit[d] = dict(n=D[d].n, schools=D[d].school_n, known=R[d].known,
                        excluded_reference=sorted(D[d].not_reference), excluded_training=sorted(D[d].not_training),
                        coords_sha=hashlib.sha256(D[d].coords.tobytes()).hexdigest(),
                        groups_sha=digest(D[d].groups.tolist()), eval_rows=rows,
                        answers_sha=digest(D[d].answers), negative_sha=hashlib.sha256(R[d].negative.tobytes()).hexdigest())
    f = ROOT/batch/'data_audit.json'
    if f.exists(): assert digest(json.loads(f.read_text(encoding='utf-8'))) == digest(json.loads(json.dumps(audit)))
    else: save(f, audit)
    return c, D, R


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert (ROOT/'tests.json').exists(), 'Run test_reference_trials.py first'
    manifest = source_manifest()
    f = ROOT/'source_manifest.json'
    if f.exists(): assert json.loads(f.read_text(encoding='utf-8')) == manifest, 'Source changed after protocol registration'
    else: save(f, manifest)
    for k, s in BATCHES: audit_data(k, s)
    log(dict(phase='registered', device=torch.cuda.get_device_name(), sources=digest(manifest), methods=METHODS))


def select():
    c, D, R = audit_data(0, 1)
    for method in METHODS:
        for d in DISTRICTS:
            dest = ROOT/'selection'/method/(d+'.json')
            if dest.exists(): continue
            v = M.VALIDATION[d]
            tasks = T.train_tasks(d, D, R, exclude={v})
            val = T.DistrictTask(D[v], R[v], M.uncovered(D[v], R[v]))
            # Check training-pool identities and report unsupported training queries explicitly.
            train_record = {}
            for t in tasks:
                unsupported = sum(t.labels[int(q)] not in t.support(int(q))[1].tolist() for q in t.qs)
                train_record[t.name] = dict(qs=t.qs.tolist(), labels=[t.labels[int(q)] for q in t.qs],
                                            groups=len(t.group_queries), unsupported=unsupported)
            assert v not in train_record
            trials = []
            for lr in [.001, .003]:
                trialfile = dest.with_name(d+f'_lr{lr}.json')
                if trialfile.exists():
                    trials.append(json.loads(trialfile.read_text(encoding='utf-8')))
                    continue
                start = time.time()
                log(dict(phase='select_start', method=method, target=d, validation=v, lr=lr))
                model, info = T.fit(method, tasks, SEEDS[0], lr, 240, val=val,
                    log=lambda z: log(dict(phase='select', method=method, target=d, lr=lr, **z)))
                row = dict(lr=lr, **info, seconds=time.time()-start)
                save(trialfile, row)
                trials.append(row)
                del model
                torch.cuda.empty_cache()
            best = max(trials, key=lambda r: (round(r['val_acc'], 9), -r['val_nll']))
            save(dest, dict(method=method, target=d, validation=v, lr=best['lr'], steps=best['steps'],
                            trials=trials, train_pool=train_record))
            log(dict(phase='selected', method=method, target=d, lr=best['lr'], steps=best['steps'], val_acc=best['val_acc']))
    choices = {m: {d: json.loads((ROOT/'selection'/m/(d+'.json')).read_text(encoding='utf-8')) for d in DISTRICTS} for m in METHODS}
    frozen = dict(choices=choices, seeds=SEEDS, batches=BATCHES, sources=source_manifest(),
                  protocol='coordinate_native_pilot_v1', selection_metric='raw accuracy then NLL')
    f = ROOT/'freeze.json'
    if f.exists(): assert json.loads(f.read_text(encoding='utf-8')) == json.loads(json.dumps(frozen))
    else: save(f, frozen)
    log(dict(phase='frozen', sha256=filehash(f)))


def run_final():
    frozen = json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    assert frozen['sources'] == source_manifest()
    frozen_sha = filehash(ROOT/'freeze.json')
    for k, s in BATCHES:
        c, D, R = audit_data(k, s)
        batch = f'k{k}_s{s}'
        for method in METHODS:
            for d in DISTRICTS:
                ch = frozen['choices'][method][d]
                tasks = T.train_tasks(d, D, R)
                task = T.DistrictTask(D[d], R[d])
                qs = [e['q'] for e in evaluate.eval_rows(D[d], R[d].known)]
                for seed in SEEDS:
                    out = ROOT/batch/method/d/str(seed)
                    if (out/'complete.json').exists():
                        rec = json.loads((out/'complete.json').read_text(encoding='utf-8'))
                        assert rec['freeze_sha256'] == frozen_sha
                        assert filehash(out/'prediction.npz') == rec['prediction_sha256']
                        continue
                    out.mkdir(exist_ok=True, parents=True)
                    start = time.time()
                    log(dict(phase='final_start', batch=batch, method=method, target=d, seed=seed, steps=ch['steps']))
                    model, info = T.fit(method, tasks, seed, ch['lr'], ch['steps'],
                        log=lambda z: log(dict(phase='final', batch=batch, method=method, target=d, seed=seed, **z)))
                    train_seconds = time.time()-start
                    torch.save(dict(state_dict=model.state_dict(), method=method, lr=ch['lr'], steps=ch['steps'],
                                    seed=seed, freeze_sha256=frozen_sha), out/'model.pt')
                    start = time.time()
                    pp = T.predict(model, task, qs)
                    np.savez_compressed(out/'prediction.npz', qs=np.asarray(qs), probs=pp)
                    save(out/'complete.json', dict(freeze_sha256=frozen_sha, prediction_sha256=filehash(out/'prediction.npz'),
                        checkpoint_sha256=filehash(out/'model.pt'), train_seconds=train_seconds,
                        inference_seconds=time.time()-start, info=info,
                        train_queries={t.name: len(t.qs) for t in tasks}))
                    log(dict(phase='final_saved', batch=batch, method=method, target=d, seed=seed,
                             train_seconds=round(train_seconds, 2), inference_seconds=round(time.time()-start, 2)))
                    del model
                    torch.cuda.empty_cache()


def comparison(a, b, cfg):
    assert len(a) == len(b) and all((x['d'], x['block']) == (y['d'], y['block']) for x,y in zip(a,b))
    delta = np.array([x['ok'] for x in a], float)-np.array([x['ok'] for x in b], float)
    return dict(delta=float(delta.mean()*100), ci=evaluate.bootstrap(a, delta, cfg['bootstrap_seed'], cfg['draws']),
                fixed=int((delta>0).sum()), broken=int((delta<0).sum()))


def report():
    result = {}
    lines = ['三个参照方法长沙初筛结果（pre_v1，坐标任务适配，开发数字）', '',
             '只评价长沙；八区训练来源；K=0/K=1各一批参照抽样；新方法每批三个训练种子。',
             '主口径为规则未知题的历史单校exact-top1，非完整政策集合流程评分。', '']
    for k, s in BATCHES:
        c, D, R = audit_data(k, s)
        batch = f'k{k}_s{s}'
        methods = {}
        summary = {}
        seeds_rows = {m: {str(seed): {'raw':[], 'policy':[]} for seed in SEEDS} for m in METHODS}
        unsupported_rows = []
        for d in DISTRICTS:
            er = evaluate.eval_rows(D[d], R[d].known)
            t = T.DistrictTask(D[d], R[d])
            unsupported_rows += [not set(e['truth']).issubset(set(t.support(e['q'])[1].tolist())) for e in er]
        for method in METHODS:
            for suffix in ['', '_policy']:
                rows = []
                for d in DISTRICTS:
                    er = evaluate.eval_rows(D[d], R[d].known)
                    ps = []
                    for seed in SEEDS:
                        path = ROOT/batch/method/d/str(seed)/'prediction.npz'
                        z = np.load(path)
                        assert z['qs'].tolist() == [e['q'] for e in er]
                        P = np.ones((D[d].n, D[d].school_n), np.float32)/D[d].school_n
                        P[z['qs']] = z['probs']
                        ps.append(P)
                        single = constrain(P, R[d]) if suffix else P
                        seeds_rows[method][str(seed)]['policy' if suffix else 'raw'] += evaluate.correct(single, D[d], er)
                    P = np.mean(ps, 0)
                    if suffix: P = constrain(P, R[d])
                    rows += evaluate.correct(P, D[d], er)
                methods[method+suffix] = rows
        oldpaths = {
            'full': OLD/batch, 'gbt_policy': OLD/(batch+'__gbt')/'gbt_policy',
            'mn_pair_policy': OLD/(batch+'__matching')/'mn_pair_policy',
            'mn_native_policy': OLD/(batch+'__matching')/'mn_policy',
            'proto_native_policy': OLD/(batch+'__matching')/'proto_policy',
            'copy_nearest': OLD/(batch+'__baselines')/'copy_nearest',
            'nearest_school': OLD/(batch+'__baselines')/'nearest_school',
        }
        historical_hashes = {}
        for name, folder in oldpaths.items():
            rows = []
            for d in DISTRICTS:
                f = folder/'probs'/(d+'.npz')
                assert f.exists(), f
                historical_hashes[str(f.relative_to(REPO))] = filehash(f)
                P = np.load(f)['probs']
                if P.ndim == 3: P = P.mean(0)
                rows += evaluate.correct(P, D[d], evaluate.eval_rows(D[d], R[d].known))
            methods[name] = rows
        for name, rows in methods.items():
            ss = evaluate.summary(rows)
            ss.pop('两城', None)
            unsupported = [r for r,u in zip(rows, unsupported_rows) if u]
            supported = [r for r,u in zip(rows, unsupported_rows) if not u]
            ss['no_support_truth'] = dict(n=len(unsupported), accuracy=float(np.mean([r['ok'] for r in unsupported])*100) if unsupported else None)
            ss['supported_truth'] = dict(n=len(supported), accuracy=float(np.mean([r['ok'] for r in supported])*100))
            summary[name] = ss
        contrasts = {}
        for method in METHODS:
            contrasts[method+'_policy'] = {other: comparison(methods[method+'_policy'], methods[other], c['evaluate'])
                for other in ['full','gbt_policy','mn_pair_policy','mn_native_policy','proto_native_policy','copy_nearest','nearest_school']}
        per_seed = {m: {seed: {v: evaluate.summary(rows)['长沙'] for v, rows in variants.items()}
                           for seed, variants in seedmap.items()} for m, seedmap in seeds_rows.items()}
        result[batch] = dict(summary=summary, comparisons=contrasts, seed_results=per_seed,
                             historical_prediction_sha256=historical_hashes)
        lines += [batch, '方法\t长沙准确率\t天心\t芙蓉\t雨花\t开福']
        for name in ['full','gbt_policy','mn_pair_policy','mn_native_policy','proto_native_policy','egnn','egnn_policy','cnaps','cnaps_policy','feat','feat_policy','copy_nearest','nearest_school']:
            ss = summary[name]
            lines.append(name+'\t'+'\t'.join(f"{ss[d]['accuracy']:.2f}" for d in ['长沙']+DISTRICTS))
        lines += ['', '新方法（政策后处理）减去各对照：差值 [95%空间块区间]，纠正/损害']
        for method, comps in contrasts.items():
            for other, r in comps.items():
                lines.append(f"{method} - {other}: {r['delta']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], {r['fixed']}/{r['broken']}")
        lines += ['', '无可用参照覆盖完整真值学校的题数：'+str(sum(unsupported_rows)), '']
    save(ROOT/'report.json', result)
    DOC.mkdir(exist_ok=True, parents=True)
    (DOC/'结果.txt').write_text('\n'.join(lines), encoding='utf-8')
    save(DOC/'全部数字.json', result)
    log(dict(phase='report_complete', summary={b:{m:r['summary'][m]['长沙']['accuracy'] for m in r['summary']} for b,r in result.items()}))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['all','prepare','select','run','report'], default='all')
    a = p.parse_args()
    if a.phase in ['all','prepare']: prepare()
    if a.phase in ['all','select']: select()
    if a.phase in ['all','run']: run_final()
    if a.phase in ['all','report']: report()

if __name__ == '__main__': main()
