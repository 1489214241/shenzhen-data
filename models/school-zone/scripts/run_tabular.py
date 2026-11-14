"""同一张"小区×候选学校"特征表上的现代模型（科研推进/同表现代模型验证_20261111/方案与事先判定.md）。不改默认流程。
  python scripts/run_tabular.py --k 0 --seed 1 --select   # K=0：每个目标区在验证区上选设置与轮数，再全来源重训
  python scripts/run_tabular.py --k 1 --seed 2            # K>0：沿用 K=0 的选择
  python scripts/run_tabular.py --report
行与特征同 run_gbt.Table（82 列，前 10 候选）；二分类；分数＝正类概率。
ModernNCA：官方模块（third_party/TALENT）的编码器；距离、softmax、标签加权与官方 forward 相同，另加"同区同楼盘的行不当邻居"屏蔽与楼盘等权的样本权重。
输出 outputs/main_pre_v1/k<K>_s<seed>__tabular/<mnca_rows>[_policy]/probs/<区>.npz、eval.json；chosen.json。
"""
import sys, json, time, argparse, itertools
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts')); sys.path.insert(0, str(REPO / 'third_party'))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from zoning import evaluate, evidence, matching as M, DEVICE
from zoning.baselines import constrain
from run_matching import setup, BATCHES
from run_gbt import Table, K_CAND

ROOT = REPO / 'outputs/main_pre_v1'; DOC = REPO.parent / '科研推进' / '同表现代模型验证_20261111'
GRID = [dict(encoding=e, lr=lr) for e, lr in itertools.product(('none', 'plr_0.1'), (1e-3, 3e-3))]
SEEDS = [20261211, 20261212, 20261213]; MAX_EPOCHS = 30; BATCH = 1024


class Rows:
    """训练行（同主模型 train.items）与任一区的预测行；每行带全局楼盘编号。"""
    def __init__(self, D, R, T):
        self.D, self.R, self.T = D, R, T; keys = sorted({f'{d}|{g}' for d in D for g in D[d].groups}); self.code = {k: i for i, k in enumerate(keys)}

    def gid(self, d, ids): return np.array([self.code[f'{d}|{g}'] for g in self.D[d].groups[ids]], np.int64)

    def train(self, target, exclude=()):
        X, Y, W, G = [], [], [], []
        for d in self.D:
            if d in exclude: continue
            lab = dict(M.supervised(self.D[d], self.R[d]))
            if d != target: lab.update(M.uncovered(self.D[d], self.R[d]))
            ids = np.array(sorted(lab), int); y = np.array([lab[q] for q in ids]); cand = self.T.c[d][ids]; k = np.flatnonzero((cand == y[:, None]).any(1))
            _, inv, cnt = np.unique(self.D[d].groups[ids], return_inverse=True, return_counts=True); w = 1.0 / cnt[inv]
            X.append(self.T.f[d][ids[k]].reshape(-1, self.T.f[d].shape[-1])); Y.append((cand[k] == y[k, None]).astype(np.int64).ravel())
            W.append(np.repeat(w[k], K_CAND)); G.append(np.repeat(self.gid(d, ids[k]), K_CAND))
        return np.concatenate(X), np.concatenate(Y), np.concatenate(W).astype(np.float32), np.concatenate(G)

    def query(self, d, ids):
        ids = np.asarray(ids, int); return self.T.f[d][ids].reshape(-1, self.T.f[d].shape[-1]), np.repeat(self.gid(d, ids), K_CAND)


# ---------------- ModernNCA（官方模块） ----------------
def _mnca(cfg, d_in):
    from TALENT.model.models.modernNCA import ModernNCA
    emb = None if cfg['encoding'] == 'none' else dict(type='PLREmbeddings', n_frequencies=4, frequency_scale=float(cfg['encoding'].split('_')[1]), d_embedding=4, lite=True)
    return ModernNCA(d_in=d_in, d_num=d_in, d_out=2, dim=64, dropout=0.1, d_block=128, n_blocks=1, num_embeddings=emb, sample_rate=0.5).to(DEVICE)


def _enc(m, x):
    if m.num_embeddings is not None: x = m.num_embeddings(x).flatten(1)
    return m.post_encoder(m.encoder(x))


def _probs(m, Xq, Gq, Xc, Yc, Gc, chunk=2048):
    """官方推理：以全部训练行为邻居；同楼盘的行屏蔽。返回正类概率。"""
    m.eval(); out = []
    with torch.no_grad():
        zc = torch.cat([_enc(m, Xc[i:i + 8192]) for i in range(0, len(Xc), 8192)]); yc = F.one_hot(Yc, 2).float()
        for i in range(0, len(Xq), chunk):
            zq = _enc(m, Xq[i:i + chunk]); d = torch.cdist(zq, zc).masked_fill(Gq[i:i + chunk, None] == Gc[None], torch.inf)
            out.append((torch.softmax(-d, 1) @ yc)[:, 1])
    return torch.cat(out).cpu().numpy()


class Fitter:
    def __init__(self, Xtr, Ytr, Wtr, Gtr):
        self.qt = QuantileTransformer(output_distribution='normal', random_state=0).fit(Xtr)
        self.X = torch.as_tensor(self.qt.transform(Xtr), dtype=torch.float32, device=DEVICE); self.Y = torch.as_tensor(Ytr, device=DEVICE)
        self.W = torch.as_tensor(Wtr, device=DEVICE); self.G = torch.as_tensor(Gtr, device=DEVICE)

    def tq(self, X, G): return torch.as_tensor(self.qt.transform(X), dtype=torch.float32, device=DEVICE), torch.as_tensor(G, device=DEVICE)

    def fit(self, cfg, seed, epochs=MAX_EPOCHS, val=None):
        """val：(Xq, Gq, 每题候选, 答案) 时每轮验一次，返回最好轮数。"""
        torch.manual_seed(seed); gen = torch.Generator(device='cpu').manual_seed(seed); m = _mnca(cfg, self.X.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=1e-4); n = len(self.X); best = (-1, 0)
        for ep in range(1, epochs + 1):
            m.train(); perm = torch.randperm(n, generator=gen).to(DEVICE)
            for st in range(0, n, BATCH):
                b = perm[st:st + BATCH]
                if len(b) < 2: continue
                keep = (torch.rand(n, generator=gen) < 0.5).to(DEVICE); keep[b] = False; c = torch.nonzero(keep).squeeze(1)   # 官方 SNS：随机留一半邻居，题本身另拼进来
                zb = _enc(m, self.X[b]); zc = _enc(m, self.X[c])
                zc = torch.cat([zb, zc]); yc = F.one_hot(torch.cat([self.Y[b], self.Y[c]]), 2).float(); gc = torch.cat([self.G[b], self.G[c]])
                d = torch.cdist(zb, zc).masked_fill(self.G[b][:, None] == gc[None], torch.inf)                                # 同楼盘（含题本身）不当邻居
                p = torch.softmax(-d, 1) @ yc; lp = torch.log(p + 1e-7)
                loss = -(self.W[b] * lp[torch.arange(len(b), device=DEVICE), self.Y[b]]).sum() / self.W[b].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            if val is not None:
                Xq, Gq, cand, truth = val; s = _probs(m, Xq, Gq, self.X, self.Y, self.G).reshape(-1, K_CAND)
                acc = float((cand[np.arange(len(cand)), s.argmax(1)] == truth).mean())
                if acc > best[0] + 1e-9: best = (acc, ep)
        return m, best

    def predict(self, m, Xq, Gq): return _probs(m, Xq, Gq, self.X, self.Y, self.G)


def run(k, seed, select):
    torch.use_deterministic_algorithms(True, warn_only=True); t0 = time.time()
    c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; out = ROOT / f'{name}__tabular'; out.mkdir(parents=True, exist_ok=True)
    n_eval = sum(len(evaluate.eval_rows(D[d], R[d].known)) for d in D); ref_n = json.loads((ROOT / name / 'eval.json').read_text(encoding='utf-8'))['summary']['两城']['n']
    assert n_eval == ref_n, f'评价集与主表不一致 {n_eval} vs {ref_n}'
    X = {d: evidence.build(D[d], R[d], c['evidence']) for d in D}; T = Table(D, R, X); del X; RW = Rows(D, R, T); print(name, '特征', f'{time.time() - t0:.0f}s', flush=True)
    if select:
        chosen = {}
        for d in D:
            V = M.VALIDATION[d]; fz = Fitter(*RW.train(d, exclude={V})); vlab = M.uncovered(D[V], R[V]); vq = np.array(sorted(vlab), int)
            Xq, Gq = fz.tq(*RW.query(V, vq)); val = (Xq, Gq, T.c[V][vq], np.array([vlab[q] for q in vq])); trials = []
            for cfg in GRID:
                _, (acc, ep) = fz.fit(cfg, SEEDS[0], val=val); trials.append(dict(cfg=cfg, epochs=ep, val_acc=acc))
                print('  试', d, cfg, ep, f'{acc:.3f}', f'{time.time() - t0:.0f}s', flush=True)
            best = max(trials, key=lambda r: r['val_acc']); chosen[d] = dict(cfg=best['cfg'], epochs=best['epochs'], val_acc=best['val_acc'], validation=V, trials=trials)
            print('选', d, best['cfg'], best['epochs'], f"{best['val_acc']:.3f}", flush=True)
            del fz; torch.cuda.empty_cache()
        (out / 'chosen.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding='utf-8')
    chosen = json.loads((ROOT / 'k0_s1__tabular' / 'chosen.json').read_text(encoding='utf-8')); rows = {}
    for d in D:
        fz = Fitter(*RW.train(d)); Xq, Gq = fz.tq(*RW.query(d, np.arange(D[d].n))); n = D[d].n; S_ = []
        for s in SEEDS:
            m, _ = fz.fit(chosen[d]['cfg'], s, epochs=chosen[d]['epochs']); S_.append(fz.predict(m, Xq, Gq).reshape(n, K_CAND))
        P = np.zeros((n, D[d].school_n), np.float32); P[np.arange(n)[:, None], T.c[d]] = np.mean(S_, 0); er = evaluate.eval_rows(D[d], R[d].known)
        for suf, PP in (('', P), ('_policy', constrain(P, R[d]))):
            od = out / ('mnca_rows' + suf) / 'probs'; od.mkdir(parents=True, exist_ok=True); np.savez_compressed(od / f'{d}.npz', probs=PP[None].astype(np.float32))
            rows.setdefault('mnca_rows' + suf, []).extend(evaluate.correct(PP, D[d], er))
        print(name, d, f'{time.time() - t0:.0f}s', flush=True); del fz; torch.cuda.empty_cache()
    for m_, rr in rows.items():
        s_ = evaluate.summary(rr); (out / m_ / 'eval.json').write_text(json.dumps(dict(summary=s_, k=k, seed=seed), ensure_ascii=False, indent=1), encoding='utf-8')
        print(m_, {kk: round(s_[kk]['accuracy'], 2) for kk in ['两城', '长沙', '杭州']}, flush=True)


def report():
    res = {}; L = ['# 同表现代模型验证：结果（pre_v1 开发数字）', '']
    for k, seed in BATCHES:
        c, D, R = setup(k, seed); name = f'k{k}_s{seed}'; ev = c['evaluate']

        def rows_of(odir):
            rr = []
            for d in D:
                f = ROOT / odir / 'probs' / f'{d}.npz'
                if not f.exists(): return None
                rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
            return rr
        meth = {'完整模型': rows_of(name), 'mnca_rows': rows_of(f'{name}__tabular/mnca_rows'), 'mnca_rows_policy': rows_of(f'{name}__tabular/mnca_rows_policy'),
                'gbt': rows_of(f'{name}__gbt/gbt'), 'gbt_policy': rows_of(f'{name}__gbt/gbt_policy'),
                'ModernNCA原版只本区': rows_of(f'{name}__baselines/modernnca_raw'), 'ModernNCA原版只本区+排除': rows_of(f'{name}__baselines/modernnca_policy'),
                'ModernNCA同监督点特征+排除': rows_of(f'{name}__matching/mnca_policy'), 'MN成对适配版+排除': rows_of(f'{name}__matching/mn_pair_policy')}
        meth = {a: b for a, b in meth.items() if b is not None}
        acc = lambda r: float(np.mean([x['ok'] for x in r]) * 100)
        best = lambda cands: max([v for v in cands if v in meth], key=lambda v: acc(meth[v]))
        mn, g = best(['mnca_rows', 'mnca_rows_policy']), best(['gbt', 'gbt_policy']); old = best(['ModernNCA原版只本区', 'ModernNCA原版只本区+排除'])
        full = meth['完整模型']
        cmp = {'N1 完整模型−ModernNCA同表版': (full, mn, '完整模型'), 'N2 ModernNCA同表版−梯度提升树': (meth[mn], g, mn), '描述 ModernNCA同表版−ModernNCA原版': (meth[mn], old, mn)}
        res[name] = dict(n=len(full), accuracy={m_: evaluate.summary(v) for m_, v in meth.items()},
                         compare={kk: dict(evaluate.compare(a, meth[b], ev), which=f'{an} vs {b}') for kk, (a, b, an) in cmp.items()})
    (DOC / '结果.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    names = list(next(iter(res.values()))['accuracy'])
    L += ['## 准确率（两城｜长沙｜杭州）', '', '| 方法 | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
    for m_ in names:
        L.append(f'| {m_} | ' + ' | '.join((lambda a: f"{a['两城']['accuracy']:.2f}｜{a['长沙']['accuracy']:.2f}｜{a['杭州']['accuracy']:.2f}")(r['accuracy'][m_]) if m_ in r['accuracy'] else '—' for r in res.values()) + ' |')
    L += ['', '## 比较（两城差值 [95% 区间]（改对/改坏）；长沙；杭州）', '']
    for kk in next(iter(res.values()))['compare']:
        L.append(f'### {kk}'); L.append('')
        for name, r in res.items():
            x = r['compare'][kk]; L.append(f"- {name}（{x['which']}）：两城 {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]（{x['两城']['fixed']}/{x['两城']['broken']}）；长沙 {x['长沙']['delta']:+.2f} [{x['长沙']['ci'][0]:+.2f}, {x['长沙']['ci'][1]:+.2f}]；杭州 {x['杭州']['delta']:+.2f} [{x['杭州']['ci'][0]:+.2f}, {x['杭州']['ci'][1]:+.2f}]")
        L.append('')
    ch = json.loads((ROOT / 'k0_s1__tabular' / 'chosen.json').read_text(encoding='utf-8'))
    L += ['## 选中的设置（K=0 验证区上选，K=1 沿用）', ''] + [f"- {d}（验证区 {x['validation']}）：{x['cfg']}，{x['epochs']} 轮，验证 {x['val_acc']:.3f}" for d, x in ch.items()]
    (DOC / '结果.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int); ap.add_argument('--select', action='store_true'); ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report: report()
    else: run(a.k, a.seed, a.select)


if __name__ == '__main__':
    main()
