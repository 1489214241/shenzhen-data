"""GraphAny（Zhao 等 ICLR 2025）同条件对照（科研推进/GraphAny同条件验证_20261111/方案与事先判定.md）。不改任何默认流程。

照官方代码（DeepGraphLearning/GraphAny，commit 02c810e，tmp/baseline_review_20261111/graph/）：
    图：小区坐标 k 近邻、双向、无自环；Â = D⁻¹A（邻居均值）。
    通道：X、L1=ÂX、L2=Â²X、H1=(I−Â)X、H2=(I−Â)H1。每个通道 W = F_ref⁺ Y_ref（最小范数最小二乘，目标区当地解析拟合），logits = F W。
    融合：各通道 logits 之间的平方欧氏距离，按困惑度 2^entropy（entropy=1）逐行二分求条件高斯分布（同 sklearn _binary_search_perplexity），
          得 5×4=20 维距离特征 → MLP（128 宽、2 层、BatchNorm、dropout 0.5）→ 3 个预测通道（X、L1、L2）的注意力 softmax(·/5)，
          融合 logits = Σ 注意力 × 通道 logits；交叉熵训练。
    训练：每步每个训练区抽 128 个有答案的小区当题；参照＝本区规则已知去掉这批题（及同楼盘），每校最多随机 5 个（官方 bootstrap）。
          AdamW lr 2e-4、weight_decay 0.02、最多 1000 步。预测：全部参照，不抽样。
与官方的差别（实现层面，照实写）：
    最小二乘用 W = F_refᵀ (F_ref F_refᵀ)⁺ Y（与 F_ref⁺ Y 恒等）：每区先算好全部小区的内积矩阵 F Fᵀ（float64），每步取参照子矩阵做特征分解求伪逆，
    截断阈值取 float32 机器精度 × max(行, 列)（官方 gelss 默认），CPU 多线程；熵归一距离用 numpy 向量化（与 sklearn 逐行二分一致）；
    同楼盘屏蔽：题的同楼盘参照不参与拟合（官方只去掉题本身）；
    多校已知小区当参照时 Y 行为平分的多热（官方只有单标签）。
特征（X 的列由本区学校决定，各区维数不同——GraphAny 原生允许）：baselines.node_table 的 geometry / policy 两种视图，数值列按本区全部小区标准化。
道路（适配）：graph='knn_cut' 时去掉跨主干、次干路或铁路河流的近邻边。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from .baselines import node_table
from .evidence import lines

CHANNELS = ['X', 'L1', 'L2', 'H1', 'H2']; PRED = ['X', 'L1', 'L2']
ENTROPY, ATT_T, HIDDEN, LR, WD, STEPS, BATCH, PER_LABEL = 1, 5., 128, 2e-4, .02, 1000, 128, 5
EPS32 = float(np.finfo(np.float32).eps)
_POOL = ThreadPoolExecutor(5)


# ---------------- 图与通道特征 ----------------
def knn_graph(dist, k=10, cut=False):
    from scipy.spatial import cKDTree
    xy = dist.coords + 1e-3 * np.arange(dist.n)[:, None]
    _, ii = cKDTree(xy).query(xy, k + 1); r = np.repeat(np.arange(dist.n), k); c = ii[:, 1:].ravel()
    e = np.array(sorted({(min(a, b), max(a, b)) for a, b in zip(r, c) if a != b}), int)
    if cut:
        import shapely
        ls = lines(dist.lines_key or dist.city); segs = shapely.linestrings(np.stack([dist.coords[e[:, 0]], dist.coords[e[:, 1]]], 1)); hit = np.zeros(len(e), bool)
        for key in ('major', 'secondary', 'rr'): hit[ls.by[key].query(segs, predicate='crosses')[0]] = True
        e = e[~hit]
    A = np.zeros((dist.n, dist.n)); A[e[:, 0], e[:, 1]] = 1; A[e[:, 1], e[:, 0]] = 1
    return A


def channel_features(dist, rules_out, view, graph):
    """{通道: [n, d] float64}。"""
    X, nnum = node_table(dist, rules_out, view); X = X.astype(np.float64)
    mu = X[:, :nnum].mean(0); sd = X[:, :nnum].std(0); sd[sd < 1e-6] = 1; X[:, :nnum] = (X[:, :nnum] - mu) / sd
    A = knn_graph(dist, 10, graph == 'knn_cut'); deg = A.sum(1, keepdims=True); M = np.where(deg > 0, A / np.maximum(deg, 1), 0)
    L1 = M @ X; L2 = M @ L1; H1 = X - L1; H2 = H1 - M @ H1
    return dict(X=X, L1=L1, L2=L2, H1=H1, H2=H2)


def ref_labels(dist, rules_out):
    """参照：规则已知（不含已登记不当参照的）；Y 行为平分的多热。"""
    refs = np.array(sorted(q for q in rules_out.known if q not in dist.not_reference), int)
    Y = np.zeros((len(refs), dist.school_n))
    for i, q in enumerate(refs):
        for s in rules_out.known[q]: Y[i, s] += 1 / len(rules_out.known[q])
    return refs, Y


# ---------------- 线性通道 ----------------
def _solve_one(K, dim, ref_idx, Y, out_idx):
    G = K[np.ix_(ref_idx, ref_idx)]; lam, V = (t.numpy() for t in torch.linalg.eigh(torch.from_numpy(G)))
    s = np.sqrt(np.clip(lam, 0, None)); keep = s > EPS32 * max(len(ref_idx), dim) * s.max()
    inv = np.where(keep, 1 / np.maximum(lam, 1e-300), 0)
    return K[np.ix_(out_idx, ref_idx)] @ (V @ (inv[:, None] * (V.T @ Y)))


def solve_logits(grams, dims, ref_idx, Y, out_idx, channels=CHANNELS):
    """每个通道 W = F_ref⁺ Y，返回 {通道: F[out] W}（用 F Fᵀ 计算，结果与直接求伪逆相同）。"""
    if len(ref_idx) == 0: return {c: np.zeros((len(out_idx), Y.shape[1])) for c in channels}
    futs = {c: _POOL.submit(_solve_one, grams[c], dims[c], ref_idx, Y, out_idx) for c in channels}
    return {c: f.result() for c, f in futs.items()}


def bootstrap(ref_idx, Y, rng, k=PER_LABEL):
    """官方 sample_k_nodes_per_label：每校最多随机 k 个参照（按主学校）。"""
    lab = Y.argmax(1); keep = []
    for s in np.unique(lab):
        ii = np.flatnonzero(lab == s); keep += rng.permutation(ii)[:k].tolist()
    keep = np.sort(np.array(keep, int)); return ref_idx[keep], Y[keep]


# ---------------- 距离特征（熵归一的条件高斯） ----------------
def entropy_dist(logits, entropy=ENTROPY, steps=100, tol=1e-5):
    """logits {通道: [n, S]} → [n, 20]。逐个小区、逐个通道行做 sklearn _binary_search_perplexity 式二分（目标熵 ln(2^entropy)）：
    每行收敛就停；概率和恰为 0 时才用 EPSILON_DBL=1e-8。距离照 sklearn 输入取 float32。"""
    Y = np.stack([logits[c] for c in CHANNELS], 1)                                  # [n, 5, S]
    D = (((Y[:, :, None] - Y[:, None]) ** 2).sum(-1)).astype(np.float32).astype(np.float64)
    n, C, _ = D.shape; off = ~np.eye(C, dtype=bool); desired = np.log(2.0 ** entropy)
    beta = np.ones((n, C, 1)); lo = np.full_like(beta, -np.inf); hi = np.full_like(beta, np.inf); done = np.zeros_like(beta, bool)
    for _ in range(steps):
        P = np.exp(-D * beta) * off; sP = P.sum(-1, keepdims=True); sP = np.where(sP == 0, 1e-8, sP)
        H = np.log(sP) + beta * (D * P).sum(-1, keepdims=True) / sP; diff = H - desired; done |= np.abs(diff) <= tol; Pn = P / sP
        if done.all(): break
        up = diff > 0; lo2 = np.where(up, beta, lo); hi2 = np.where(up, hi, beta)
        nb = np.where(up, np.where(np.isinf(hi2), beta * 2, (beta + hi2) / 2), np.where(np.isinf(lo2), beta / 2, (beta + lo2) / 2))
        beta = np.where(done, beta, nb); lo = np.where(done, lo, lo2); hi = np.where(done, hi, hi2)
    return torch.as_tensor(Pn[:, off], dtype=torch.float32)                         # 同 sklearn：返回最后一轮循环里算的概率（没收敛的行也是）                          # 行优先、去对角，与官方 pair_index 顺序相同


class MLP(nn.Module):
    """官方 MLP（改自 C&S）：Linear → ReLU → BN → dropout → Linear。"""
    def __init__(self, d_in, hidden, d_out, dropout=.5):
        super().__init__(); self.l1 = nn.Linear(d_in, hidden); self.bn = nn.BatchNorm1d(hidden); self.l2 = nn.Linear(hidden, d_out); self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.l1(x))
        if x.shape[0] > 1: x = self.bn(x)
        return self.l2(F.dropout(x, self.dropout, self.training))


class GraphAny(nn.Module):
    def __init__(self):
        super().__init__(); self.mlp = MLP(len(CHANNELS) * (len(CHANNELS) - 1), HIDDEN, len(PRED))

    def forward(self, logits, dist=None):
        dist = entropy_dist(logits) if dist is None else dist
        att = torch.softmax(self.mlp(dist) / ATT_T, -1)
        pred = torch.as_tensor(np.stack([logits[c] for c in PRED], 1), dtype=torch.float32)
        return (att[:, :, None] * pred).sum(1), att


# ---------------- 一个区的数据 ----------------
class Graph:
    def __init__(self, dist, rules_out, view, graph, labeled=None):
        """labeled：{小区: 学校} 训练题（训练用）；预测时不给。"""
        self.name = dist.name; self.groups = dist.groups; self.S = dist.school_n
        feats = channel_features(dist, rules_out, view, graph)
        self.grams = {c: v @ v.T for c, v in feats.items()}; self.dims = {c: v.shape[1] for c, v in feats.items()}
        self.refs, self.Y = ref_labels(dist, rules_out); self.labeled = labeled
        if labeled:
            self.lq = np.array(sorted(labeled), int)

    def logits_for(self, out_idx, rng=None, exclude=None, channels=CHANNELS):
        """out_idx 的各通道 logits；参照去掉 exclude（小区集合）及与 out_idx 同楼盘的；rng 给定时按官方每校抽 5 个。"""
        bad = set(self.groups[out_idx].tolist())
        keep = np.array([i for i, q in enumerate(self.refs) if self.groups[q] not in bad and (exclude is None or q not in exclude)], int)
        ri, Y = (self.refs[keep], self.Y[keep]) if len(keep) else (self.refs[:0], self.Y[:0])
        if rng is not None and len(ri): ri, Y = bootstrap(ri, Y, rng)
        return solve_logits(self.grams, self.dims, ri, Y, np.asarray(out_idx, int), channels)

    def predict_all(self, channels=CHANNELS):
        """全部小区的通道 logits：先用全部参照算一次；与参照同楼盘的小区按楼盘去掉那几个参照重算（同主模型"同楼盘参照不能用"）。"""
        n = len(self.groups); out = solve_logits(self.grams, self.dims, self.refs, self.Y, np.arange(n), channels)
        rg = set(self.groups[self.refs].tolist()); by = {}
        for q in range(n):
            if self.groups[q] in rg: by.setdefault(self.groups[q], []).append(q)
        for g, qs in by.items():
            qs = np.array(qs, int); sub = self.logits_for(qs, channels=channels)
            for c in channels: out[c][qs] = sub[c]
        return out


def train(graphs, seed, steps=STEPS, val=None, val_every=100):
    """graphs：带训练题的 Graph 列表。val：(Graph, 题, 答案) 时每 val_every 步验一次，返回 (模型, (最好准确率, NLL, 步数))。"""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = GraphAny()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); best = (-1, np.inf, 0); vc = None
    if val is not None:
        vg, vq, vy = val; lg = vg.predict_all(); lg = {c: v[vq] for c, v in lg.items()}; vc = (lg, torch.as_tensor(np.asarray(vy, np.int64)), entropy_dist(lg))
    for step in range(1, steps + 1):
        model.train(); opt.zero_grad(); loss = 0
        for g in graphs:
            b = g.lq[rng.permutation(len(g.lq))[:BATCH]] if len(g.lq) > BATCH else g.lq
            yb = torch.as_tensor(np.array([g.labeled[q] for q in b], np.int64))
            fused, _ = model(g.logits_for(b, rng=rng, exclude=set(b.tolist()))); loss = loss + F.cross_entropy(fused, yb)
        assert torch.isfinite(loss); loss.backward(); opt.step()
        if vc is not None and step % val_every == 0:
            model.eval()
            with torch.no_grad():
                fz, _ = model(vc[0], vc[2]); acc = float((fz.argmax(1) == vc[1]).float().mean()); nll = float(F.cross_entropy(fz, vc[1]))
            if acc > best[0] + 1e-9 or (abs(acc - best[0]) <= 1e-9 and nll < best[1]): best = (acc, nll, step)
    return model, best


def predict(model, g, lg=None):
    lg = g.predict_all() if lg is None else lg; model.eval()
    with torch.no_grad():
        fz, att = model(lg)
    return torch.softmax(fz, 1).numpy(), att.mean(0).numpy()


def softmax_np(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
