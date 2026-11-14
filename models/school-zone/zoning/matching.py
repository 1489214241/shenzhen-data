"""Matching Networks 同监督验证用的对照（科研推进/MatchingNetworks同监督验证_20261111/方案与事先判定.md）。不改任何默认流程。

和主模型同一批训练题（train.items 默认：目标区规则已知＋其余七区规则已知＋其余七区规则没判出的官方单校答案），
按"参照集＋待判小区"出题：每一步每区一题，参照集＝该区全部参照（同楼盘的屏蔽），待判小区＝该区训练题，楼盘等权。

点特征（跨区可共用的只有坐标）：坐标减本区全部小区均值，单位公里。共用编码器＝ModernNCA 官方模块（third_party/TALENT）。
    mn        Matching Networks 式 1：softmax_r(τ·cos(f(q), g(r)))，按学校加总；f=g；τ 固定 1（原文）或可学（初值 10）
    mn + fce  原文 §2.1.2 / 附录 A：g(r,S)=g'(r)+双向 LSTM 前后向；f(q,S)=注意力 LSTM 读支持集 3 步（读时屏蔽同楼盘参照）
              实现选择：LSTMCell 的隐状态用 h+r（原文写作拼接 [h, r]），h_k = ĥ_k + f'(q)
    proto     原型网络：学校原型＝该校参照嵌入均值（逐题去掉同楼盘参照），平方欧氏
    mnca      ModernNCA：欧氏（非平方）、温度 1，训练时每步随机留 50% 参照（SNS）
成对（适配版）：
    mn_pair   注意力分数＝线性项＋小网络，作用于主模型同一份"小区—参照"成对证据（evidence.build 的 pair），
              无学校一路、无参照所属学校证据、无周边统计、无楼盘修正；初值同主模型（距离列 −1）
没有参照的学校概率为 0；一个可用参照都没有的小区给均匀分布。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from . import DEVICE
from .model import mlp

REPO = Path(__file__).resolve().parents[1]
if str(REPO / 'third_party') not in sys.path: sys.path.insert(0, str(REPO / 'third_party'))
DIM = 64
FCE_STEPS = 3


# ---------------- 训练题与支持集 ----------------
def supervised(dist, rules_out):
    return {q: ss[0] for q, ss in rules_out.known.items() if len(ss) == 1 and q not in dist.not_training}


def uncovered(dist, rules_out):
    """规则没判出、唯一官方答案（同 train.uncovered_answers）。"""
    return {q: a[0] for q, a in dist.answers.items() if q not in rules_out.known and len(a) == 1 and a[0] >= 0}


class Task:
    """一个区的一道题：参照集（该区全部参照）＋待判小区（qs）及其答案（ys，预测时为 None）。"""
    def __init__(self, dist, rules_out, qs, ys=None, X=None):
        refs = sorted({(q, s) for q, ss in rules_out.known.items() if q not in dist.not_reference for s in ss})
        self.ri = np.array([r[0] for r in refs], np.int64); self.rs = np.array([r[1] for r in refs], np.int64); self.S = dist.school_n
        self.qs = np.asarray(qs, int); self.name = dist.name
        xy = (dist.coords - dist.coords.mean(0)) / 1000
        self.x = torch.as_tensor(xy, dtype=torch.float32, device=DEVICE)
        self.onehot = F.one_hot(torch.as_tensor(self.rs, device=DEVICE), self.S).float() if len(self.rs) else torch.zeros(0, self.S, device=DEVICE)
        self.valid = torch.as_tensor(dist.groups[self.qs][:, None] != dist.groups[self.ri][None], device=DEVICE)
        _, inv, cnt = np.unique(dist.groups[self.qs], return_inverse=True, return_counts=True)
        self.w = torch.as_tensor(1 / cnt[inv], dtype=torch.float32, device=DEVICE); self.n_groups = len(cnt)
        self.y = None if ys is None else torch.as_tensor(np.asarray(ys, int), device=DEVICE)
        if X is not None:                                                                  # 成对证据（mn_pair）
            assert np.array_equal(X.ref_sites, self.ri) and np.array_equal(X.ref_schools, self.rs), '成对证据的参照顺序与 Task 不一致'
            pair = torch.as_tensor(X.pair[self.qs], device=DEVICE); px = pair.clone(); px[:, :, 0] /= 10
            self.pair, self.px = pair, px


def tasks_for_training(target, D, R, exclude=(), X=None):
    """目标区规则已知＋其余区（exclude 之外）规则已知与没判出的单校答案。返回 Task 列表，权重按楼盘等权。"""
    out = []
    for d in D:
        if d in exclude: continue
        lab = dict(supervised(D[d], R[d]))
        if d != target: lab.update(uncovered(D[d], R[d]))
        if not lab: continue
        qs = sorted(lab); out.append(Task(D[d], R[d], qs, [lab[q] for q in qs], X[d] if X else None))
    tot = sum(t.n_groups for t in out)
    for t in out: t.w = t.w / tot
    return out


# ---------------- 点表示的模型 ----------------
def _encoder(cfg):
    from TALENT.model.models.modernNCA import ModernNCA
    emb = None if cfg['encoding'] == 'none' else dict(type='PLREmbeddings', n_frequencies=4, frequency_scale=float(cfg['encoding'].split('_')[1]), d_embedding=4, lite=True)
    return ModernNCA(d_in=2, d_num=2, d_out=2, dim=DIM, dropout=0.1, d_block=128, n_blocks=1, num_embeddings=emb, sample_rate=0.5)


def _embed(m, x):
    if m.num_embeddings is not None: x = m.num_embeddings(x).flatten(1)
    return m.post_encoder(m.encoder(x))


def _readout(logit, valid, onehot, S):
    logit = logit.masked_fill(~valid, -torch.inf); none = ~valid.any(1)
    a = torch.softmax(logit.masked_fill(none[:, None], 0), 1); p = a @ onehot
    return torch.where(none[:, None], torch.full_like(p, 1 / S), p)


class PointModel(nn.Module):
    def __init__(self, method, cfg):
        super().__init__(); self.method = method; self.cfg = cfg; self.enc = _encoder(cfg)
        if method == 'mn':
            self.log_tau = nn.Parameter(torch.tensor(float(np.log(10.)))) if cfg.get('tau') == 'learn' else None
            if cfg.get('fce'):
                self.g_lstm = nn.LSTM(DIM, DIM, bidirectional=True, batch_first=True); self.f_cell = nn.LSTMCell(DIM, DIM)

    def forward(self, t, gen=None):
        e = _embed(self.enc, t.x); fq = e[t.qs]; gr = e[t.ri]; valid = t.valid
        if len(t.ri) == 0: return torch.full((len(t.qs), t.S), 1 / t.S, device=DEVICE)
        if self.method == 'mn':
            if self.cfg.get('fce'):
                perm = torch.randperm(len(t.ri), generator=gen).to(DEVICE) if gen is not None else torch.arange(len(t.ri), device=DEVICE)
                out, _ = self.g_lstm(gr[perm][None]); g2 = torch.empty_like(gr); g2[perm] = gr[perm] + out[0, :, :DIM] + out[0, :, DIM:]; gr = g2
                h = torch.zeros_like(fq); c = torch.zeros_like(fq); r = torch.zeros_like(fq)
                for _ in range(FCE_STEPS):
                    hh, c = self.f_cell(fq, (h + r, c)); h = hh + fq
                    att = torch.softmax((h @ gr.T).masked_fill(~valid, -1e9), 1); r = att @ gr
                fq = h
            tau = self.log_tau.exp() if self.log_tau is not None else 1.0
            logit = tau * (F.normalize(fq, dim=-1) @ F.normalize(gr, dim=-1).T)
            return _readout(logit, valid, t.onehot, t.S)
        if self.method == 'mnca':
            if self.training:
                keep = torch.rand(len(t.ri), generator=gen).to(DEVICE) < 0.5
                if keep.sum() == 0: keep[0] = True
                valid = valid & keep[None]
            return _readout(-torch.cdist(fq, gr), valid, t.onehot, t.S)
        if self.method == 'proto':
            sums = t.onehot.T @ gr; cnt = t.onehot.sum(0)
            qi, rj = (~valid).nonzero(as_tuple=True)
            sub = torch.zeros(len(t.qs), t.S, DIM, device=DEVICE).index_put((qi, torch.as_tensor(t.rs, device=DEVICE)[rj]), gr[rj], accumulate=True)
            subc = torch.zeros(len(t.qs), t.S, device=DEVICE).index_put((qi, torch.as_tensor(t.rs, device=DEVICE)[rj]), torch.ones(len(qi), device=DEVICE), accumulate=True)
            n_ = cnt[None] - subc; proto = (sums[None] - sub) / n_.clamp_min(1)[..., None]
            logit = -(fq[:, None] - proto).square().sum(-1); has = n_ > 0.5
            logit = logit.masked_fill(~has, -torch.inf); none = ~has.any(1)
            p = torch.softmax(logit.masked_fill(none[:, None], 0), 1)
            return torch.where(none[:, None], torch.full_like(p, 1 / t.S), p)
        raise ValueError(self.method)


class PairModel(nn.Module):
    def __init__(self, np_, width):
        super().__init__(); self.p = nn.Parameter(torch.zeros(np_)); self.net = mlp(np_, width)
        with torch.no_grad(): self.p[0] = -1

    def forward(self, t, gen=None):
        if len(t.ri) == 0: return torch.full((len(t.qs), t.S), 1 / t.S, device=DEVICE)
        return _readout(t.pair @ self.p + self.net(t.px).squeeze(-1), t.valid, t.onehot, t.S)


def build(method, cfg, np_=None):
    return PairModel(np_, cfg['width']) if method == 'mn_pair' else PointModel(method, cfg)


# ---------------- 训练、验证、预测 ----------------
MAX_STEPS, VAL_EVERY, MIN_STEPS, PATIENCE = 800, 20, 100, 100     # 上限 800：冒烟时 ModernNCA 在天心验证区 400 步后仍略升（方案冻结前改）


def _score(model, task):
    model.eval()
    with torch.no_grad(): p = model(task)
    yy = task.y; acc = float((p.argmax(1) == yy).float().mean()); nll = float(-torch.log(p[torch.arange(len(yy), device=DEVICE), yy] + 1e-7).mean())
    return acc, nll


def fit(method, cfg, train_tasks, seed, steps=None, val=None):
    """steps 给定：照此步数训练；否则在 val（Task）上每 20 步按准确率（并列看 NLL）选步数并早停。返回 (模型, 选中步数, 验证准确率, 验证 NLL)。"""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(int(seed))
    model = build(method, cfg, train_tasks[0].pair.shape[-1] if method == 'mn_pair' else None).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4); best = (-1, np.inf, 0); last = 0
    for step in range(1, (steps or MAX_STEPS) + 1):
        model.train(); opt.zero_grad(); loss = 0
        for t in train_tasks:
            p = model(t, gen); loss = loss - (torch.log(p[torch.arange(len(t.qs), device=DEVICE), t.y] + 1e-7) * t.w).sum()
        assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        if steps is None and val is not None and step % VAL_EVERY == 0:
            acc, nll = _score(model, val)
            if acc > best[0] + 1e-9 or (abs(acc - best[0]) <= 1e-9 and nll < best[1]): best = (acc, nll, step); last = step
            if step >= MIN_STEPS and step - last >= PATIENCE: break
    if steps is not None: return model, steps, None, None
    return model, best[2], best[0], best[1]


def predict(model, task):
    model.eval()
    with torch.no_grad(): return model(task).cpu().numpy()


def grid(method):
    enc = ['none', 'plr_0.3', 'plr_1.0']; lr = [1e-3, 3e-3]
    if method == 'mn': return [dict(encoding=e, lr=l, fce=f, tau=tau) for e in enc for l in lr for f in (False, True) for tau in ('fixed', 'learn')]
    if method in ('proto', 'mnca'): return [dict(encoding=e, lr=l) for e in enc for l in lr]
    if method == 'mn_pair': return [dict(lr=l, width=w) for l in lr for w in (32, 64)]
    raise ValueError(method)


VALIDATION = {'天心区': '芙蓉区', '芙蓉区': '天心区', '雨花区': '开福区', '开福区': '雨花区', '拱墅区': '萧山区', '萧山区': '拱墅区', '富阳区': '拱墅区', '临安区': '萧山区'}
