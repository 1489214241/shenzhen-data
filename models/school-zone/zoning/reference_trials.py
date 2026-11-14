"""Changsha pilot: original EGNN graph / CNAPs FiLM / FEAT attention with coordinate encoders.

All legal references, per-query project removal BEFORE any contextual processing.
No query-query inference and no target answers in training. See frozen experiment protocol.
"""
import sys
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import DEVICE
from . import matching as M

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party'))
from reference_cores.egnn import GraphNetwork
from reference_cores.feat import MultiHeadAttention
from reference_cores.cnaps import FilmLayerNetwork, LinearClassifierAdaptationNetwork

DIM = 32


class BatchedClassifier(LinearClassifierAdaptationNetwork):
    """Official independent classwise residual processors, evaluated in one batch."""
    def forward(self, representation_dict):
        reps = torch.cat([representation_dict[c] for c in sorted(representation_dict)], 0)
        return dict(weight_mean=self.weight_means_processor(reps),
                    bias_mean=self.bias_means_processor(reps).reshape(-1))


def independent_class_attention(attn, h, labels):
    """FEAT auxiliary self-attention, one block per class in a single masked matrix.

    Algebraically the same independent class sequences as the official loop. There
    is no interaction between differently labeled instances. One attention head.
    """
    q, k, v = attn.w_qs(h), attn.w_ks(h), attn.w_vs(h)
    score = q @ k.T / attn.attention.temperature
    score = score.masked_fill(labels[:, None] != labels[None], -torch.inf)
    weights = attn.attention.dropout(score.softmax(-1))
    return attn.layer_norm(h + attn.dropout(attn.fc(weights @ v)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freq', torch.tensor([.25, .5, 1., 2.]))
        self.l1 = nn.Linear(18, DIM)
        self.l2 = nn.Linear(DIM, DIM)

    def forward(self, x, film=None):
        a = (x[..., None] * self.freq * (2 * np.pi)).flatten(-2)
        h = torch.cat((x, a.sin(), a.cos()), -1)
        h = self.l1(h)
        if film is not None: h = h * film['gamma1'] + film['beta1']
        h = F.relu(h)
        h = self.l2(h)
        if film is not None: h = h * film['gamma2'] + film['beta2']
        return F.relu(h)


class DistrictTask:
    def __init__(self, dist, rules, labels=None):
        self.name, self.S = dist.name, dist.school_n
        self.x = torch.as_tensor((dist.coords - dist.coords.mean(0)) / 1000, dtype=torch.float32, device=DEVICE)
        self.groups = np.asarray(dist.groups)
        refs = sorted({(q, s) for q, ss in rules.known.items() if q not in dist.not_reference for s in ss})
        self.ri = np.asarray([q for q, _ in refs], dtype=int)
        self.rs = torch.tensor([s for _, s in refs], device=DEVICE, dtype=torch.long)
        self.labels = dict(labels or {})
        self.qs = np.asarray(sorted(self.labels), dtype=int)
        self.y = torch.tensor([self.labels[q] for q in self.qs], device=DEVICE)
        gs = sorted(set(self.groups[self.qs]))
        self.group_queries = [self.qs[self.groups[self.qs] == g] for g in gs]

    def support(self, q):
        ix = np.flatnonzero(self.groups[self.ri] != self.groups[q])
        return self.x[self.ri[ix]], self.rs[ix]


def train_tasks(target, D, R, exclude=()):
    out = []
    for d in D:
        if d in exclude: continue
        lab = M.supervised(D[d], R[d])
        if d != target: lab.update(M.uncovered(D[d], R[d]))
        if lab: out.append(DistrictTask(D[d], R[d], lab))
    return out


def class_means(h, labels):
    classes, inv = labels.unique(sorted=True, return_inverse=True)
    # matmul instead of CUDA atomic scatter for reproducibility
    oh = F.one_hot(inv, len(classes)).to(h.dtype)
    return classes, (oh.T @ h) / oh.sum(0)[:, None]


def expand_probability(logits, classes, nschools):
    p = torch.zeros(len(logits), nschools, device=logits.device)
    return p.index_copy(1, classes, logits.softmax(-1))


class ReferenceModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.enc = Encoder()
        if method == 'egnn':
            self.graph = GraphNetwork(DIM, DIM, 16, 3, dropout=0.)
        elif method == 'feat':
            self.attn = MultiHeadAttention(1, DIM, DIM, DIM, dropout=.5)
        elif method == 'cnaps':
            self.set_enc = nn.Sequential(nn.Linear(18, DIM), nn.ReLU(), nn.Linear(DIM, DIM), nn.ReLU())
            self.film = FilmLayerNetwork(DIM, 1, DIM)
            self.classifier = BatchedClassifier(DIM)
        else:
            raise ValueError(method)

    def forward(self, sx, sy, qx, nschools, y=None):
        if len(sy) == 0:
            z = next(self.parameters()).sum() * 0
            return torch.ones(len(qx), nschools, device=qx.device) / nschools + z, z
        if self.method == 'egnn':
            # One independent graph per query; full support, no joint-query graph.
            h = self.enc(torch.cat((sx, qx), 0))
            hs, hq = h[:len(sx)], h[len(sx):]
            nodes = torch.cat((hs[None].expand(len(qx), -1, -1), hq[:, None]), 1)
            n = len(sy) + 1
            edge = torch.full((len(qx), 2, n, n), .5, device=qx.device)
            same = (sy[:, None] == sy[None]).float()
            edge[:, 0, :-1, :-1] = same
            edge[:, 1, :-1, :-1] = 1 - same
            # Non-transductive official code initializes query self edge to .5.
            edges = self.graph(nodes, edge)
            vote = edges[-1][:, 0, -1, :-1] @ F.one_hot(sy, nschools).float()
            p = vote / vote.sum(-1, keepdim=True).clamp_min(1e-20)
            loss = h.sum() * 0
            if y is not None:
                truth = (y[:, None] == sy[None]).float()
                # Include both query->support and support->query, as official loss.
                truth = torch.stack((truth, truth), 1)
                for i, e in enumerate(edges):
                    v = torch.stack((e[:, 0, -1, :-1], e[:, 0, :-1, -1]), 1)
                    bce = F.binary_cross_entropy(1-v, 1-truth, reduction='none')
                    pos = (bce * truth).sum((1, 2)) / truth.sum((1, 2)).clamp_min(1)
                    neg = (bce * (1-truth)).sum((1, 2)) / (1-truth).sum((1, 2)).clamp_min(1)
                    loss = loss + (.5 if i < len(edges)-1 else 1.) * (pos + neg).mean() / len(edges)
            return p, loss
        if self.method == 'cnaps':
            a = (sx[..., None] * self.enc.freq * (2*np.pi)).flatten(-2)
            context = self.set_enc(torch.cat((sx, a.sin(), a.cos()), -1)).mean(0, keepdim=True)
            film = self.film(context)[0]
            hs, hq = self.enc(sx, film), self.enc(qx, film)
            classes, proto = class_means(hs, sy)
            params = self.classifier(dict(enumerate(proto.split(1))))
            logits = F.linear(hq, params['weight_mean'], params['bias_mean'])
        else:
            hs, hq = self.enc(sx), self.enc(qx)
            classes, proto = class_means(hs, sy)
            proto = self.attn(proto[None], proto[None], proto[None])[0]
            logits = -(hq[:, None] - proto[None]).square().sum(-1)
        p = expand_probability(logits, classes, nschools)
        loss = logits.sum() * 0
        if y is not None:
            supported = (y[:, None] == classes[None]).any(1)
            if supported.any():
                local_y = (y[supported, None] == classes[None]).long().argmax(1)
                loss = F.cross_entropy(logits[supported], local_y)
            if self.method == 'feat':
                # Original FEAT auxiliary objective, grouped by actual unequal shot counts.
                # Training-query labels enter only this loss, never primary forward logits.
                all_h, all_y = torch.cat((hs, hq)), torch.cat((sy, y))
                adapted = independent_class_attention(self.attn, all_h, all_y)
                ac, centers = class_means(adapted, all_y)
                aux_logits = -(all_h[:, None] - centers[None]).square().sum(-1)
                aux_y = (all_y[:, None] == ac[None]).long().argmax(1)
                loss = loss + .1 * F.cross_entropy(aux_logits, aux_y)
            else:
                loss = loss + .001 * self.film.regularization_term()
        return p, loss


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def pretrain(enc, tasks, steps=200):
    """Shared encoder pretraining with a separate supervised head per allowed district."""
    heads = nn.ModuleList([nn.Linear(DIM, t.S).to(DEVICE) for t in tasks])
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=.003, weight_decay=1e-4)
    total = sum(len(t.group_queries) for t in tasks)
    weights = []
    for t in tasks:
        _, inv, counts = np.unique(t.groups[t.qs], return_inverse=True, return_counts=True)
        weights.append(torch.tensor(1 / counts[inv] / total, device=DEVICE, dtype=torch.float32))
    enc.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = sum((F.cross_entropy(head(enc(t.x[t.qs])), t.y, reduction='none') * w).sum()
                   for t, head, w in zip(tasks, heads, weights))
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(model, task, qs, chunk=4):
    model.eval()
    out = np.zeros((len(qs), task.S), np.float32)
    # Reuse a context only when the physically retained support set is identical.
    refgroups = set(task.groups[task.ri])
    buckets = {}
    for i, q in enumerate(qs):
        key = task.groups[q] if task.groups[q] in refgroups else None
        buckets.setdefault(key, []).append((i, q))
    for pairs in buckets.values():
        sx, sy = task.support(pairs[0][1])
        size = chunk if model.method == 'egnn' else 256
        for j in range(0, len(pairs), size):
            part = pairs[j:j+size]
            p, _ = model(sx, sy, task.x[[q for _, q in part]], task.S)
            out[[i for i, _ in part]] = p.cpu().numpy()
    assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-5)
    return out


def fit(method, tasks, seed, lr, steps, val=None, check_steps=(80, 160, 240), log=None):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = ReferenceModel(method).to(DEVICE)
    if method in ('cnaps', 'feat'):
        pretrain(model.enc, tasks)
    if method == 'cnaps':
        model.enc.requires_grad_(False)
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad and id(p) in enc_ids], 'lr': lr * (.1 if method == 'feat' else 1.)},
        {'params': [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids], 'lr': lr}], weight_decay=1e-4)
    groups = [(ti, qs) for ti, t in enumerate(tasks) for qs in t.group_queries]
    best = None
    history = []
    for step in range(1, steps+1):
        model.train()
        opt.zero_grad()
        losses = []
        # Four independently sampled property groups per update; all groups equally likely.
        for gi in rng.integers(len(groups), size=4):
            ti, qs = groups[gi]
            t = tasks[ti]
            q = int(rng.choice(qs))
            sx, sy = t.support(q)
            _, loss = model(sx, sy, t.x[q:q+1], t.S, torch.tensor([t.labels[q]], device=DEVICE))
            (loss / 4).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step % 40 == 0 and log: log(dict(step=step, loss=float(np.mean(losses))))
        if val is not None and step in check_steps:
            p = predict(model, val, val.qs)
            y = val.y.cpu().numpy()
            acc = float((p.argmax(1) == y).mean())
            nll = float(-np.log(p[np.arange(len(y)), y].clip(1e-20)).mean())
            rec = dict(step=step, acc=acc, nll=nll)
            history.append(rec)
            if log: log(dict(validation=rec))
            if best is None or (round(acc, 9), -nll) > (round(best['acc'], 9), -best['nll']):
                best = dict(**rec, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best is not None: model.load_state_dict(best['state'])
    return model, dict(steps=best['step'] if best else steps, history=history,
                       val_acc=best['acc'] if best else None, val_nll=best['nll'] if best else None)
