"""Additional mechanism tests independent of real evaluation answers."""
import sys
import json
from types import SimpleNamespace
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from zoning.reference_trials import ReferenceModel, seed_all, BatchedClassifier, LinearClassifierAdaptationNetwork, independent_class_attention

seed_all(88)
model = ReferenceModel('feat').cuda()
# Disable dropout in both copies to compare deterministic training objectives.
model.attn.dropout.p = 0
model.attn.attention.dropout.p = 0
src = (REPO.parent/'tmp/reference_broad_20261111/code/FEAT/model__models__feat.py').read_text(encoding='utf-8')
ns = dict(torch=torch, nn=nn, np=np, F=F, FewShotModel=nn.Module)
exec(src[src.index('class ScaledDotProductAttention'):], ns)
official = ns['FEAT'].__new__(ns['FEAT'])
nn.Module.__init__(official)
official.args = SimpleNamespace(way=3, shot=2, query=1, use_euclidean=True, temperature=1., temperature2=1.)
official.slf_attn = model.attn
sx, qx = torch.randn(6,2,device='cuda'), torch.randn(3,2,device='cuda')
sy, y = torch.tensor([0,1,2,0,1,2],device='cuda'), torch.arange(3,device='cuda')
model.train(); official.train()
p, loss = model(sx, sy, qx, 3, y)
h = model.enc(torch.cat((sx,qx)))
logits, aux_logits = official._forward(h, torch.arange(6,device='cuda').view(1,2,3),
                                    torch.arange(6,9,device='cuda').view(1,1,3))
official_loss = F.cross_entropy(logits,y) + .1*F.cross_entropy(aux_logits,torch.arange(3,device='cuda').repeat(3))
pd = float((p-logits.softmax(-1)).abs().max().detach())
ld = float((loss-official_loss).abs().detach())
assert pd < 2e-6 and ld < 2e-5, (pd,ld)
result = dict(feat_official_balanced_episode=dict(probability_maxdiff=pd,total_loss_absdiff=ld))

# Check vectorized class processors against official independent class calls.
a=BatchedClassifier(32).cuda(); b=LinearClassifierAdaptationNetwork(32).cuda()
b.load_state_dict(a.state_dict())
h=torch.randn(7,32,device='cuda',requires_grad=True); hh=h.detach().clone().requires_grad_(True)
aa=a(dict(enumerate(h.split(1)))); bb=b(dict(enumerate(hh.split(1))))
diff=max(float((aa[k]-bb[k]).abs().max().detach()) for k in aa)
sum(x.square().sum() for x in aa.values()).backward();sum(x.square().sum() for x in bb.values()).backward()
grad=float((h.grad-hh.grad).abs().max())
assert diff<2e-6 and grad<2e-5, (diff,grad)
result['cnaps_batched_official_parity']=dict(output_maxdiff=diff,input_gradient_maxdiff=grad)

# Compare unequal-shot FEAT auxiliary attention and gradients with a literal official per-class call.
att=model.attn
h=torch.randn(17,32,device='cuda',requires_grad=True); hh=h.detach().clone().requires_grad_(True)
y=torch.tensor([0]*2+[1]*4+[2]*11,device='cuda')
fast=independent_class_attention(att,h,y)
slow=torch.cat([att(hh[y==c][None],hh[y==c][None],hh[y==c][None])[0] for c in y.unique()])
diff=float((fast-slow).abs().max().detach())
w=torch.randn_like(fast)
(fast*w).sum().backward();(slow*w).sum().backward()
grad=float((h.grad-hh.grad).abs().max())
assert diff<2e-6 and grad<2e-5,(diff,grad)
result['feat_unequal_class_batching_parity']=dict(output_maxdiff=diff,input_gradient_maxdiff=grad)

# Each full adapted model must be able to learn a deliberately easy spatial support task.
for name in ['egnn','cnaps','feat']:
    seed_all(123)
    m=ReferenceModel(name).cuda()
    centers=torch.tensor([[-1.,0.],[0.,1.],[1.,0.]],device='cuda')
    sx=centers.repeat_interleave(4,0)+.04*torch.randn(12,2,device='cuda')
    sy=torch.arange(3,device='cuda').repeat_interleave(4)
    qx=centers+.02*torch.randn(3,2,device='cuda')
    y=torch.arange(3,device='cuda')
    if name=='cnaps':
        head=nn.Linear(32,3).cuda()
        opt=torch.optim.Adam(list(m.enc.parameters())+list(head.parameters()),lr=.01)
        for _ in range(100):
            opt.zero_grad(); F.cross_entropy(head(m.enc(sx)),sy).backward(); opt.step()
        m.enc.requires_grad_(False)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003)
    losses=[]
    for step in range(120):
        m.train(); opt.zero_grad()
        p,l=m(sx,sy,qx,3,y); l.backward(); opt.step()
        losses.append(float(l.detach()))
    m.eval()
    with torch.no_grad(): p,_=m(sx,sy,qx,3)
    acc=float((p.argmax(1)==y).float().mean())
    result[name+'_synthetic_fit']=dict(accuracy=acc,initial_loss=losses[0],final_loss=losses[-1])
    assert acc==1., (name,acc)
path=REPO/'outputs/reference_three_changsha_20261111/additional_tests.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
