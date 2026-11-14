"""Mechanism parity and reference isolation checks, no real query labels."""
import sys
import json
import types
from pathlib import Path
from collections import OrderedDict
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from zoning.reference_trials import ReferenceModel, DistrictTask, predict, GraphNetwork, seed_all


def main():
    seed_all(17)
    results = {}
    # Large wrong-class margins must retain a finite, nonzero gradient.
    class ExtremeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1000., -1000.], device='cuda'))
        def forward(self, reps):
            return dict(weight_mean=torch.zeros(2, 32, device='cuda'), bias_mean=self.bias)
    extreme = ReferenceModel('cnaps').cuda()
    extreme.classifier = ExtremeClassifier()
    _, stable_ce = extreme(torch.randn(2,2,device='cuda'), torch.tensor([0,1],device='cuda'),
                           torch.randn(1,2,device='cuda'), 2, torch.tensor([1],device='cuda'))
    stable_ce.backward()
    assert stable_ce.item() >= 2000 and extreme.classifier.bias.grad.abs().sum().item() == 2
    results['stable_cross_entropy_extreme_margin'] = 'PASS'
    # Compare extracted graph forward/backward with the unmodified official class body.
    source = (REPO.parent / 'tmp/baseline_unrestricted_20261111/root/EGNN_official_model.py').read_text(encoding='utf-8')
    ns = dict(torch=torch, nn=nn, F=F, OrderedDict=OrderedDict,
              tt=types.SimpleNamespace(arg=types.SimpleNamespace(device='cuda')))
    exec(source[source.index('class NodeUpdateNetwork'):], ns)
    a = GraphNetwork(8, 8, 4, 3).cuda().eval()
    b = ns['GraphNetwork'](8, 8, 4, 3).cuda().eval()
    b.load_state_dict(a.state_dict())
    h = torch.randn(2, 9, 8, device='cuda', requires_grad=True)
    h2 = h.detach().clone().requires_grad_(True)
    e = torch.rand(2, 2, 9, 9, device='cuda'); e = e / e.sum(1, keepdim=True)
    aa, bb = a(h, e), b(h2, e)
    diff = max(float((x-y).abs().max()) for x,y in zip(aa, bb))
    aa[-1][:, 0].square().sum().backward(); bb[-1][:, 0].square().sum().backward()
    gd = float((h.grad-h2.grad).abs().max())
    assert diff == 0 and gd == 0
    results['egnn_official_forward_and_gradient_maxdiff'] = [diff, gd]
    # FEAT and CNAPs core code is extracted verbatim. Check the composed pipelines' invariants.
    dist = types.SimpleNamespace(name='synthetic', school_n=5,
        coords=np.array([[0,0],[5,0],[10,10],[20,20],[30,0],[40,0]], float)*100,
        groups=np.array(['A','A','B','C','D','E']), not_reference=set())
    rules = types.SimpleNamespace(known={0:[0],1:[0],2:[1],3:[1],4:[2]})
    task = DistrictTask(dist, rules)
    sx, sy = task.support(0)
    assert len(sy) == 3 and 0 not in sy.tolist()  # entire A removed
    perm = torch.tensor([2,4,0,1,3], device='cuda')
    for method in ['egnn', 'cnaps', 'feat']:
        model = ReferenceModel(method).cuda().eval()
        sx, sy = task.support(5)
        q = task.x[5:6]
        with torch.no_grad():
            p, _ = model(sx, sy, q, 5)
            pp, _ = model(sx, perm[sy], q, 5)
            ix = torch.tensor([4,0,3,1,2], device='cuda')
            po, _ = model(sx[ix], sy[ix], q, 5)
            p0, _ = model(sx, sy, q, 5, torch.tensor([0], device='cuda'))
            p1, _ = model(sx, sy, q, 5, torch.tensor([1], device='cuda'))
            batch, _ = model(sx, sy, torch.cat((q, task.x[2:3])), 5)
            empty, _ = model(sx[:0], sy[:0], q, 5)
        assert torch.allclose(p, pp[:, perm], atol=2e-6)
        assert torch.allclose(p, po, atol=2e-6)
        assert torch.equal(p0, p1)  # supplied training query label never affects forward distribution
        assert torch.allclose(p, batch[:1], atol=2e-6)
        assert p[0, 3:].sum() == 0
        assert torch.allclose(empty, torch.ones_like(empty)/5)
        # Inference removes same-project feature AND label influence, not just final votes.
        before = predict(model, task, [0])
        task2 = DistrictTask(dist, types.SimpleNamespace(known={0:[4],1:[3],2:[1],3:[1],4:[2]}))
        task2.x[:2] = torch.tensor([[999.,999.],[-999.,-999.]], device='cuda')
        task2.x[0] = task.x[0]  # query feature held fixed; removed neighbor changed
        after = predict(model, task2, [0])
        assert np.allclose(before, after, atol=2e-6)
        model.train()
        _, loss = model(sx, sy, q, 5, torch.tensor([1], device='cuda'))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        results[method] = 'label permutation, support order, query independence, absent class, empty support, project isolation, finite backward: PASS'
    print(json.dumps(results, ensure_ascii=False, indent=2))
    out = REPO / 'outputs/reference_three_changsha_20261111'
    out.mkdir(exist_ok=True, parents=True)
    (out / 'tests.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
