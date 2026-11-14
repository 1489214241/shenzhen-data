"""Independent small numerical checks, no district/answer data."""
import json
import sys
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from zoning.modernnca_support import SupportModernNCA
from run_modernnca_relation import save,digest


def main():
    torch.manual_seed(42)
    e=torch.randn(1,8,4,dtype=torch.double,requires_grad=True)
    q=torch.randn(1,4,dtype=torch.double,requires_grad=True)
    w=torch.tensor([[.5,.5,1.,1.,1.,1.,1.,1.]],dtype=torch.double)
    valid=torch.ones(1,8,dtype=torch.bool)
    mem=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,0,1,2]),3).double()
    t=dict(px=torch.empty(1,5,1),mem=mem)
    m=SupportModernNCA(1,1,7,'covariance',dim=4).double().eval()
    actual=m.readout(e,q,w,valid,t)
    # Independently form weighted covariance and directly solve the metric.
    ref=e[0,:5]; rw=w[0,:5]; n=rw.sum(); center=(ref*rw[:,None]).sum(0)/n
    z=ref-center; C=sum(rw[i]*torch.outer(z[i],z[i]) for i in range(5))/n
    alpha=.5*n/(n+1); Q=(1-alpha)*torch.eye(4,dtype=torch.double)+alpha*C
    delta=q[0]-e[0]
    distances=torch.stack([(d@torch.linalg.solve(Q,d)).sqrt() for d in delta])
    expected=(-distances+w[0].log()).softmax(0)@mem
    error=float((expected-actual[0]).abs().max().detach())
    assert error<1e-12
    gradient=torch.autograd.gradcheck(lambda ee,qq:m.readout(ee,qq,w,valid,t),(e,q),eps=1e-6,atol=1e-5)
    assert gradient
    result=dict(direct_solve_probability_max_error=error,full_covariance_gradient_check=gradient,
        source_sha256=digest(REPO/'zoning/modernnca_support.py'),audit_sha256=digest(Path(__file__)))
    save(REPO.parent/'科研推进/ModernNCA参照条件化尝试_20261113/协方差数值核对.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
