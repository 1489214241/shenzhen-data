import copy
import importlib.util
from pathlib import Path
import unittest
import torch
from test_full_evidence import toy
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.modernnca_interaction import InteractionModernNCA, WeightedMAB


class InteractionTests(unittest.TestCase):
    def test_initial_identity_and_rng(self):
        t,p,a = toy()
        base = LayerNormModernNCA(p,a,7).eval()
        state = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all()
        for kind in ['identity','rowmlp','statistics','isab']:
            m = InteractionModernNCA(p,a,7,kind).eval()
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())))
            for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,m.state_dict()[k]))
            self.assertTrue(torch.equal(base(t),m(t)))

    def test_active_contract_and_gradients(self):
        for kind in ['rowmlp','statistics','isab']:
            t,p,a = toy();m = InteractionModernNCA(p,a,7,kind).eval()
            # Exercise active adapters, not only the trivial zero-residual path.
            torch.nn.init.normal_(m.output.weight,std=.04)
            out=m(t)
            self.assertTrue((out[:,3]>0).all())
            self.assertTrue(torch.allclose(out.sum(-1),torch.ones(3),atol=1e-6))
            c=copy.deepcopy(t);c['px'][:,0]=1e20;c['y']=t['y'].flip(0)
            self.assertTrue(torch.equal(out,m(c)))
            perm=torch.tensor([4,2,0,3,1])
            c=dict(t,px=t['px'][:,perm],valid=t['valid'][:,perm],gb=t['gb'][perm],mem=torch.cat([t['mem'][:5][perm],t['mem'][5:]]))
            self.assertTrue(torch.allclose(out,m(c),atol=3e-6))
            perm=torch.tensor([3,1,0,2])
            c=dict(t,ax=t['ax'][:,perm],mem=torch.cat([t['mem'][:5],t['mem'][5:][perm]])[:,perm])
            self.assertTrue(torch.allclose(out[:,perm],m(c),atol=3e-6))
            single={k:(v[:1] if k in ['px','ax','valid','y'] else v) for k,v in t.items()}
            self.assertTrue(torch.allclose(out[:1],m(single),atol=3e-6))
            t['px'].requires_grad_();t['ax'].requires_grad_()
            m(t)[:,1].log().sum().backward()
            self.assertEqual(float(t['px'].grad[:,0].abs().sum()),0.)
            self.assertTrue((t['px'].grad[:,1:].abs().sum((0,1))>0).all())
            self.assertTrue((t['ax'].grad.abs().sum((0,1))>0).all())
            for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            for n in [0,1]:
                c=dict(t,px=t['px'][:,:n],valid=torch.ones_like(t['valid'][:,:n]),gb=t['gb'][:n]*0,mem=torch.cat([t['mem'][:n],t['mem'][5:]]))
                m.zero_grad();pr=m(c);pr[:,0].log().sum().backward()
                self.assertTrue(torch.isfinite(pr).all())
                for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            self.assertTrue(torch.isfinite(m(dict(t,valid=torch.zeros_like(t['valid'])))).all())

    def test_official_mab_and_mass(self):
        source=Path(__file__).resolve().parents[2]/'科研推进/参照交互与不均衡文献续查_20261113/code_sources/set_transformer/modules.py'
        spec=importlib.util.spec_from_file_location('official',source)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        torch.manual_seed(6)
        official=mod.MAB(8,8,8,2,ln=True).double()
        local=WeightedMAB(8,2).double();local.load_state_dict(official.state_dict())
        q=torch.randn(2,3,8,dtype=torch.double,requires_grad=True)
        k=torch.randn(2,5,8,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.equal(official(q,k),local(q,k)))
        weight=torch.ones(2,5,dtype=torch.double);weight[:,0]=0
        expected=local(q,k[:,1:]);actual=local(q,k,weight)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12))
        ix=torch.tensor([0,1,1,2,3,4]);ww=weight[:,ix].clone();ww[:,1:3]/=2
        self.assertTrue(torch.allclose(actual,local(q,k[:,ix],ww),atol=1e-12))
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:local(a,b,weight),(q,k)))


if __name__=='__main__': unittest.main()
