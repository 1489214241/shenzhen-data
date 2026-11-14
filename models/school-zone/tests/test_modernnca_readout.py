import copy
import unittest
import torch
from test_full_evidence import toy
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.modernnca_readout import ReadoutModernNCA


class ReadoutTests(unittest.TestCase):
    def test_identity_initialization_and_rng(self):
        t, p, a = toy()
        base = LayerNormModernNCA(p, a, 7).eval()
        rng = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all()
        for kind in ['identity', 'summary', 'content']:
            m = ReadoutModernNCA(p, a, 7, kind).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())))
            for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,m.state_dict()[k]))
            self.assertTrue(torch.equal(base(t), m(t, bypass=True)))
            self.assertTrue(torch.allclose(base(t), m(t), atol=1e-7))

    def test_active_masks_permutations_gradients(self):
        for kind in ['summary','content']:
            t,p,a = toy()
            m = ReadoutModernNCA(p,a,7,kind).eval()
            torch.nn.init.normal_(m.head[-1].weight,std=.04)
            out = m(t)
            c = copy.deepcopy(t); c['px'][:,0] = float('nan'); c['y'] = t['y'].flip(0)
            self.assertTrue(torch.equal(out, m(c)))
            perm = torch.tensor([4,2,0,3,1])
            c = dict(t,px=t['px'][:,perm],valid=t['valid'][:,perm],gb=t['gb'][perm],mem=torch.cat([t['mem'][:5][perm],t['mem'][5:]]))
            self.assertTrue(torch.allclose(out,m(c),atol=3e-6))
            perm = torch.tensor([3,1,0,2])
            c = dict(t,ax=t['ax'][:,perm],mem=torch.cat([t['mem'][:5],t['mem'][5:][perm]])[:,perm])
            self.assertTrue(torch.allclose(out[:,perm],m(c),atol=3e-6))
            single = {k:(v[:1] if k in ['px','ax','valid','y'] else v) for k,v in t.items()}
            self.assertTrue(torch.allclose(out[:1],m(single),atol=3e-6))
            t['px'].requires_grad_(); t['ax'].requires_grad_()
            m(t)[:,1].log().sum().backward()
            self.assertEqual(float(t['px'].grad[:,0].abs().sum()),0.)
            self.assertTrue((t['px'].grad[:,1:].abs().sum((0,1))>0).all())
            self.assertTrue((t['ax'].grad.abs().sum((0,1))>0).all())
            for n in [0,1]:
                c = dict(t,px=t['px'][:,:n],valid=torch.ones_like(t['valid'][:,:n]),gb=t['gb'][:n]*0,mem=torch.cat([t['mem'][:n],t['mem'][5:]]))
                m.zero_grad(); pr=m(c);pr[:,0].log().sum().backward()
                self.assertTrue(torch.isfinite(pr).all())
                self.assertTrue(torch.allclose(pr.sum(-1),torch.ones(3),atol=1e-6))
                self.assertTrue((pr[:,3]>0).all())
                for v in m.parameters(): self.assertTrue(v.grad is None or torch.isfinite(v.grad).all())
            self.assertTrue(torch.isfinite(m(dict(t,valid=torch.zeros_like(t['valid'])))).all())

    def test_content_path_and_pool(self):
        t,p,a=toy();m=ReadoutModernNCA(p,a,7,'content').eval()
        parts=m.components(t);f=m.school_features(t,parts)
        c=copy.deepcopy(t);c['px'][:,2,3]+=2
        changed=m.school_features(c,parts)  # fixed votes and embeddings, change raw content only
        common=a+6+7
        self.assertTrue(torch.equal(f[:,:,:common],changed[:,:,:common]))
        self.assertGreater(float((f-changed).abs().sum()),0.)
        # School 3 has no references: new mean/variance/embedding summaries are zero.
        self.assertEqual(float(f[:,3,common:].abs().sum()),0.)
        # Duplicate a fixed content value and split its weight: same pooled moments.
        v=torch.tensor([[[1.,2.],[3.,8.]]],dtype=torch.double,requires_grad=True)
        w=torch.tensor([[.2,.8]],dtype=torch.double)
        mem=torch.ones(2,1,dtype=torch.double)
        expected=m.pool(v,w,mem,w.sum(1,keepdim=True))
        actual=m.pool(v[:,[0,1,1]],torch.tensor([[.2,.4,.4]],dtype=torch.double),torch.ones(3,1,dtype=torch.double),w.sum(1,keepdim=True))
        self.assertTrue(torch.allclose(expected,actual,atol=1e-12))
        self.assertTrue(torch.autograd.gradcheck(lambda z:m.pool(z,w,mem,w.sum(1,keepdim=True)),(v,)))


if __name__ == '__main__': unittest.main()
