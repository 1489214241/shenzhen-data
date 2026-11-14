import unittest
import torch
from zoning.modernnca_numeric import NumericModernNCA
from zoning.modernnca_normalization import LayerNormModernNCA
from test_full_evidence import toy


class NumericTests(unittest.TestCase):
    def build(self,kind):
        t,p,a=toy();cols=[0,1,2,4,p,p+1,p+a,p+a+6]
        mean=torch.linspace(-1,1,len(cols));std=torch.linspace(.5,2,len(cols))
        return t,NumericModernNCA(p,a,7,cols,mean,std,kind).eval()

    def test_raw_and_standard_initial_function(self):
        t,p,a=toy();original=LayerNormModernNCA(p,a,7).eval()
        for kind in ['raw','standard']:
            _,m=self.build(kind)
            self.assertTrue(torch.allclose(m(t),original(t),atol=2e-6))
            self.assertTrue(torch.allclose(m.regularization(.01),original.regularization(.01),atol=1e-6))

    def test_interface_and_masks(self):
        for kind in ['standard','lr','plr']:
            t,m=self.build(kind);out=m(t)
            self.assertTrue(torch.isfinite(out).all());self.assertTrue(torch.allclose(out.sum(1),torch.ones(3)))
            self.assertTrue((out[:,3]>0).all())
            self.assertTrue(torch.equal(out,m(dict(t,y=t['y'].flip(0)))))
            changed=dict(t,px=t['px'].clone());changed['px'][:,0]=1e8
            self.assertTrue(torch.equal(out,m(changed)))
            perm=torch.tensor([4,2,0,3,1])
            changed=dict(t,px=t['px'][:,perm],valid=t['valid'][:,perm],gb=t['gb'][perm],
                mem=torch.cat([t['mem'][:5][perm],t['mem'][5:]]))
            self.assertTrue(torch.allclose(out,m(changed),atol=2e-6))
            perm=torch.tensor([3,1,0,2])
            changed=dict(t,ax=t['ax'][:,perm],mem=torch.cat([t['mem'][:5],t['mem'][5:][perm]])[:,perm])
            self.assertTrue(torch.allclose(out[:,perm],m(changed),atol=2e-6))
            changed=dict(t,valid=torch.zeros_like(t['valid']))
            self.assertTrue(torch.isfinite(m(changed)).all())
            t['px'].requires_grad_();t['ax'].requires_grad_()
            m(t)[:,1].log().sum().backward()
            self.assertTrue((t['px'].grad[:,1:].abs().sum((0,1))>0).all())
            self.assertTrue((t['ax'].grad.abs().sum((0,1))>0).all())
            self.assertEqual(t['px'].grad[:,0].abs().sum().item(),0.)

    def test_official_periodic_formula(self):
        t,m=self.build('plr');mod=m.encoder.encoder
        x=torch.randn(11,len(mod.cols));freq=mod.numeric[0].frequencies
        theta=2*torch.pi*x[:,:,None]*freq[None]
        expected=mod.numeric[2](mod.numeric[1](torch.cat([theta.cos(),theta.sin()],-1)))
        self.assertTrue(torch.allclose(mod.numeric(x),expected,atol=1e-6,rtol=1e-6))


if __name__=='__main__':unittest.main()
