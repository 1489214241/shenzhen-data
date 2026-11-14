import unittest
from types import SimpleNamespace
import torch
from zoning.modernnca_provenance import ProvenanceModernNCA,ProvenanceRetrieval
from zoning.modernnca_normalization import LayerNormModernNCA
from zoning.model import RetrievalScorer
from test_full_evidence import toy

def fixture():
    t,p,a=toy();t['pair']=t['px'][:,:,:p].clone();t['anchor']=t['ax'][:,:,:a].clone()
    t['source_type']=torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    start=SimpleNamespace(p=torch.zeros(p),a=torch.zeros(a),offset=torch.tensor(-2.))
    return t,p,a,start

class ProvenanceTests(unittest.TestCase):
    def models(self,kind):
        t,p,a,start=fixture()
        if kind=='ln':old=LayerNormModernNCA(p,a,17);state=torch.get_rng_state();new=ProvenanceModernNCA(p,a,17)
        else:old=RetrievalScorer(p,a,start,17);state=torch.get_rng_state();new=ProvenanceRetrieval(p,a,start,17)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        return t,old,new

    def test_initial_function_and_penalty(self):
        for kind in ['ln','reference']:
            t,old,new=self.models(kind);old.eval();new.eval()
            self.assertTrue(torch.equal(old(t),new(t)))
            self.assertTrue(torch.equal(old.regularization(.01),new.regularization(.01)))

    def test_masks_permutations_labels(self):
        for kind in ['ln','reference']:
            t,old,m=self.models(kind);m.eval()
            with torch.no_grad():m.source_projection.weight.copy_(torch.linspace(-.2,.3,m.source_projection.weight.numel()).reshape_as(m.source_projection.weight))
            if kind=='reference':
                with torch.no_grad():m.source_linear.fill_(.2)
            P=m(t);self.assertTrue(torch.allclose(P.sum(1),torch.ones(3)))
            old.eval();self.assertFalse(torch.allclose(P,old(t),atol=1e-7))
            self.assertTrue(torch.equal(P,m(dict(t,y=t['y'].flip(0)))))
            source=t['source_type'].clone();source[0]=1e7
            self.assertTrue(torch.equal(P,m(dict(t,source_type=source))))
            perm=torch.tensor([4,2,0,3,1]);u=dict(t,source_type=t['source_type'][perm],px=t['px'][:,perm],pair=t['pair'][:,perm],valid=t['valid'][:,perm],gb=t['gb'][perm],mem=torch.cat([t['mem'][:5][perm],t['mem'][5:]]))
            self.assertTrue(torch.allclose(P,m(u),atol=2e-6))
            self.assertTrue(torch.isfinite(m(dict(t,valid=torch.zeros_like(t['valid'])))).all())
            t['source_type'].requires_grad_();m(t)[:,1].log().sum().backward()
            self.assertEqual(float(t['source_type'].grad[0].abs().sum()),0.)
            self.assertGreater(float(t['source_type'].grad[1:].abs().sum()),0.)

    def test_zero_source_training_is_original_training(self):
        for kind in ['ln','reference']:
            t,old,new=self.models(kind);t['source_type'].zero_()
            oo=torch.optim.Adam(old.parameters(),lr=.003);on=torch.optim.Adam(new.parameters(),lr=.003)
            for step in range(3):
                for model,opt in [(old,oo),(new,on)]:
                    torch.manual_seed(100+step);model.train();opt.zero_grad();P=model(t)
                    loss=-P[torch.arange(3),t['y']].log().mean()+model.regularization(.01);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            old.eval();new.eval();self.assertTrue(torch.allclose(old(t),new(t),atol=2e-6))
            self.assertEqual(float(new.source_projection.weight.abs().sum()),0.)

if __name__=='__main__':unittest.main()
