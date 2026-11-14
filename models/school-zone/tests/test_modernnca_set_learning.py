import unittest
import torch
from zoning.modernnca_set_learning import LearnedSetReadout,complete_set_targets,training_loss,evidence_latents
from zoning.modernnca_multilabel import MultiLabelModernNCA


class SetLearningTests(unittest.TestCase):
    def features(self):
        torch.manual_seed(8);f=torch.randn(2,3,99);f[:,:,96]=torch.tensor([.5,.3,.2]).log();return f

    def test_zero_head_is_frozen_geometric_decoder(self):
        f=self.features();m=LearnedSetReadout(2);mem=torch.tensor([[1.,0,0],[1,1,0]])
        expected=torch.stack([f[:,0,96],f[:,:2,96].mean(1)+torch.tensor(2.).log()],1)
        torch.testing.assert_close(m(f,mem),expected)

    def test_school_and_candidate_permutation(self):
        f=self.features();m=LearnedSetReadout(2);torch.nn.init.normal_(m.joint[-1].weight)
        mem=torch.tensor([[1.,0,0],[1,1,0]])
        torch.testing.assert_close(m(f[:,[2,0,1]],mem[:,[2,0,1]]),m(f,mem))
        torch.testing.assert_close(m(f,mem[[1,0]]),m(f,mem)[:,[1,0]])

    def test_partial_labels_rejected(self):
        with self.assertRaises(ValueError):complete_set_targets(torch.tensor([[1.,-1]]),torch.eye(2))

    def test_complete_set_loss_trains_joint_target(self):
        mem=torch.tensor([[1.,0],[0,1],[1,1]])
        y=complete_set_targets(torch.tensor([[1.,1.]]),mem);z=torch.zeros(1,3,requires_grad=True)
        training_loss(z,y,torch.tensor([[.5,.5]]),[0,1],torch.tensor([False]),torch.ones(1)).backward()
        self.assertLess(z.grad[0,2],0);self.assertTrue((z.grad[0,:2]>0).all())

    def test_evidence_cache_matches_teacher_and_excludes_self_reference(self):
        torch.manual_seed(5);m=MultiLabelModernNCA(41,27,3).eval()
        t=dict(px=torch.randn(2,2,74),ax=torch.randn(2,3,33),gb=torch.zeros(2),
            valid=torch.tensor([[False,True],[False,True]]),mem=torch.tensor([[1.,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]))
        f,p=evidence_latents(m,t)
        with torch.no_grad():torch.testing.assert_close(p,m(t).softmax(1),atol=2e-6,rtol=2e-6)
        t['px'][:,0]=1e6;f2,p2=evidence_latents(m,t)
        torch.testing.assert_close(f,f2);torch.testing.assert_close(p,p2)


if __name__=='__main__':unittest.main()
