import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_local_reference_dropout import local_validity,masked_batch
from zoning.model import context


class LocalDropoutTests(unittest.TestCase):
    def test_whole_property_and_self_exclusion(self):
        valid=np.array([[1,1,1,0],[0,0,1,1]],bool)
        dist=np.array([[100,800,900,0],[0,0,501,500]],float)
        out=local_validity(valid,dist,np.array(['a','a','b','c']))
        np.testing.assert_array_equal(out,[[0,0,1,0],[0,0,1,0]])
        self.assertFalse((out&~valid).any())

    def test_permutation_and_empty(self):
        v=np.ones((2,4),bool);d=np.array([[100,900,800,200],[700,800,900,100]])
        g=np.array(['a','a','b','c']);p=np.array([3,1,0,2])
        np.testing.assert_array_equal(local_validity(v,d,g)[:,p],local_validity(v[:,p],d[:,p],g[p]))
        self.assertEqual(local_validity(np.zeros((2,0),bool),np.zeros((2,0)),np.array([])).shape,(2,0))

    def test_context_recomputed_and_other_evidence_preserved(self):
        pair=torch.zeros((1,3,41));pair[0,:,0]=torch.tensor([100.,800.,900.])/300
        anchor=torch.zeros((1,2,27));vt=torch.ones((1,3),dtype=torch.bool)
        gb=torch.zeros(3);mem=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
        rs=np.array([0,0,1]);meta=(0,[27,28,29,30,31],41)
        px,ax=context(pair,anchor,vt,gb,mem[:3],rs,*meta)
        t=dict(pair=pair,anchor=anchor,valid=vt,gb=gb,mem=mem,rs=torch.from_numpy(rs),meta=meta,px=px,ax=ax,
               labels=torch.tensor([[1.,-1.]]),sample_weight=torch.ones(1))
        result=masked_batch(t,np.array(['a','a','b']))
        np.testing.assert_array_equal(result['valid'],[[0,0,1]])
        for key in ['pair','anchor','gb','mem','labels','sample_weight']:
            self.assertIs(result[key],t[key])
        expected_px,expected_ax=context(pair,anchor,torch.tensor([[False,False,True]]),gb,mem[:3],rs,*meta)
        torch.testing.assert_close(result['px'],expected_px,rtol=0,atol=0)
        torch.testing.assert_close(result['ax'],expected_ax,rtol=0,atol=0)
        self.assertTrue(t['valid'].all())
        self.assertEqual(result['ax'][0,0,-1],0)
        self.assertEqual(result['ax'][0,1,-1],1)


if __name__=='__main__':unittest.main()
