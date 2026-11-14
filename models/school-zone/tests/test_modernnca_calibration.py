import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from test_full_evidence import toy
from zoning.full_evidence import FullEvidenceModel, tokens
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_modernnca_normalization as audit
from run_modernnca_relation import ROW_KEYS


class CalibrationTests(unittest.TestCase):
    def test_exact_moments_order_masks_labels_parameters(self):
        t,p,a=toy(); t['ids']=np.arange(3)
        u=copy.deepcopy(t); u['px']+=.5; u['ax']-=.3
        tr=[('a',t),('b',u)]
        original=FullEvidenceModel('modernnca',p,a,7).eval()
        forward=copy.deepcopy(original); reverse=copy.deepcopy(original)
        old=audit.device_batch
        audit.device_batch=lambda task,ix:{k:(v[ix] if k in ROW_KEYS else v) for k,v in task.items() if torch.is_tensor(v)}
        try:
            audit.pooled_bn(forward,tr); audit.pooled_bn(reverse,tr[::-1])
            for name,param in original.named_parameters():
                self.assertTrue(torch.equal(param,dict(forward.named_parameters())[name]))
            for name,value in forward.state_dict().items():
                self.assertTrue(torch.allclose(value,reverse.state_dict()[name],atol=1e-6))
            z=[]
            for _,task in tr:
                x,valid=tokens(task,p); z.append(original.encoder.encoder(x[valid]))
            z=torch.cat(z).double()
            bn=forward.encoder.post_encoder[0].block[0]
            self.assertTrue(torch.allclose(bn.running_mean,z.mean(0).float(),atol=1e-6))
            self.assertTrue(torch.allclose(bn.running_var,z.var(0,unbiased=False).float(),atol=1e-6))
            perturbed=copy.deepcopy(tr)
            for _,task in perturbed:
                task['y']=task['y'].flip(0); task['px'][:,0]=1e6
            changed=copy.deepcopy(original); audit.pooled_bn(changed,perturbed)
            for name,value in forward.state_dict().items():
                self.assertTrue(torch.equal(value,changed.state_dict()[name]))
        finally:
            audit.device_batch=old


if __name__=='__main__': unittest.main()
