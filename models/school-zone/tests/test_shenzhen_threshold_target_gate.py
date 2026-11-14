import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experiment_shenzhen_threshold_target as T


class GateTests(unittest.TestCase):
    def fixture(self):
        return [dict(selected_target_candidate='adaptive_head',all_adaptive_converged=True),
            dict(selected_target_candidate='adaptive_head'),dict(all_metrics_independently_recomputed=True,source_gate_passed=True),
            {'adaptive_head/1':dict(reason='gradient_converged',final=dict(gradient_inf=1e-7))},[1],1e-6]

    def test_each_gate_required(self):
        self.assertTrue(all(T.require_gate(*self.fixture()).values()))
        cases=[(0,'selected_target_candidate',None),(0,'all_adaptive_converged',False),
            (1,'selected_target_candidate',None),(2,'all_metrics_independently_recomputed',False),(2,'source_gate_passed',False)]
        for i,key,value in cases:
            args=self.fixture();args[i][key]=value
            with self.assertRaises(ValueError):T.require_gate(*args)
        args=self.fixture();args[3]['adaptive_head/1']['final']['gradient_inf']=1e-4
        with self.assertRaises(ValueError):T.require_gate(*args)

    def test_real_round35_rejected(self):
        out=T.R.ROOT/'round35_threshold_convergence'
        c=T.F.read(out/'protocol.json')
        with self.assertRaises(ValueError):T.require_gate(T.F.read(out/'admission_gate.json'),T.F.read(out/'source_report.json'),
            T.F.read(T.R.ROOT/'round35_independent_completion_audit.json'),T.F.read(out/'training_history.json'),c['seeds'],c['gradient_tolerance'])
        self.assertFalse(T.OUT.exists())

    def test_strict_zero_cut_and_existing_relative_cut(self):
        class PassThrough:
            def apply(self,x):return x
        p=np.array([[-1.,0.,np.nextafter(0.,np.inf),1.]])
        with patch.object(T,'decode',side_effect=lambda values,d,t:(values,t)):
            values,t=T.target_rows(p,'toy','adaptive_head',{},'balanced_exact',PassThrough())
            np.testing.assert_array_equal(values>=t,p>0)
            values,t=T.target_rows(np.array([[.2,.8]]),'toy','original_relative',{'balanced_exact':{'threshold':.7}},'balanced_exact',PassThrough())
            np.testing.assert_array_equal(values,[[.25,1.]])
            self.assertEqual(t,.7)


if __name__=='__main__':unittest.main()
