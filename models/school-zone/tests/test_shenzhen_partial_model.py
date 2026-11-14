import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shenzhen_partial_model import school_log_mass,masked_bce,choose_threshold,predict_sets


def test_aggregation_matches_native_probability_and_permutation():
    a=torch.tensor([[1.,2.,-torch.inf,3.],[-2.,0.,1.,2.]])
    mem=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    out=school_log_mass(a,mem)
    torch.testing.assert_close(out.softmax(1),a.softmax(1)@mem)
    permutation=[2,0,3,1]
    torch.testing.assert_close(school_log_mass(a[:,permutation],mem[permutation]),out)
    assert (out[0].sigmoid()>.5).sum()==2


def test_unknown_has_no_output_gradient_and_negative_does():
    logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
    labels=torch.tensor([[1.,-1.,0.]])
    masked_bce(logits,labels,torch.tensor([1.])).backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[-.5,0.,.5]]))


def test_chunk_weights_preserve_objective_and_gradient():
    torch.manual_seed(1)
    a=torch.randn(7,3,requires_grad=True);labels=torch.tensor([[1.,0.,-1.]]*7)
    w=torch.arange(1,8).float();w=w/w.sum()
    whole=masked_bce(a,labels,w);whole.backward();g=a.grad.clone();a.grad=None
    parts=sum(masked_bce(a[i:i+2],labels[i:i+2],w[i:i+2]) for i in range(0,7,2))
    parts.backward();torch.testing.assert_close(parts,whole);torch.testing.assert_close(a.grad,g)


def test_assume_negative_is_explicit_separate_arm():
    a=torch.tensor([[0.,0.]],requires_grad=True)
    masked_bce(a,torch.tensor([[1.,-1.]]),torch.ones(1),assume_negative=True).backward()
    torch.testing.assert_close(a.grad,torch.tensor([[-.5,.5]]))


def test_threshold_uses_complete_set_and_keeps_at_least_one():
    p=[np.array([[.8,.7],[.3,.1]])];y=[np.array([[True,True],[True,False]])]
    threshold,rows=choose_threshold(p,y,[.5,.9])
    assert threshold==.5 and rows[0]['correct']==2


def test_university_not_multi_expanded_and_scopes_not_transitive():
    policy=dict(school_ids=['A','B','C','D'],year=2026,version='test',relations=[
        dict(kind='university',rule_id='U',school_ids=['A','B'])])
    graph=dict(edges=[dict(trigger_school_id='A',implied_school_id='C'),dict(trigger_school_id='C',implied_school_id='D')])
    r=predict_sets(np.array([.9,.8,.1,.1]),.5,['A','B','C','D'],policy,graph,
        dict(coordinate_group_id='q',building_name='未知苑'))
    assert r['predicted_school_ids']==['A','C']


def test_partial_positive_is_lower_bound_not_exhaustive_set():
    policy=dict(school_ids=['A','B'],year=2026,version='test',relations=[])
    r=predict_sets(np.array([.1,.8]),.5,['A','B'],policy,dict(edges=[]),
        dict(coordinate_group_id='q',building_name='甲苑'),positive_ids=['A'])
    assert r['predicted_school_ids']==['A','B']
