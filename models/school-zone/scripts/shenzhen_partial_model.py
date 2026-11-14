"""Native retrieval features and scores, independent school Bernoulli outputs.

This is a controlled masked-BCE adaptation, not a reproduction of ROLE.
No expected target label cardinality, target map labels, or fixed school IDs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF
from zoning import model as M


def school_log_mass(candidate_logits, membership):
    """Same candidate aggregation as native softmax, before across-school normalization."""
    offset=candidate_logits.max(dim=1,keepdim=True).values
    mass=torch.exp(candidate_logits-offset) @ membership
    return mass.clamp_min(1e-30).log()+offset


class BernoulliRetrieval(M.RetrievalScorer):
    def __init__(self, np_, na, start, seed, width=32):
        super().__init__(np_,na,start,seed,width)
        self.binary_bias=nn.Parameter(torch.tensor(0.))

    def forward(self,t):
        assert not self.cond
        pl=t['pair']@self.p+t['gb'][None]+t['px'][:,:,self.np:]@self.pc+self.pnet(t['px']).squeeze(-1)
        al=t['anchor']@self.a+self.offset+t['ax'][:,:,self.na:]@self.ac+self.anet(t['ax']).squeeze(-1)
        pl=pl.masked_fill(~t['valid'],-torch.inf)
        al=M._school_mask(self,al,t['valid'])
        return school_log_mass(torch.cat([pl,al],1),t['mem'])+self.binary_bias

    def regularization(self,reg):
        return super().regularization(reg)+reg*self.binary_bias.square()


def masked_bce(logits,labels,weight,assume_negative=False):
    """labels: -1 unknown, 0 observed negative, 1 observed positive.

    Sum over classes, weighted average over property groups. Unknowns contribute
    no loss in the primary arm. The AN comparator intentionally assumes negatives.
    """
    if logits.shape!=labels.shape:raise ValueError('Label shape mismatch')
    if not torch.all((labels==-1)|(labels==0)|(labels==1)):raise ValueError('Invalid label state')
    mask=torch.ones_like(labels,dtype=torch.bool) if assume_negative else labels>=0
    loss=TF.binary_cross_entropy_with_logits(logits,labels.clamp_min(0),reduction='none')
    return (loss.masked_fill(~mask,0).sum(1)*weight).sum()


def choose_threshold(probabilities,labels,grid):
    """Source validation only; exact set, then fewer FP, then higher threshold."""
    rows=[]
    for threshold in grid:
        correct=fp=fn=n=0
        for p,y in zip(probabilities,labels):
            pred=p>=threshold
            # Preserve the existing always-at-least-one-school decision contract.
            empty=~pred.any(1);pred[empty,p[empty].argmax(1)]=True
            correct+=int((pred==y).all(1).sum());fp+=int((pred&~y).sum());fn+=int((~pred&y).sum());n+=len(y)
        rows.append(dict(threshold=float(threshold),n=n,correct=correct,fp=fp,fn=fn))
    if not rows or not rows[0]['n']:raise ValueError('Empty validation population')
    best=max(rows,key=lambda r:(r['correct'],-r['fp'],r['threshold']))
    return best['threshold'],rows


def predict_sets(probabilities,threshold,school_ids,policy,graph,site,positive_ids=()):
    """Multiple base relations; no university member expansion or transitive closure."""
    from zoning import shenzhen_policy as SP
    from shenzhen_scope_implication import expand_original
    selected={school_ids[j] for j in np.flatnonzero(probabilities>=threshold)}
    if not selected:selected={school_ids[int(np.argmax(probabilities))]}
    positives=set(positive_ids)
    byid={sid:j for j,sid in enumerate(school_ids)}
    pending=[]
    for relation in policy['relations']:
        if relation['kind']!='university':continue
        members=set(relation['school_ids']);hit=selected&members
        if len(hit)>1:
            base=max(hit,key=lambda s:float(probabilities[byid[s]]))
            selected=(selected-members)|{base}
            pending.append('university_model_base_only')
        if len(positives&members)>1:pending.append('multiple_source_positives_in_university_requires_review')
    original=selected|positives
    expanded=set(expand_original(original,graph))
    # SP.apply may prefer exact named facts; preserve independently estimated
    # positives separately rather than declaring this a complete confirmed set.
    decision=SP.apply(policy,site,sorted(original))
    prediction=expanded|set(decision['predicted_school_ids'])
    return dict(predicted_school_ids=sorted(prediction),raw_school_ids=sorted(selected),
        positive_lower_bound=sorted(positives),pending=pending+decision['pending'],
        status='model_estimated_partial_label_completion',enrollment_eligibility=False)
