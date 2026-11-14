"""Stable ratios of model odds from original logits, without probability clipping.

For an ensemble, preserve arithmetic averaging of Bernoulli probabilities:
logit(mean(sigmoid(z))) = logsumexp(logsigmoid(z))
                        - logsumexp(logsigmoid(-z)).
The final ratios are scores, not calibrated probabilities.
"""
import numpy as np


def logsumexp(a, axis=0):
    high=np.max(a,axis=axis,keepdims=True)
    return np.squeeze(high,axis=axis)+np.log(np.exp(a-high).sum(axis=axis))


def mixture_log_odds(logits):
    a=np.asarray(logits,dtype=np.float64)
    if a.ndim!=3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected finite [models,queries,schools] logits')
    positive=-np.logaddexp(0.,-a);negative=-np.logaddexp(0.,a)
    return logsumexp(positive)-logsumexp(negative)


def odds_ratio(logits):
    z=mixture_log_odds(logits)
    return np.exp(z-z.max(1,keepdims=True))


def probability_ratio(logits):
    a=np.asarray(logits,dtype=np.float64)
    p=np.exp(-np.logaddexp(0.,-a)).mean(0)
    return np.divide(p,p.max(1,keepdims=True),out=np.zeros_like(p),where=p.max(1,keepdims=True)>0)
