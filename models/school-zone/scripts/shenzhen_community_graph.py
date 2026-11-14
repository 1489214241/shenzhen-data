"""Coordinate graph estimates of community identity; never enrollment facts."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components


def graph(xy,k=10,max_edge_m=1500.):
    xy=np.asarray(xy,float);n=len(xy);dist=cdist(xy,xy)
    np.fill_diagonal(dist,np.inf)
    neighbours=np.argsort(dist,axis=1,kind='stable')[:,:min(k,n-1)]
    dd=np.take_along_axis(dist,neighbours,axis=1)
    sigma=max(float(dd[:,-1].mean()),1.)
    rows=np.repeat(np.arange(n),neighbours.shape[1]);cols=neighbours.ravel();ds=dd.ravel()
    keep=ds<=max_edge_m
    w=sparse.coo_matrix((np.exp(-(ds[keep]/sigma)**2),(rows[keep],cols[keep])),shape=(n,n)).tocsr()
    w=w.maximum(w.T);w.setdiag(0);w.eliminate_zeros()
    return w,dict(sigma_m=sigma,undirected_edges=w.nnz//2,components=connected_components(w,directed=False)[0])


def infer(w,xy,anchor_ids,anchor_labels,classes,alpha=.99):
    n=w.shape[0];ids=np.asarray(anchor_ids,int);labs=np.asarray(anchor_labels,int);c=len(classes)
    assert len(ids)==len(labs) and len(set(ids))==len(ids) and len(ids)>0
    y=np.zeros((n,c));y[ids,labs]=1
    degree=np.asarray(w.sum(1)).ravel();inv=np.zeros(n);nz=degree>0;inv[nz]=1/np.sqrt(degree[nz])
    normalized=sparse.diags(inv)@w@sparse.diags(inv)
    lgc=(1-alpha)*spsolve(sparse.eye(n,format='csc')-alpha*normalized.tocsc(),y)
    if lgc.ndim==1:lgc=lgc[:,None]
    ghf=np.zeros_like(y);ghf[ids]=y[ids]
    _,component=connected_components(w,directed=False)
    active=np.isin(component,component[ids]);unlabeled=np.flatnonzero(active&~np.isin(np.arange(n),ids))
    if len(unlabeled):
        lap=sparse.diags(degree)-w
        solution=spsolve(lap[unlabeled][:,unlabeled].tocsc(),w[unlabeled][:,ids]@y[ids])
        if solution.ndim==1:solution=solution[:,None]
        ghf[unlabeled]=solution
    distance=cdist(np.asarray(xy),np.asarray(xy)[ids]);order=np.argsort(distance,axis=1,kind='stable');nearest=order[:,0]
    nd=distance[np.arange(n),nearest]
    nearest_class=labs[nearest]
    other=np.where(labs[None,:]!=nearest_class[:,None],distance,np.inf).min(1)
    margin=1-np.divide(nd,other,out=np.zeros(n),where=np.isfinite(other)&(other>0))
    margin[np.isfinite(other)&(other==0)]=0
    out={}
    for name,score in [('lgc',lgc),('harmonic',ghf)]:
        assert np.isfinite(score).all() and score.min()>-1e-9
        score=np.maximum(score,0);mass=score.sum(1)
        p=np.divide(score,mass[:,None],out=np.zeros_like(score),where=mass[:,None]>1e-14)
        out[name]=dict(predicted=p.argmax(1),confidence=p.max(1),supported=mass>1e-14,nearest_m=nd,values=p)
    out['nearest']=dict(predicted=nearest_class,confidence=np.clip(margin,0,1),supported=np.ones(n,bool),nearest_m=nd)
    return out
