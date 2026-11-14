import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy import sparse
from shenzhen_community_graph import infer,graph


def test_harmonic_chain_and_unanchored_component():
    w=sparse.csr_matrix([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,1,0]],dtype=float)
    xy=np.array([[0,0],[1,0],[2,0],[10,0],[11,0]])
    r=infer(w,xy,[0,2],[0,1],['a','b'])
    np.testing.assert_allclose(r['harmonic']['values'][:3],[[1,0],[.5,.5],[0,1]])
    for method in ['lgc','harmonic']:
        assert not r[method]['supported'][3:].any()
        np.testing.assert_array_equal(r[method]['values'][3:],0)


def test_coordinate_graph_and_iteration_agree():
    xy=np.array([[0,0],[0,0],[100,0],[200,0],[4000,0]],float)
    w,_=graph(xy,k=2,max_edge_m=1500)
    assert not w.diagonal().any() and (w-w.T).nnz==0 and w[4].nnz==0
    r=infer(w,xy,[0,3],[0,1],['a','b'],alpha=.8)
    degree=np.asarray(w.sum(1)).ravel();inv=np.divide(1,np.sqrt(degree),out=np.zeros(5),where=degree>0)
    s=inv[:,None]*w.toarray()*inv[None,:];y=np.zeros((5,2));y[0,0]=y[3,1]=1;f=y.copy()
    for _ in range(300):f=.8*s@f+.2*y
    mass=f.sum(1);p=np.divide(f,mass[:,None],out=np.zeros_like(f),where=mass[:,None]>0)
    np.testing.assert_allclose(p,r['lgc']['values'],atol=1e-12)
