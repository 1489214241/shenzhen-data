"""Rebuild community splits and compare sparse propagation to native dense solve."""
from collections import Counter
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import experiment_shenzhen_community_graph as D
from shenzhen_community_graph import graph,infer
from zoning.baselines import _propagate

F=D.F


def main():
    F.guard();D.verify();dest=D.ROOT/'round21_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    c=F.read(D.OUT/'protocol.json')
    for p,h in F.read(D.OUT/'prediction_freeze.json')['files'].items():assert F.sha(p)==h,p
    assert F.sha(D.OUT/'calibration.json')==F.read(D.OUT/'selection_freeze.json')['calibration_sha256']
    members=[f for p in D.SOURCES for f in F.read(p)['facts']];audit={}
    for d in c['districts']:
        inp=F.read(D.OUT/'inputs'/f'{d}.json');xy=np.asarray(inp['xy']);sites=F.read(F.SZ.DATA/d/'小区数据/坐标组.json')
        assert [s['coordinate_group_id'] for s in sites]==inp['site_ids']
        np.testing.assert_array_equal(xy,F.project([s['representative_point_wgs84'] for s in sites]))
        allrows=[];splits={}
        for split in ['train','calibration','confirmation','buffer_excluded']:
            splits[split]=F.read(D.OUT/'labels'/split/f'{d}.json');allrows+=splits[split]
        orig={f['site_id']:f for f in members if f['district']==d}
        assert len(allrows)==len(orig) and {r['site_id'] for r in allrows}==set(orig)
        block_splits={}
        for r in allrows:
            m=orig[r['site_id']];assert r['identity']==[m['street'],m['community']] and inp['site_ids'][r['q']]==r['site_id']
            b=tuple(int(z//1000) for z in xy[r['q']]);assert list(b)==r['block']
            bucket=int(hashlib.sha256(f"20261131|{d}|{b[0]}|{b[1]}".encode()).hexdigest(),16)%10
            expected='train' if bucket<6 else 'calibration' if bucket<8 else 'confirmation'
            assert r.get('excluded_from',r['split'])==expected
            assert b not in block_splits or block_splits[b]==expected;block_splits[b]=expected
        for split,opponents in [('train',['calibration','confirmation']),('calibration',['confirmation'])]:
            left=[r['q'] for r in splits[split]];right=[r['q'] for s in opponents for r in splits[s]]
            if left and right:assert cdist(xy[left],xy[right]).min()>100
        for r in splits['buffer_excluded']:
            right=[s['q'] for s in allrows if s.get('excluded_from',s['split']) in (['calibration','confirmation'] if r['excluded_from']=='train' else ['confirmation'])]
            assert cdist(xy[[r['q']]],xy[right]).min()<=100
        train=splits['train'];classes=sorted({tuple(r['identity']) for r in train});index={k:i for i,k in enumerate(classes)}
        ids=np.array([r['q'] for r in train]);labels=np.array([index[tuple(r['identity'])] for r in train])
        w,ga=graph(xy,k=10,max_edge_m=1500);pred=infer(w,xy,ids,labels,classes,alpha=.99)
        saved=F.read(D.OUT/'predictions'/f'{d}.json');assert saved['classes']==[list(x) for x in classes]
        for method,p in pred.items():
            for q,row in enumerate(saved['methods'][method]):
                assert row==dict(identity=list(classes[int(p['predicted'][q])]),confidence=float(p['confidence'][q]),nearest_m=float(p['nearest_m'][q]),supported=bool(p['supported'][q]))
        # Existing native baseline uses dense linear solve, without class priors.
        deg=np.asarray(w.sum(1)).ravel();iv=np.divide(1,np.sqrt(deg),out=np.zeros(len(xy)),where=deg>0)
        sm=iv[:,None]*w.toarray()*iv[None,:];y=np.zeros((len(xy),len(classes)));y[ids,labels]=1
        dense=_propagate(sm,y,.99,False,np.zeros(len(classes)));mass=dense.sum(1)
        dense=np.divide(dense,mass[:,None],out=np.zeros_like(dense),where=mass[:,None]>1e-12)
        np.testing.assert_allclose(dense,pred['lgc']['values'],atol=2e-12,rtol=2e-12)
        _,cc=connected_components(w,directed=False);active=np.isin(cc,cc[ids]);unknown=np.flatnonzero(active&~np.isin(np.arange(len(xy)),ids))
        harmonic=pred['harmonic']['values'];residual=(sparse.diags(deg)-w)@harmonic
        np.testing.assert_allclose(residual[unknown],0,atol=1e-10)
        np.testing.assert_array_equal(harmonic[ids],y[ids]);assert not pred['harmonic']['supported'][~active].any()
        audit[d]=dict(n=len(xy),known_community_sites=len(allrows),split_counts={k:len(v) for k,v in splits.items()},
            graph=ga,lgc_native_dense_maxdiff=float(np.max(np.abs(dense-pred['lgc']['values']))),harmonic_residual=float(np.max(np.abs(residual[unknown]))))
        print('REPLAYED',d,audit[d],flush=True)
    cal=F.read(D.OUT/'calibration.json');report=F.read(D.OUT/'results.json');coverage={}
    for method in [c['primary']]+c['comparators']:
        rows=D.join_labels('calibration',method,c);assert D.statistics(rows)==cal[method]['all']
        for option in cal[method]['candidates']:
            stat=D.statistics(D.subset(rows,option['threshold']));assert all(option[k]==v for k,v in stat.items())
        assert not any(option['passed'] for option in cal[method]['candidates']) and cal[method]['selected'] is None
        confirmed=D.join_labels('confirmation',method,c)
        assert not any(r['correct'] and not r['supported'] for r in confirmed+rows)
        assert D.statistics(confirmed)==report['reports'][method]['all']
        assert report['reports'][method]['accepted']['n']==0 and not report['reports'][method]['gate_passed']
        coverage[method]={split:dict(all=D.statistics(rs),with_class_support=D.statistics([r for r in rs if r['class_has_anchor']]),
            no_class_support=D.statistics([r for r in rs if not r['class_has_anchor']]),
            per_district={d:D.statistics([r for r in rs if r['district']==d]) for d in c['districts']}) for split,rs in [('calibration',rows),('confirmation',confirmed)]}
    assert not report['eligible_for_school_experiment'] and not report['school_predictions_generated']
    F.write(dest,dict(districts=audit,validation=coverage,source_labels_and_coordinates_rebuilt=True,
        no_cross_split_reference_within100m=True,all_predictions_replayed=True,school_gold_read=False,
        caveat='Only3Longhua confirmation points, below5district gate. No resplit. Overall scores also inadequate.',
        independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('COMMUNITY_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
