"""Uniform geographic-union features for eight source and two target districts.

Validate source observation setup against the frozen native policy features;
retain all baseline inputs and store only a three-state additive feature block.
"""
from collections import defaultdict
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment_shenzhen_reference_retraining as R
from zoning import rules
from zoning.evidence import policy as policy_features
from shenzhen_composed_features import composed_scope,encode

F=R.F
RAW=F.REPO/'data/深圳市/边界合取单项修复输入_20261112'
TARGET=F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
OUT=F.REPO/'data/深圳市/完整范围特征输入_20261112'
CACHE=F.REPO/'outputs/shenzhen_full_source_cache_20261111'


def source_observations(d,cfg,links):
    base=F.REPO/'data'/d;meta=F.read(base/'district.json')
    obs=base/('observations_bv2' if meta['city']=='杭州' else 'observations')
    paths=[base/'district.json',obs/'observations.json',obs/'arrays.npz']
    O,A=rules.load_observations(obs)
    if meta['city']=='杭州':
        p=base/'observations/names_policy_text.json';t=F.read(p);paths.append(p)
        O['name_known']={int(q):ss for q,ss in t['name_known'].items()};O['name_conflict']=t['name_conflict']
        A['name_negative']=np.zeros((O['n'],O['school_n']),bool)
        for q,s in t['name_negative']:A['name_negative'][q,s]=True
        p=base/'observations/names_policy_text_aliases.json'
        if p.exists():
            paths.append(p)
            for q,ss in F.read(p)['name_known'].items():O['name_known'][int(q)]=sorted(set(O['name_known'].get(int(q),[]))|set(ss))
    links.update({str(p):F.sha(p) for p in paths})
    index=np.asarray(meta['orig'],int) if meta['orig'] is not None else np.arange(O['n'])
    return O,A,rules.PRESETS[cfg['rules']['preset']][d],index


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve feature snapshot')
    c=R.verify();cfg=F.read(CACHE/'manifest.json')['config'];R.check(TARGET,'manifest.json');R.check(RAW,'manifest.json')
    xs,_=R.load(c,'new_references_labels');links={};audit={}
    for d in R.TARGETS:
        with (TARGET/d/'inputs.pkl').open('rb') as handle:xs[d]=pickle.load(handle)
    for d,x in xs.items():
        if d in R.TARGETS:
            O=F.read(RAW/d/'aligned_observations.json');O['name_known']={int(q):ss for q,ss in O['name_known'].items()}
            A=dict(np.load(RAW/d/'observation_arrays.npz'));S=rules.PRESETS['unified_20261106']['芙蓉区'];index=np.arange(x.n)
        else:O,A,S,index=source_observations(d,cfg,links)
        assert len(index)==x.n and O['school_n']==x.school_n
        states,detail=composed_scope(O,A,S);states=states[index]
        if d not in R.TARGETS:
            # Independently reconstruct exactly the native source policy block.
            by=defaultdict(list)
            for cl in O['clauses']:by[cl['school']].append(cl)
            st=np.full((O['n'],x.school_n,4),-1,np.int8);negative=np.zeros((O['n'],x.school_n),bool)
            for j in range(x.school_n):_,st[:,j],negative[:,j]=rules.school(j,by[j],O,A,S)
            for q,ss in O['name_known'].items():negative[q,ss]=False
            negative|=A['name_negative'].astype(bool)
            expected=policy_features(st[index],negative[index])
            np.testing.assert_array_equal(expected,x.anchor[:,:,x.anchor_blocks['policy']])
        dest=OUT/d;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'composed_scope.npz',states=states,features=encode(states))
        F.write(dest/'school_audit.json',detail)
        audit[d]=dict(sites=x.n,schools=x.school_n,state_counts={str(k):int((states==k).sum()) for k in [-1,0,1]},
            source_policy_features_reproduced_exactly=d not in R.TARGETS,
            native_K1_K8_X2_preserved=True,pure_named_clauses_not_promoted=True)
        print(d,audit[d],flush=True)
    F.write(OUT/'input_audit.json',dict(districts=audit,target_gold_read=False,target_polygon_read=False,
        new_features=3,old_features_unchanged=True,scope='Geographic clause union; unknown alternative prevents false exclusion; no new references or training labels.'))
    for folder,manifest in [(TARGET,'manifest.json'),(RAW,'manifest.json'),(R.OLD,'input_freeze.json')]:
        links[str(folder/manifest)]=F.sha(folder/manifest)
        for rel,h in F.read(folder/manifest)['files'].items():links[str(folder/rel)]=h
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_composed_features.py',F.REPO/'tests/test_shenzhen_composed_features.py',F.REPO/'zoning/rules.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


if __name__=='__main__':main()
