"""Stage proven negatives in native feature columns and partial-label arrays.

No model training, target scoring, new references or invented negative labels.
"""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import shenzhen_full_inputs as F
from zoning.evidence import policy

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
BASE=F.REPO/'data/深圳市/边界合取修复模型输入_20261112'
FACTS=ROOT/'explicit_scope_facts_v1'
OUT=F.REPO/'data/深圳市/明确排除模型输入_20261113'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve staged inputs')
    links={}
    for folder in [BASE,FACTS]:
        m=F.read(folder/'manifest.json');links[str(folder/'manifest.json')]=F.sha(folder/'manifest.json')
        for p,h in m['files'].items():assert F.sha(folder/p)==h,p;links[str(folder/p)]=h
        for p,h in m.get('sources',{}).items():assert F.sha(p)==h,p;links[p]=h
    audit={}
    for d in F.SZ.DISTRICTS:
        with (BASE/d/'inputs.pkl').open('rb') as f:x=pickle.load(f)
        with (BASE/d/'rules.pkl').open('rb') as f:rules=pickle.load(f)
        with (BASE/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        assert dist.answers=={} and dist.train_answers=={}
        oldy=dict(np.load(BASE/d/'labels.npz'));y={k:v.copy() for k,v in oldy.items()}
        out=deepcopy(x);rr=deepcopy(rules);facts=F.read(FACTS/d/'facts.json')['negative_facts'];relations=[]
        for f in facts:
            assert not f['conflicts_with_prior_positive'];q=dist.site_ids.index(f['site_id']);j=dist.school_ids.index(f['school_id'])
            assert y['labels'][q,j]==-1 and j not in rules.known.get(q,[])
            rr.negative[q,j]=True;y['labels'][q,j]=0
            relations.append(dict(q=q,school=j,site_id=f['site_id'],school_id=f['school_id'],feature_was_already_negative=bool(rules.negative[q,j])))
        y['train_ids']=np.flatnonzero((y['labels']>=0).any(1))
        np.testing.assert_array_equal(x.anchor[:,:,x.anchor_blocks['policy']],policy(rules.statuses,rules.negative))
        np.testing.assert_array_equal(x.pair[:,:,x.pair_blocks['policy']],policy(rules.statuses[:,x.ref_schools],rules.negative[:,x.ref_schools]))
        out.anchor[:,:,out.anchor_blocks['policy']]=policy(rr.statuses,rr.negative)
        out.pair[:,:,out.pair_blocks['policy']]=policy(rr.statuses[:,x.ref_schools],rr.negative[:,x.ref_schools])
        for name,sl in x.anchor_blocks.items():
            if name!='policy':np.testing.assert_array_equal(x.anchor[:,:,sl],out.anchor[:,:,sl])
        for name,sl in x.pair_blocks.items():
            if name!='policy':np.testing.assert_array_equal(x.pair[:,:,sl],out.pair[:,:,sl])
        for key in ['ref_sites','ref_schools','group_bias','self_mask','groups','y','supervised']:
            np.testing.assert_array_equal(getattr(x,key),getattr(out,key))
        np.testing.assert_array_equal(rules.statuses,rr.statuses)
        np.testing.assert_array_equal(oldy['labels']==1,y['labels']==1)
        assert int((y['labels']!=oldy['labels']).sum())==len(facts)
        assert not len(y['valid_ids']) and np.isfinite(out.pair).all() and np.isfinite(out.anchor).all()
        dest=OUT/d;dest.mkdir(parents=True)
        for name in ['district.pkl','source_facts.json','model_metadata.json']:shutil.copyfile(BASE/d/name,dest/name)
        for name,value in [('inputs.pkl',out),('rules.pkl',rr)]:
            with (dest/name).open('wb') as f:pickle.dump(value,f,protocol=5)
        np.savez_compressed(dest/'labels.npz',**y)
        np.savez_compressed(dest/'control_labels.npz',**oldy)
        F.write(dest/'explicit_negative_facts.json',dict(facts=facts,relations=relations))
        audit[d]=dict(proved_negative_relations=len(facts),new_negative_feature_cells=int((rr.negative!=rules.negative).sum()),
            pair_feature_cells_changed=int((out.pair!=x.pair).sum()),school_feature_cells_changed=int((out.anchor!=x.anchor).sum()),
            training_rows_before=len(oldy['train_ids']),training_rows_after=len(y['train_ids']),
            new_training_ids=sorted(set(y['train_ids'].tolist())-set(oldy['train_ids'].tolist())),
            negatives_before=int((oldy['labels']==0).sum()),negatives_after=int((y['labels']==0).sum()),
            positive_labels_unchanged=True,reference_bank_unchanged=True,coordinates_unchanged=True)
        print('STAGED',d,audit[d],flush=True)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_explicit_scope_facts.py',F.REPO/'zoning/evidence.py']:
        links[str(p)]=F.sha(p);dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_audit.json',dict(districts=audit,model_training_performed=False,target_gold_read=False,target_polygon_read=False,
        no_implicit_negative_labels=True,positive_parser_repairs_not_added_to_training=True,
        weight_requirement='New negative-only row must receive the same original per-property mass; source/old target weights remain fixed. Never reuse a zero weight for a newly active row.',adopted=False))
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))


if __name__=='__main__':main()
