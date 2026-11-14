"""Expose legal complete source-training multi sets as references, never validation labels."""
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import pickle
import shutil
import numpy as np
import torch
import experiment_shenzhen_reference_retraining as R
import build_shenzhen_source_reference_completion as C
from zoning import data, rules, evidence as E

F=R.F
OUT=F.REPO/'data/深圳市/来源完整多校参照输入_20261112'
CACHE=F.REPO/'outputs/shenzhen_full_source_cache_20261111'


def source_district(d,links):
    base=F.REPO/'data'/d
    paths=[base/p for p in ['district.json','arrays.npz','extra_evidence.npz','exceptions.json']]
    m=F.read(paths[0]);a=dict(np.load(paths[1]));e=dict(np.load(paths[2]));exc=F.read(paths[3])
    if m['city']=='杭州' and (base/'community_mca.npz').exists():
        paths.append(base/'community_mca.npz');e.update(dict(np.load(paths[-1])))
    obs=base/('observations_bv2' if m['city']=='杭州' else 'observations')
    paths.extend([obs/'observations.json',obs/'arrays.npz'])
    if m['city']=='杭州':
        paths.extend(p for p in [base/'observations/names_policy_text.json',base/'observations/names_policy_text_aliases.json'] if p.exists())
    paths.append(F.REPO/'data/lines'/((m.get('lines') or m['city'])+'.pkl'))
    links.update({str(p):F.sha(p) for p in paths})
    return data.District(name=d,city=m['city'],crs=m['crs'],site_names=m['site_names'],school_names=m['school_names'],
        orig=np.array(m['orig']) if m['orig'] is not None else None,coords=a['coords'],groups=a['groups'].astype(str),school_xy=a['school_xy'],
        answers={},eval_rows=[],train_answers={},scope=None,not_reference=set(exc['not_reference']),not_training=set(exc['not_training']),
        community_name=e['community_name'].astype(str),community_strict=e['community_strict'],community_school=e['community_school'],
        name_match=e['name_match'],lines_key=m.get('lines'))


def select_pairs(x,y,excluded):
    old=defaultdict(set)
    for q,s in zip(x.ref_sites,x.ref_schools):old[int(q)].add(int(s))
    added=[];held=[]
    valid_groups=set(x.groups[y['valid_ids']])
    for q in sorted(map(int,y['train_ids'])):
        row=y['labels'][q]
        if not (row>=0).all() or (row==1).sum()<2:continue
        true=set(map(int,np.flatnonzero(row==1)))
        reason='excluded_reference' if q in excluded else 'conflicting_reference' if not old[q]<=true else None
        assert x.groups[q] not in valid_groups
        if reason:held.append(dict(site=q,reason=reason));continue
        added.extend((q,s) for s in sorted(true-old[q]))
    return added,held


def enrich(x,y,dist,known):
    added,held=select_pairs(x,y,dist.not_reference);out=deepcopy(x)
    if not added:return out,added,held
    oldcol={int(q):i for i,q in enumerate(x.ref_sites)}
    new=sorted({q for q,s in added if q not in oldcol});geo={}
    if new:
        ls=E.lines(dist.lines_key or dist.city);cross,within=E.crossing(ls,dist.coords,dist.coords[new])
        border=E.learned_boundary(ls,dist.coords,dist.groups,known,np.array(new))
        geometry=E.geometry(dist.coords[new][None]-dist.coords[:,None],300)
        cn=dist.community_name;strict=dist.community_strict
        same=(cn[:,None]==cn[new][None])&(cn[:,None]!='');both=strict[:,None]&strict[new][None]
        comm=np.stack([same&both,(~same)&both],-1).astype(np.float32)
        for i,q in enumerate(new):
            a=np.zeros((x.n,x.np_),np.float32)
            for block,val in [('distance',geometry),('crossing',cross),('within_2km',within),('learned_boundary',border),('same_community',comm)]:a[:,x.pair_blocks[block]]=val[:,i]
            geo[q]=a
    st=x.anchor[:,:,x.anchor_blocks['policy']][...,:12].reshape(x.n,x.school_n,4,3).argmax(-1)-1
    cols=[]
    for q,s in added:
        a=x.pair[:,oldcol[q]].copy() if q in oldcol else geo[q].copy()
        a[:,x.pair_blocks['named_reference']]=0
        a[:,x.pair_blocks['policy']]=x.anchor[:,s,x.anchor_blocks['policy']]
        a[:,x.pair_blocks['agreement']]=((st[:,s]>=0)&(st[q,s]>=0)&(st[:,s]==st[q,s])).astype(np.float32)
        a[:,x.pair_blocks['disagreement']]=((st[:,s]>=0)&(st[q,s]>=0)&(st[:,s]!=st[q,s])).astype(np.float32)
        cols.append(a)
    out.pair=np.concatenate([x.pair,np.stack(cols,axis=1)],axis=1)
    out.ref_sites=np.r_[x.ref_sites,[q for q,s in added]].astype(int);out.ref_schools=np.r_[x.ref_schools,[s for q,s in added]].astype(int)
    _,inv,count=np.unique(out.groups[out.ref_sites],return_inverse=True,return_counts=True)
    out.group_bias=-np.log(count[inv]).astype(np.float32);out.self_mask=out.groups[:,None]==out.groups[out.ref_sites][None]
    out.membership=torch.nn.functional.one_hot(torch.tensor(np.r_[out.ref_schools,np.arange(out.school_n)],dtype=torch.long),out.school_n).float()
    return out,added,held


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve snapshot')
    cfg=R.verify();R.check(C.OUT,'manifest.json');xs,ys=R.load(cfg,'new_references_labels')
    config=F.read(CACHE/'manifest.json')['config'];links={};audit={}
    F.write(OUT/'protocol.json',dict(intervention='All eligible complete multi SOURCE TRAINING sites as additional references, including nonreference locations.',
        exclusions='No validation reference labels, target complete labels, conflicting or excluded reference sites.',
        unchanged='Old pair columns, anchors, labels, query-property masks, original rule-derived singleton boundary support; no new negative boundary support.',
        target='Same frozen conjunction-repaired inputs and positive-only labels as control.',
        limitation='Source complete training labels are richer than target partial policy reference evidence; transfer remains to be tested.',adopted=False))
    for d in cfg['sources']:
        x,y=xs[d],ys[d];dist=source_district(d,links);ro=rules.run(dist,**config['rules'])
        expected={(q,s) for q,ss in ro.known.items() if q not in dist.not_reference for s in ss}
        assert expected==set(zip(x.ref_sites,x.ref_schools)),d
        np.testing.assert_array_equal(dist.groups,x.groups)
        np.testing.assert_array_equal(E.policy(ro.statuses,ro.negative),x.anchor[:,:,x.anchor_blocks['policy']])
        out,added,held=enrich(x,y,dist,ro.known)
        np.testing.assert_array_equal(out.pair[:,:len(x.ref_sites)],x.pair);np.testing.assert_array_equal(out.anchor,x.anchor)
        np.testing.assert_array_equal(out.self_mask,out.groups[:,None]==out.groups[out.ref_sites][None])
        assert np.isfinite(out.pair).all()
        dest=OUT/d/'inputs.pkl';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('wb') as h:pickle.dump(out,h,protocol=5)
        np.savez_compressed(OUT/d/'labels.npz',**y)
        F.write(OUT/d/'added_pairs.json',dict(pairs=[dict(site=q,school=s) for q,s in added],held=held))
        audit[d]=dict(added_pairs=len(added),new_locations=len(set(out.ref_sites)-set(x.ref_sites)),
            previous_pairs=len(x.ref_sites),current_pairs=len(out.ref_sites),old_coverage=len(set(x.ref_schools)),new_coverage=len(set(out.ref_schools)),held=held,
            old_pair_anchor_exact=True,old_rule_bank_reproduced=True,new_negative_boundary_labels=0)
        print(d,audit[d],flush=True)
    links.update(F.read(C.OUT/'input_links.json')['files']);links[str(CACHE/'manifest.json')]=F.sha(CACHE/'manifest.json')
    for p in [Path(__file__),F.REPO/'zoning/evidence.py',F.REPO/'zoning/rules.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    F.write(OUT/'input_audit.json',audit);F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False,target_polygon_read=False))


if __name__=='__main__':main()
