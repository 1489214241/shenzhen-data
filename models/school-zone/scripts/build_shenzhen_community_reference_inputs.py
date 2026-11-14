"""Isolated model inputs: original reference positives plus official source facts.

No target Gold, school polygons, or scored diagnostic files are read. The
round03 source inputs/labels remain frozen. New positive-only estimates must
never become a complete singleton training target or a learned negative border.
"""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import numpy as np
import shenzhen_full_inputs as F
from zoning import evidence as E

ROOT=F.REPO/'outputs/shenzhen_resolution_20261112'
OLD=ROOT/'round03_partial_labels'
RAW=F.REPO/'data/深圳市/修复实验输入_20261112'
FACTS=ROOT/'round07_community_facts'
OUT=F.REPO/'data/深圳市/社区归属参照输入_20261112'


def union_positive_references(known,kind,site_ids,school_ids,facts):
    out={int(q):sorted(set(ss)) for q,ss in known.items()};kinds=dict(kind)
    qi={s:i for i,s in enumerate(site_ids)};si={s:i for i,s in enumerate(school_ids)}
    if len(qi)!=len(site_ids) or len(si)!=len(school_ids):raise ValueError('Duplicate identity')
    added=set();touched=set()
    for f in facts:
        q=qi[f['site_id']];s=si[f['school_id']]
        if s not in out.get(q,[]):added.add((q,s));touched.add(q)
        out[q]=sorted(set(out.get(q,[]))|{s})
        if q not in kinds:kinds[q]='boundary'
    return out,kinds,added,touched


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:pickle.dump(value,f,protocol=5)


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve input snapshot')
    for folder,filename in [(OLD,'input_freeze.json'),(RAW,'manifest.json'),(FACTS,'prediction_freeze.json')]:
        for rel,h in F.read(folder/filename)['files'].items():assert F.sha(folder/rel)==h,(folder,rel)
    cfg=F.read(F.REPO/'configs/default.json')['evidence'];audit={}
    for d in F.SZ.DISTRICTS:
        with (RAW/d/'district.pkl').open('rb') as f:dist=pickle.load(f)
        with (RAW/d/'rules.pkl').open('rb') as f:oldrules=pickle.load(f)
        with (OLD/'inputs'/f'{d}.pkl').open('rb') as f:oldx=pickle.load(f)
        meta=F.read(OLD/'inputs'/f'{d}.json');facts=F.read(FACTS/'facts'/f'{d}.json')['facts']
        assert dist.answers=={} and dist.train_answers=={}
        assert dist.site_ids==[s['coordinate_group_id'] for s in meta['sites']]
        assert dist.school_ids==[s['school_id'] for s in meta['schools']]
        assert {str(q):ss for q,ss in oldrules.known.items()}==meta['known']
        rr=deepcopy(oldrules)
        rr.known,rr.kind,added,touched=union_positive_references(oldrules.known,oldrules.kind,dist.site_ids,dist.school_ids,facts)
        oldpairs={(q,s) for q,ss in oldrules.known.items() for s in ss}
        newpairs={(q,s) for q,ss in rr.known.items() for s in ss}
        assert oldpairs<=newpairs and newpairs-oldpairs==added
        # Keep existing directional/negative arrays unchanged; source positive
        # estimates are separate, not invented cardinal sides or negative labels.
        np.testing.assert_array_equal(rr.statuses,oldrules.statuses)
        np.testing.assert_array_equal(rr.negative,oldrules.negative)
        original_boundary=E.learned_boundary
        def unchanged_boundary(ls,coords,groups,known,ref_sites,**kwargs):
            # New partial positive sets cannot imply inter-school borders. Also
            # withdraw an old singleton used for border learning if supplemented.
            trusted={q:ss for q,ss in oldrules.known.items() if q not in touched}
            return original_boundary(ls,coords,groups,trusted,ref_sites,**kwargs)
        try:
            E.learned_boundary=unchanged_boundary
            x=E.build(dist,rr,cfg)
        finally:E.learned_boundary=original_boundary
        # RuleOutput stores kind per site, whereas old named and new community
        # evidence can coexist at that site. Do not call the new school pair a
        # literal name-listed reference just because another school was named.
        for k,(q,s) in enumerate(zip(x.ref_sites,x.ref_schools)):
            if (int(q),int(s)) in added:
                x.pair[:,k,x.pair_blocks['named_reference']]=0
        # Avoid suppressing existing name evidence just because a new partial
        # positive lower bound is known. Isolate the reference-bank intervention.
        x.anchor=oldx.anchor.copy()
        # The Bernoulli trainer receives the separate three-state label matrix;
        # prohibit accidental reuse by the native singleton-only trainer.
        x.y=np.full(x.n,-1,int);x.supervised=np.empty(0,int)
        yold=dict(np.load(OLD/'inputs'/f'{d}_labels.npz'));ynew={k:v.copy() for k,v in yold.items()}
        for q,s in added:ynew['labels'][q,s]=1
        ynew['train_ids']=np.flatnonzero((ynew['labels']>=0).any(1))
        assert not (ynew['labels']==0).any()
        assert np.all(ynew['labels'][yold['labels']==1]==1)
        assert len(ynew['valid_ids'])==0
        assert x.np_==41 and x.na==27
        np.testing.assert_array_equal(x.groups,oldx.groups)
        np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
        assert set(zip(x.ref_sites,x.ref_schools))=={(q,s) for q,s in newpairs if q not in dist.not_reference}
        assert np.isfinite(x.pair).all() and np.isfinite(x.anchor).all()
        meta=deepcopy(meta);meta['known']={str(q):ss for q,ss in rr.known.items()}
        meta['known_kind']={str(q):v for q,v in rr.kind.items()}
        meta.update(policy_reference_sites=len(rr.known),singleton_training_sites=0,
            complete_school_set=False,reference_estimates_are_positive_only=True,
            caution='For partial-label experiment only: observed positives, no complete singleton targets; no negative boundaries from new facts.')
        dump(OUT/d/'inputs.pkl',x);dump(OUT/d/'district.pkl',dist);dump(OUT/d/'rules.pkl',rr)
        np.savez_compressed(OUT/d/'labels.npz',**ynew)
        np.savez_compressed(OUT/d/'control_labels.npz',**yold)
        F.write(OUT/d/'model_metadata.json',meta);F.write(OUT/d/'source_facts.json',facts)
        audit[d]=dict(sites=x.n,schools=x.school_n,old_reference_sites=len(oldrules.known),
            new_reference_sites=len(rr.known),old_reference_pairs=len(oldx.ref_sites),new_reference_pairs=len(x.ref_sites),
            old_reference_schools=len(set(oldx.ref_schools)),new_reference_schools=len(set(x.ref_schools)),
            added_positive_pairs=len(added),changed_sites=len(touched),
            new_positives_conflicting_with_old_negative_features=sum(bool(rr.negative[q,s]) for q,s in added),
            old_training_rows=len(yold['train_ids']),new_training_rows=len(ynew['train_ids']),
            negative_supervision_cells=int((ynew['labels']==0).sum()),
            old_positives_preserved=True,coordinates_unchanged=True,source_labels_empty=True,
            query_same_property_references_masked=True,school_anchor_features_unchanged=True,
            new_partial_references_used_as_boundary_labels=False,
            input_limitation='Only reference bank changed. Membership features, unresolved roads and full parser adaptation still require controlled integration.')
        print(d,audit[d],flush=True)
    F.write(OUT/'input_audit.json',dict(districts=audit,target_gold_read=False,target_polygon_read=False,adopted=False))
    for p in [Path(__file__),F.REPO/'zoning/evidence.py',F.REPO/'tests/test_shenzhen_community_reference_inputs.py']:
        dest=OUT/'code'/p.relative_to(F.REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'manifest.json',dict(schema='shenzhen_community_positive_reference_v1',
        parent_inputs_sha256=F.sha(OLD/'input_freeze.json'),source_facts_freeze_sha256=F.sha(FACTS/'prediction_freeze.json'),
        evidence_config=cfg,files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        target_gold_read=False,target_polygon_read=False,adopted=False))


if __name__=='__main__':main()
