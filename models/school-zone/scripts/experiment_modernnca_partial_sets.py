"""Correct the source-only set-head omission using existing target partial labels.

This is a separate frozen development experiment. No new target facts/Gold,
and no partial positive set is labelled complete. Inference candidates unchanged.
"""
import argparse
import collections
import copy
import os
from pathlib import Path
import numpy as np
import torch
import experiment_modernnca_set_learning as E
from zoning.partial_set_learning import compatible_sets,marginal_set_loss
from zoning.modernnca_set_learning import LearnedSetReadout,complete_set_targets,training_loss
from run_modernnca_set_learning_fast import fast_site_matrices

BASE=E.OUT;OUT=E.REPO/'outputs/modernnca_partial_sets_20261113'
DOC=E.REPO.parent/'科研推进/ModernNCA集合学习_20261113'
original_tasks=E.tasks;original_validation=E.validation


def install():
    E.OUT=OUT;E.DOC=DOC;E.tasks=tasks;E.optimize=optimize;E.validation=validation;E.site_matrices=fast_site_matrices


def prepare():
    E.M.guard();E.verify();E.check('prediction_freeze.json');E.check('cache_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve partial-set experiment')
    c=copy.deepcopy(E.read(BASE/'protocol.json'))
    c.update(intervention='Add existing target partial-positive supervision to the learned set head via marginal compatible-set likelihood. Previously it reached only the frozen teacher and inference constraints.',
        source_labels='Existing complete source labels as before, plus original frozen target partial labels. All unknown target cells remain -1; no invented complete sets or negative school labels.',
        target_training_candidates='Global official-policy templates, singleton hypotheses, and observed positive lower-bound hypotheses. Do NOT union a row positive set into every candidate before training; that would make partial loss identically zero.',
        target_training_loss='logsumexp(all candidate scores) minus logsumexp(scores of sets containing every observed positive); unknown schools unrestricted. Source complete-set CE and singleton KL unchanged.',
        training_weights='Original per-project source-balanced and target weights from Modern experiment, jointly renormalized over complete source rows plus original target partial rows. No target weight tuned on evaluation labels.',
        comparator='Source-only trained head804/1761 is a development comparator with identical architecture/input facts but no direct partial head loss; native-policy898 and untrained set879 remain controls.',
        prior_source_only_target_result_seen=True,source_only_failure_not_attributed_to_missing_official_data=True)
    E.write(OUT/'protocol.json',c)
    links=E.read(BASE/'inputs_sha256.json');links.update({str(BASE/p):E.sha(BASE/p) for p in ['protocol.json','cache_freeze.json','prediction_freeze.json']})
    # Immutable caches are hard-linked, read-only in every experiment phase.
    for rel,h in E.read(BASE/'cache_freeze.json').items():
        dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);os.link(BASE/rel,dest);assert E.sha(dest)==h
    E.write(OUT/'cache_freeze.json',E.read(BASE/'cache_freeze.json'))
    for d in c['sources']:E.write(OUT/'source_candidates'/f'{d}.json',E.read(BASE/'source_candidates'/f'{d}.json'))
    audit={}
    for d in c['targets']:
        y=dict(np.load(E.M.OLD/'inputs'/f'{d}_labels.npz'));assert not(y['labels']==0).any()
        meta=E.read(E.M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']];byid={s:i for i,s in enumerate(ids)}
        cat=E.read(E.X.OUT/'catalogues'/f'{d}.json')['candidates']
        sets={tuple(sorted(byid[s] for s in r['school_ids'])) for r in cat}
        sets.update((j,) for j in range(len(ids)))
        sets.update(tuple(np.flatnonzero(y['labels'][q]==1).tolist()) for q in y['train_ids'])
        sets.discard(())
        candidates=[dict(school_ids=list(ss),complete_school_set=False,role='hypothesis_not_complete_target') for ss in sorted(sets,key=lambda ss:(len(ss),ss))]
        E.write(OUT/'target_training_candidates'/f'{d}.json',candidates)
        mem=E.membership(candidates,len(ids),device='cpu');labels=torch.tensor(y['labels'][y['train_ids']]);mask=compatible_sets(labels,mem)
        assert (mask.sum(1)<len(candidates)).all()
        audit[d]=dict(partial_training_rows=len(y['train_ids']),positive_count_histogram=dict(collections.Counter(map(int,(labels==1).sum(1)))),
            complete_target_rows=0,training_hypotheses=len(candidates),compatible_set_count_min=int(mask.sum(1).min()),compatible_set_count_max=int(mask.sum(1).max()),
            uses_new_facts=False,inference_candidates_unchanged=True)
    links.update({str(p):E.sha(p) for folder in ['source_candidates','target_training_candidates'] for p in (OUT/folder).glob('*.json')})
    E.write(OUT/'input_audit.json',audit);E.write(OUT/'inputs_sha256.json',links)
    code=[Path(__file__),E.REPO/'zoning/partial_set_learning.py',E.REPO/'tests/test_partial_set_learning.py',
        E.REPO/'scripts/experiment_modernnca_set_learning.py',E.REPO/'zoning/modernnca_set_learning.py',
        E.REPO/'scripts/run_modernnca_set_learning_fast.py',E.REPO/'scripts/audit_modernnca_set_learning.py',E.REPO/'scripts/audit_modernnca_partial_sets.py']
    E.write(OUT/'code_sha256.json',{str(p.relative_to(E.REPO)):E.sha(p) for p in code})
    for p in code:
        dest=OUT/'code'/p.relative_to(E.REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    E.write(OUT/'protocol_receipt.json',dict(protocol_sha256=E.sha(OUT/'protocol.json'),target_gold_read=False,prior_source_only_result_seen=True))
    E.log('PARTIAL SET PREPARED',audit)


def tasks(c,seed):
    # Original source task builder reads the current OUT installed above.
    tt=original_tasks(c,seed);total=0
    for d,t in tt.items():
        t['weight']=torch.tensor(np.load(E.M.OUT/f'{d}_weights.npy')[t['train']],dtype=torch.float32,device=E.DEVICE)
        t['partial']=False;total+=float(t['weight'].sum())
    for d in c['targets']:
        data=torch.load(OUT/'cache'/str(seed)/f'{d}.pt',map_location=E.DEVICE,weights_only=True)
        y=dict(np.load(E.M.OLD/'inputs'/f'{d}_labels.npz'));ids=y['train_ids']
        candidates=E.read(OUT/'target_training_candidates'/f'{d}.json');mem=E.membership(candidates,y['labels'].shape[1])
        labels=torch.tensor(y['labels'][ids],device=E.DEVICE);allowed=compatible_sets(labels,mem)
        w=torch.tensor(np.load(E.M.OUT/f'{d}_weights.npy')[ids],dtype=torch.float32,device=E.DEVICE);total+=float(w.sum())
        tt[d]=dict(**data,train=ids,membership=mem,allowed=allowed,weight=w,partial=True)
    for t in tt.values():t['weight']/=total
    return tt


def validation(model,tt):return original_validation(model,{d:t for d,t in tt.items() if not t['partial']})


def optimize(c,seed,hp,steps,folder,selection):
    tt=tasks(c,seed);m=LearnedSetReadout(seed,width=c['width']).to(E.DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=hp['lr']);rng=np.random.default_rng(seed);target_rng=np.random.default_rng(seed+101);history=[];best=None
    for step in range(1,steps+1):
        m.train();opt.zero_grad();reg=hp['reg']*sum(p.square().mean() for p in m.parameters());reg.backward();total=float(reg.detach())
        for d,t in tt.items():
            n=len(t['train']);sample_rng=target_rng if t['partial'] else rng
            ix=sample_rng.choice(n,min(c['minibatch'],n),replace=False);q=t['train'][ix]
            z=m(t['features'][q],t['membership'])
            if t['partial']:loss=marginal_set_loss(z,t['allowed'][ix],t['weight'][ix])
            else:loss=training_loss(z,t['target'][ix],t['probabilities'][q],t['singleton_columns'],t['is_single'][ix],t['weight'][ix])
            loss=loss*n/len(ix);assert torch.isfinite(loss),(step,d);loss.backward();total+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(norm);opt.step()
        if step%80==0:E.log('PARTIAL TRAIN',seed,hp,step,round(total,5))
        if selection and step in c['checkpoints']:
            v,_=validation(m,tt);v.update(step=step,loss=total);history.append(v)
            if best is None or E.rank(v)>E.rank(best):best=v;folder.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),folder/'best.pt')
            E.write(folder/'progress.json',dict(hp=hp,best=best,history=history,complete=False));E.log('SOURCE VALID',v)
    folder.mkdir(parents=True,exist_ok=True)
    if selection:
        result=dict(hp=hp,best=best,history=history,complete=True,budget_edge=best['step']==steps);E.write(folder/'trial.json',result);return result
    torch.save(m.state_dict(),folder/f'{seed}.pt');E.write(folder/f'{seed}.json',dict(seed=seed,hp=hp,steps=steps,final_loss=total));E.log('PARTIAL FIT COMPLETE',seed)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','select','fit','infer','score']);phase=ap.parse_args().phase
    if phase=='prepare':prepare()
    else:install();getattr(E,phase)()
