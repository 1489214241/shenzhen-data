"""Audit whether hierarchy loss has the supervision its mechanism requires.

No training, source threshold selection, target inference, or target scoring.
Uses source text and already authorized positive-only target training labels.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
import experiment_shenzhen_set_evidence as D
from shenzhen_hierarchy_loss import masked_max_constraint_loss, relation_matrix

F, R = D.F, D.R
OUT = R.ROOT / 'hierarchical_method_review' / 'supervision_preflight.json'


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed preflight; compare a separate replay')
    c = D.verify(); xs, ys = D.X.load(c, 'all_multi_training')
    links = dict(F.read(D.OUT / 'input_links.json')['files'])
    for p in [Path(__file__), F.REPO / 'scripts/shenzhen_hierarchy_loss.py',
              F.REPO / 'tests/test_shenzhen_hierarchy_loss.py']:
        links[str(p)] = F.sha(p)
    for p in OUT.parent.iterdir():
        if p.is_file(): links[str(p)] = F.sha(p)
    # A deliberately conservative whole-school exact-clause probe. It cannot
    # disprove other local shared-zone relations or geometric containment.
    source = {}
    for d in c['sources']:
        path = F.REPO / 'data' / d / 'observations/observations.json'
        links[str(path)] = F.sha(path); obs = F.read(path)
        groups = {i: [] for i in range(obs['school_n'])}
        for clause in obs['clauses']: groups[clause['school']].append(clause)
        admitted = {}
        for s, cs in groups.items():
            if cs and all(v.get('admitted_by_parse') and v.get('parse_complete')
                          and not v.get('qualification_reason') and not v.get('exclusions') for v in cs):
                admitted[s] = {v['text'] for v in cs}
        edges = [(a, b) for a, av in admitted.items() for b, bv in admitted.items() if a != b and av <= bv]
        source[d] = dict(schools=obs['school_n'], clause_kinds=dict(Counter(v['kind'] for v in obs['clauses'])),
                         complete_school_clause_sets=len(admitted), exact_clause_implications=edges)
    target = {}; matrices = {}; batches = {}
    for d in c['targets']:
        meta_path = R.OLD / 'inputs' / f'{d}.json'
        graph_path = R.ROOT / 'scope_graph_refresh_review/graphs' / f'{d}.json'
        for p in [meta_path, graph_path]: links[str(p)] = F.sha(p)
        meta = F.read(meta_path); graph = F.read(graph_path)
        ids = [s['school_id'] for s in meta['schools']]
        assert len(ids) == xs[d].school_n
        rel = relation_matrix(ids, graph['edges']); index = {s:i for i,s in enumerate(ids)}
        closure = rel.clone()
        for k in range(len(ids)): closure |= closure[:, k:k+1] & closure[k:k+1, :]
        assert torch.equal(rel, closure), 'Declare a closure intervention separately'
        y = ys[d]['labels'][ys[d]['train_ids']]; assert not (y == 0).any()
        positive_pairs=[]; pending_pairs=[]
        for e in graph['edges']:
            a=index[e['trigger_school_id']]; b=index[e['implied_school_id']]
            both=int(((y[:,a]==1)&(y[:,b]==1)).sum())
            pending=int(((y[:,a]==1)&(y[:,b]<0)).sum())
            if both: positive_pairs.append(dict(trigger=e['trigger_school_name'],implied=e['implied_school_name'],rows=both))
            if pending: pending_pairs.append(dict(trigger=e['trigger_school_name'],implied=e['implied_school_name'],rows=pending))
        target[d] = dict(rows=len(y), schools=len(ids), positive_cells=int((y==1).sum()),
                         negative_cells=int((y==0).sum()), edges=len(graph['edges']),
                         positive_pairs=positive_pairs, known_child_unknown_parent=pending_pairs,
                         transitive_edges_added=0, per_seed={})
        matrices[d] = rel.to(R.DEVICE)
        batches[d] = R.frames(c,xs,ys,'new_references_labels',[d])
    start = R.M.LinearRetrieval(next(iter(xs.values())))
    for seed in c['seeds']:
        parent_path=R.PARENT / 'checkpoints' / f'balanced_rank_{seed}.pt'
        links[str(parent_path)] = F.sha(parent_path)
        parent=torch.load(parent_path,map_location=R.DEVICE,weights_only=False)
        model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
        model.load_state_dict(parent['model']); model.eval()
        for d in c['targets']:
            stats=dict(old_loss=0.,new_loss=0.,changed_gradient_cells=0,changed_gradient_rows=0,
                       positive_downward_gradient_cells=0,unknown_nonzero_gradient_cells=0,
                       positive_cells_losing_direct_gradient=0, identity_exact=True)
            for t in batches[d]:
                with torch.no_grad(): z=model(t)
                z=z.detach().requires_grad_(); y=t['labels']; w=t['sample_weight']
                old=R.masked_bce(z,y,w); new=masked_max_constraint_loss(z,y,w,matrices[d])
                identity=masked_max_constraint_loss(z,y,w,torch.eye(z.shape[1],dtype=torch.bool,device=R.DEVICE))
                go=torch.autograd.grad(old,z)[0]; gn=torch.autograd.grad(new,z)[0]
                gi=torch.autograd.grad(identity,z)[0]
                torch.testing.assert_close(old,identity,rtol=0,atol=0)
                torch.testing.assert_close(go,gi,rtol=0,atol=0)
                assert torch.isfinite(new) and torch.isfinite(gn).all()
                stats['old_loss']+=float(old); stats['new_loss']+=float(new)
                stats['changed_gradient_cells']+=int((go!=gn).sum())
                stats['changed_gradient_rows']+=int((go!=gn).any(1).sum())
                stats['positive_downward_gradient_cells']+=int((gn>0).sum())
                stats['unknown_nonzero_gradient_cells']+=int(((y<0)&(gn!=0)).sum())
                stats['positive_cells_losing_direct_gradient']+=int(((y==1)&(go!=0)&(gn==0)).sum())
            assert stats['positive_downward_gradient_cells']==0 and stats['unknown_nonzero_gradient_cells']==0
            target[d]['per_seed'][str(seed)]=stats
            print('AUDIT',seed,d,stats,flush=True)
    F.write(OUT,dict(sources=source,targets=target,files=links,
        no_target_gold=True,no_target_polygon=True,target_inference=False,training_performed=False,
        source_probe_limit='Exact admitted complete clause sets only; zero edges is not proof no school containment exists.',
        mechanism='With positive-only target supervision, loss can redirect upward gradient among observed positives; it cannot directly lower a wrong target-school logit. Shared parameter updates could change other logits indirectly, not measured here.',
        policy_graph_limit='Textual implications are not automatically compatible with the fixed map evaluation semantics.',
        adopted=False))


if __name__ == '__main__': main()
