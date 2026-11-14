"""Round28: source-only partial proofs, frozen original roots, fixed scoring."""
import argparse
from pathlib import Path
import shutil
import experiment_shenzhen_scope_graph_refresh as D
import shenzhen_partial_scope as S

F=D.F; R=D.R; T=D.T
PARENT=D.OUT
OUT=R.ROOT/'round28_partial_scope'


def prepare():
    F.guard()
    if OUT.exists(): raise FileExistsError('Preserve frozen experiment')
    links={}
    for folder, name in [(PARENT,'prediction_freeze.json'), (D.G.OUT,'manifest.json')]:
        R.check(folder,name)
        manifest=F.read(folder/name); links[str(folder/name)]=F.sha(folder/name)
        links.update({str(folder/rel):h for rel,h in manifest['files'].items()})
        for p,h in manifest.get('sources',{}).items():
            assert F.sha(p)==h,p
            links[p]=h
    for n in [10,11]:
        p=R.ROOT/'partial_scope_review'/f'longgang_page-{n}.png'
        links[str(p)]=F.sha(p)
    c=F.read(PARENT/'protocol.json')
    F.write(OUT/'protocol.json',dict(round=28, arms=c['arms'], objectives=c['objectives'], labels=c['labels'],
        primary='round27/round07/balanced_exact/ensemble',
        intervention='Retain known Boolean branches while unsupported subtrees remain distinct unknown atoms; require implication for all truth assignments.',
        roots='Only original raw_school_ids union positive_lower_bound; no transitive expansion.',
        source_review='2026 Longgang guide pages9-11 visually checked, including Mumianwan page continuation.',
        training_steps=0, probabilities_unchanged=True, thresholds_unchanged=True, references_unchanged=True,
        target_gold_read=False, target_polygon_read=False, unknown_is_not_negative=True,
        independent_validation=False, adopted=False))
    changes={};graph_summary={}
    for d in R.TARGETS:
        rows=F.read(D.G.OUT/f'{d}_changes.json')['school_inventory']
        old=F.read(D.G.OUT/f'{d}.json');graph=S.build(rows,old)
        policy=F.read(F.SZ.DATA/d/'政策规则/规则包.json')
        for e in graph['added']:
            e['existing_explicit_rules']=[r['rule_id'] for r in policy['relations']
                if r['kind']=='shared' and e['trigger_school_id'] in r['trigger_school_ids'] and e['implied_school_id'] in r['school_ids']]
        F.write(OUT/'graphs'/f'{d}.json',graph)
        graph_summary[d]=dict(schools=len(rows),old_edges=len(old['edges']),new_edges=len(graph['edges']),added=len(graph['added']))
        for arm in c['arms']:
            for obj in c['objectives']:
                for label in c['labels']:
                    parent_path=PARENT/'predictions'/arm/obj/label/f'{d}.json'
                    root_path=D.root_path(arm,obj,label,d);links[str(root_path)]=F.sha(root_path)
                    prior=F.read(parent_path);root=F.read(root_path)
                    assert [r['site_id'] for r in prior]==[r['site_id'] for r in root]
                    output=[]
                    for p,q in zip(prior,root):
                        seeds=set(q['decision']['raw_school_ids'])|set(q['decision']['positive_lower_bound'])
                        extra={e['implied_school_id'] for e in graph['added'] if e['trigger_school_id'] in seeds}-set(p['after'])
                        output.append(dict(district=d,site_id=p['site_id'],building_name=p['building_name'],
                            before=sorted(p['after']),after=sorted(set(p['after'])|extra),added_school_ids=sorted(extra),
                            original_trigger_school_ids=sorted(seeds),complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',output)
                    changes[f'{arm}/{obj}/{label}/{d}']=sum(bool(r['added_school_ids']) for r in output)
    for p in [Path(__file__),Path(S.__file__),Path(D.__file__),Path(T.__file__),F.REPO/'scripts/shenzhen_scope_implication.py']:
        links[str(p)]=F.sha(p);dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'source_only_changes.json',dict(graphs=graph_summary,changed_rows=changes,total_changed=sum(changes.values()),target_gold_read=False))
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',graph_summary,'changed',sum(changes.values()),flush=True)


def score():
    T.OUT=OUT; T.PARENT=PARENT; T.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
