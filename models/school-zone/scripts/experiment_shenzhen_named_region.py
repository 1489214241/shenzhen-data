"""Round29: source-named regional witnesses; no model-selected trigger school."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import experiment_shenzhen_residential_supplement as T
import build_shenzhen_named_region_facts as G

F=T.F;R=T.R
PARENT=R.ROOT/'round27_scope_graph_refresh'
OUT=R.ROOT/'round29_named_region'


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen experiment')
    links={}
    for folder,name in [(PARENT,'prediction_freeze.json'),(G.OUT,'manifest.json')]:
        R.check(folder,name);m=F.read(folder/name);links[str(folder/name)]=F.sha(folder/name)
        links.update({str(folder/rel):h for rel,h in m['files'].items()})
        for p,h in m.get('sources',{}).items():assert F.sha(p)==h,p;links[p]=h
    cfg=F.read(PARENT/'protocol.json')
    F.write(OUT/'protocol.json',dict(round=29,arms=cfg['arms'],objectives=cfg['objectives'],labels=cfg['labels'],
        primary='round27/round07/balanced_exact/ensemble',
        hypothesis='Explicit named housing within a complete source region can support another school whose scope contains that same region.',
        intervention='Two mixed-range syntax repairs plus source-named region witnesses; preserve road qualifiers, exclusions and eligibility restrictions.',
        population='All202 source school scopes, all1802 site groups; exact original1761 scored and41 unscored.',
        prediction_trigger_used=False,transitive_expansion=False,training_steps=0,model_probabilities_unchanged=True,
        references_unchanged=True,thresholds_unchanged=True,target_gold_read=False,target_polygon_read=False,
        complete_school_set=False,enrollment_eligibility=False,independent_validation=False,adopted=False))
    counts={}
    for d in R.TARGETS:
        source=F.read(G.OUT/d/'facts.json');F.write(OUT/'facts'/f'{d}.json',source)
        index=defaultdict(set)
        for f in source['facts']:index[f['site_id']].add(f['school_id'])
        for arm in cfg['arms']:
            for obj in cfg['objectives']:
                for label in cfg['labels']:
                    rows=[];prior=F.read(PARENT/'predictions'/arm/obj/label/f'{d}.json')
                    assert set(index)<={r['site_id'] for r in prior}
                    for p in prior:
                        before=set(p['after']);extra=index[p['site_id']]-before
                        rows.append(dict(district=d,site_id=p['site_id'],building_name=p['building_name'],
                            before=sorted(before),after=sorted(before|extra),added_school_ids=sorted(extra),
                            prediction_trigger_used=False,complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    counts[f'{arm}/{obj}/{label}/{d}']=sum(bool(r['added_school_ids']) for r in rows)
    for p in [Path(__file__),Path(G.__file__),Path(T.__file__),F.REPO/'scripts/shenzhen_named_region_scope.py']:
        links[str(p)]=F.sha(p);dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'source_only_changes.json',dict(changed_rows=counts,total_changed=sum(counts.values()),target_gold_read=False))
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN changed',sum(counts.values()),flush=True)
    print('ensemble', {k:v for k,v in counts.items() if '/ensemble/' in k},flush=True)


def score():
    T.OUT=OUT;T.PARENT=PARENT;T.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
