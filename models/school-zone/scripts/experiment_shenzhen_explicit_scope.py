"""Round31: source-only parenthesis repair and whole-scope exclusions."""
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
import experiment_shenzhen_residential_supplement as T
import build_shenzhen_explicit_scope_facts as G

F,R=T.F,T.R
PARENT=R.ROOT/'round29_named_region'
OUT=R.ROOT/'round31_explicit_scope'
VARIANTS=['positive_only','negative_only','combined']


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve frozen trial')
    links={}
    for folder,name in [(PARENT,'prediction_freeze.json'),(G.OUT,'manifest.json')]:
        R.check(folder,name);m=F.read(folder/name);links[str(folder/name)]=F.sha(folder/name)
        links.update({str(folder/rel):h for rel,h in m['files'].items()})
        for p,h in m.get('sources',{}).items():assert F.sha(p)==h,p;links[p]=h
    for rel in ['scope_graph_refresh_review/longgang_page9.png','partial_scope_review/longgang_page-10.png',
                'partial_scope_review/longgang_page-11.png','mixed_range_review/longgang_page-13.png']:
        p=R.ROOT/rel;links[str(p)]=F.sha(p)
    cfg=F.read(PARENT/'protocol.json')
    for variant in VARIANTS:
        dest=OUT/variant
        F.write(dest/'protocol.json',dict(round=31,variant=variant,arms=cfg['arms'],objectives=cfg['objectives'],labels=cfg['labels'],
            primary='combined/round07/balanced_exact/ensemble',
            intervention='Add source-proven restored positive names and/or remove whole-scope source-proven exclusions, separately reported.',
            population='All202 schools audited,1802 sites output,1761 scored,41 unscored; original Gold unchanged.',
            no_name_match_is_unknown=True,local_exclusion_is_not_whole_school_exclusion=True,
            official_pdf_pages_visually_reviewed=[9,10,11,13],
            official_url='https://www.lg.gov.cn/attachment/1/1722/1722313/12839768.pdf',
            prediction_trigger_used=False,transitive_expansion=False,training_steps=0,
            references_unchanged=True,thresholds_unchanged=True,model_probabilities_unchanged=True,
            no_new_empty_set='Abort preparation if any source-approved edit would create empty prediction; do not invent a fallback school.',
            target_gold_read=False,target_polygon_read=False,independent_validation=False,adopted=False))
        counts={}
        for d in R.TARGETS:
            source=F.read(G.OUT/d/'facts.json');F.write(dest/'facts'/f'{d}.json',source)
            plus=defaultdict(set);minus=defaultdict(set)
            for f in source['positive_repairs']:plus[f['site_id']].add(f['school_id'])
            for f in source['negative_facts']:
                assert not f['conflicts_with_prior_positive'],'Resolve contradiction before intervention'
                minus[f['site_id']].add(f['school_id'])
            assert all(not (ss&minus[q]) for q,ss in plus.items())
            for arm in cfg['arms']:
                for obj in cfg['objectives']:
                    for label in cfg['labels']:
                        prior=F.read(PARENT/'predictions'/arm/obj/label/f'{d}.json');rows=[]
                        for p in prior:
                            before=set(p['after']);q=p['site_id'];after=set(before)
                            if variant!='negative_only':after|=plus[q]
                            if variant!='positive_only':after-=minus[q]
                            assert after,'Declared nonempty-output gate failed'
                            rows.append(dict(district=d,site_id=q,building_name=p['building_name'],before=sorted(before),after=sorted(after),
                                added_school_ids=sorted(after-before),removed_school_ids=sorted(before-after),
                                prediction_trigger_used=False,complete_school_set=False))
                        F.write(dest/'predictions'/arm/obj/label/f'{d}.json',rows)
                        counts[f'{arm}/{obj}/{label}/{d}']=dict(changed=sum(r['before']!=r['after'] for r in rows),
                            additions=sum(len(r['added_school_ids']) for r in rows),removals=sum(len(r['removed_school_ids']) for r in rows))
        for p in [Path(__file__),Path(G.__file__),F.REPO/'scripts/shenzhen_scoped_parenthesis.py',F.REPO/'scripts/shenzhen_explicit_scope_facts.py']:
            links[str(p)]=F.sha(p);cp=dest/'code'/p.name;cp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,cp)
        F.write(dest/'source_only_changes.json',dict(counts=counts,target_gold_read=False))
        F.write(dest/'input_links.json',dict(files=links))
        F.write(dest/'prediction_freeze.json',dict(files={str(p.relative_to(dest)):F.sha(p) for p in dest.rglob('*') if p.is_file()},target_gold_read=False))
        print('FROZEN',variant,{k:v for k,v in counts.items() if '/ensemble/' in k},flush=True)


def score():
    for variant in VARIANTS:
        T.OUT=OUT/variant;T.PARENT=PARENT;T.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
