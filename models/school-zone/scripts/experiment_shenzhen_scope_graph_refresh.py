"""Round27: wire the repaired source ASTs into the original-root implication path."""
import argparse
from pathlib import Path
import shutil
import build_shenzhen_refreshed_scope_graphs as G
from shenzhen_scope_implication import expand_original
import experiment_shenzhen_residential_supplement as T

F=T.F;R=T.R
PARENT=R.ROOT/'round25_joint_geography'
OUT=R.ROOT/'round27_scope_graph_refresh'
ROOT_PARENTS={'round07':R.ROOT/'round06_balanced_ranking', 'round12':R.ROOT/'round12_conjunction_replay'}


def root_path(arm,obj,label,d):
    base=ROOT_PARENTS[arm]/'predictions'
    if arm=='round07':base=base/'balanced_rank'
    return base/obj/label/f'{d}.json'


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve intervention')
    links={str(Path(__file__)):F.sha(__file__),str(Path(T.__file__)):F.sha(T.__file__)}
    for folder,filename in [(G.OUT,'manifest.json'),(PARENT,'prediction_freeze.json')]+[(p,'prediction_freeze.json') for p in ROOT_PARENTS.values()]:
        R.check(folder,filename);manifest=F.read(folder/filename);links[str(folder/filename)]=F.sha(folder/filename)
        links.update({str(folder/rel):h for rel,h in manifest['files'].items()})
        for p,h in manifest.get('sources',{}).items():assert F.sha(p)==h,p;links[p]=h
    c=F.read(PARENT/'protocol.json')
    F.write(OUT/'protocol.json',dict(round=27,arms=c['arms'],objectives=c['objectives'],labels=c['labels'],
        defect='Parser corrections updated scope observations but original-root prediction graphs still used round03 expressions.',
        intervention='Rebuild all202scope graphs; one new direct inclusion, zero removed. Apply only to original raw_school_ids union positive_lower_bound saved before any graph or policy expansion.',
        source='2026 official Longgang primary guide page9: Shanghai International Studies affiliated school scope equals Ailian primary scope branch2.',
        root_preservation='Do not trigger from previous policy additions, road supplements, community supplements or graph additions. No new transitive closure.',
        university='Existing university/eligibility/pending exclusions unchanged; no new university-wide expansion.',
        training_steps=0,model_probabilities_unchanged=True,model_references_unchanged=True,thresholds_unchanged=True,
        original_root_correctness_unknown=True,complete_school_set=False,enrollment_eligibility=False,
        target_gold_read=False,target_polygon_read=False,independent_validation=False,adopted=False))
    changes={}
    for d in R.TARGETS:
        old=F.read(G.OLD/f'{d}.json');new=F.read(G.OUT/f'{d}.json')
        F.write(OUT/'graphs'/f'{d}.json',new)
        for arm in c['arms']:
            for obj in c['objectives']:
                for label in c['labels']:
                    parent=F.read(PARENT/'predictions'/arm/obj/label/f'{d}.json');roots=F.read(root_path(arm,obj,label,d))
                    assert [r['site_id'] for r in parent]==[r['site_id'] for r in roots];rows=[]
                    for r,q in zip(parent,roots):
                        decision=q['decision'];original=set(decision['raw_school_ids'])|set(decision['positive_lower_bound'])
                        assert original<=set(r['after'])
                        expected_old=set(expand_original(original,old));expected_new=set(expand_original(original,new))
                        assert expected_old<=expected_new
                        extra=expected_new-set(r['after'])
                        rows.append(dict(district=d,site_id=r['site_id'],building_name=r['building_name'],
                            before=r['after'],after=sorted(set(r['after'])|extra),added_school_ids=sorted(extra),
                            original_trigger_school_ids=sorted(original),new_direct_graph_relations=sorted(expected_new-expected_old),
                            complete_school_set=False,enrollment_eligibility=False))
                    F.write(OUT/'predictions'/arm/obj/label/f'{d}.json',rows)
                    changes[f'{arm}/{obj}/{label}/{d}']=sum(bool(r['added_school_ids']) for r in rows)
    for p in [Path(__file__),Path(G.__file__),Path(T.__file__),F.REPO/'scripts/shenzhen_scope_implication.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'source_only_changes.json',dict(changes=changes,target_gold_read=False))
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN graph delta',F.read(G.OUT/'summary.json')['districts'],'changed rows',sum(changes.values()),flush=True)


def score():
    T.OUT=OUT;T.PARENT=PARENT;T.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
