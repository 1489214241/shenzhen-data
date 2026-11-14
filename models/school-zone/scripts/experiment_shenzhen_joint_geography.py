"""Freeze a community AND road evidence intervention, separately from round24."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import build_shenzhen_joint_geographic_estimates as G
from shenzhen_joint_geographic_proof import contains_witness
import experiment_shenzhen_residential_supplement as T

F=T.F;R=T.R
SOURCE=G.OUT
PARENT=R.ROOT/'round20_residential_membership'
OUT=R.ROOT/'round25_joint_geography'


def prepare():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve intervention')
    R.check(SOURCE,'manifest.json');R.check(PARENT,'prediction_freeze.json')
    links={str(Path(__file__)):F.sha(__file__),str(Path(T.__file__)):F.sha(T.__file__)}
    for folder,manifest_name in [(SOURCE,'manifest.json'),(PARENT,'prediction_freeze.json')]:
        manifest=F.read(folder/manifest_name);links[str(folder/manifest_name)]=F.sha(folder/manifest_name)
        for rel,h in manifest['files'].items():assert F.sha(folder/rel)==h;links[str(folder/rel)]=h
        for p,h in manifest.get('sources',{}).items():assert F.sha(p)==h;links[p]=h
    # Rebuild every source state, not just the single newly accepted relation.
    G.OUT=OUT/'source_replay';G.main()
    compared=[]
    for rel,h in F.read(SOURCE/'manifest.json')['files'].items():
        assert F.sha(G.OUT/rel)==h,rel;compared.append(rel)
    c=F.read(PARENT/'protocol.json')
    F.write(OUT/'protocol.json',dict(round=25,arms=c['arms'],objectives=c['objectives'],labels=c['labels'],
        hypothesis='Combine dated exact community membership with the unchanged Changsha finite-road predicate.',
        intervention='Union all newly derived geographic estimates with both round20 comparator outputs; no new transitive closure.',
        complete_source_population='229 eligible property-community facts, both districts, all202 school scopes; no error-selected source subset.',
        primary_comparator='round20 round07 branch, best previous development candidate',secondary_comparator='round20 round12 parser-repair branch',
        geographic_estimate=True,official_catchment_used_as_input=False,enrollment_eligibility=False,complete_school_set=False,
        training_steps=0,model_probabilities_unchanged=True,model_references_unchanged=True,thresholds_unchanged=True,
        target_gold_read=False,target_polygons_read=False,unknown_conditions_remain_unknown=True,
        independent_validation=False,adopted=False))
    changed={};coverage={}
    for d in R.TARGETS:
        facts=F.read(SOURCE/d/'estimates.json')['facts'];index=defaultdict(set)
        for fact in facts:
            assert contains_witness(fact['proof'],'official_community_membership')
            assert contains_witness(fact['proof'],'independent_road_side_estimate')
            assert fact['is_geographic_estimate'] and not fact['verified_enrollment_eligibility']
            index[fact['site_id']].add(fact['school_id'])
        coverage[d]=dict(relations=len(facts),sites=len(index))
        F.write(OUT/'facts'/f'{d}.json',dict(facts=facts,geographic_estimates=True))
        for arm in c['arms']:
            for objective in c['objectives']:
                for label in c['labels']:
                    prior=F.read(PARENT/'predictions'/arm/objective/label/f'{d}.json')
                    assert set(index)<={r['site_id'] for r in prior};rows=[]
                    for row in prior:
                        before=set(row['after']);extra=index[row['site_id']]-before
                        rows.append(dict(district=d,site_id=row['site_id'],building_name=row['building_name'],
                            before=sorted(before),after=sorted(before|extra),added_school_ids=sorted(extra),
                            geographic_estimate=True,enrollment_eligibility=False,complete_school_set=False))
                    F.write(OUT/'predictions'/arm/objective/label/f'{d}.json',rows)
                    changed[f'{arm}/{objective}/{label}/{d}']=sum(bool(r['added_school_ids']) for r in rows)
    F.write(OUT/'source_only_changes.json',dict(coverage=coverage,changed_rows=changed,source_files_rebuilt=compared))
    for p in [Path(__file__),Path(T.__file__),Path(G.__file__),F.REPO/'scripts/shenzhen_joint_geographic_proof.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p)
        for p in OUT.rglob('*') if p.is_file()},target_gold_read=False))
    print('FROZEN',coverage,'changed rows across all outputs',sum(changed.values()),flush=True)


def score():
    # Identical scoring implementation, candidate universe and original Gold.
    T.OUT=OUT;T.PARENT=PARENT;T.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);globals()[p.parse_args().phase]()
