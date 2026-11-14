"""Apply the cumulative, frozen round31 supplements to original model decisions.

No new facts, graph edges, transitive expansion, or target-label access. This
module makes the previously separate intervention chain reusable for new trials.
"""
from collections import defaultdict
import copy
import experiment_shenzhen_reference_retraining as R
from shenzhen_scope_implication import expand_original

F = R.F
FACT_PATHS = {
    'community_round07': R.OUT / 'facts/{district}.json',
    'road_corridor': R.ROOT / 'road_corridor_scope_estimates/{district}/estimates.json',
    'residential_community': R.ROOT / 'round20_residential_membership/facts/{district}.json',
    'community_road': R.ROOT / 'round25_joint_geography/facts/{district}.json',
    'named_region': R.ROOT / 'named_region_facts_v1/{district}/facts.json',
}


def source_paths(district):
    from pathlib import Path
    paths = {k: Path(str(p).format(district=district)) for k, p in FACT_PATHS.items()}
    paths['graph'] = R.ROOT / 'round27_scope_graph_refresh/graphs' / f'{district}.json'
    paths['explicit_scope'] = R.ROOT / 'explicit_scope_facts_v1' / district / 'facts.json'
    return paths


class CurrentSupplements:
    def __init__(self, district):
        self.district = district
        self.paths = source_paths(district)
        self.hashes = {str(p): F.sha(p) for p in self.paths.values()}
        self.graph = F.read(self.paths['graph'])
        self.positive = defaultdict(lambda: defaultdict(set))
        self.negative = defaultdict(set)
        for kind in FACT_PATHS:
            for fact in F.read(self.paths[kind])['facts']:
                self.positive[fact['site_id']][kind].add(fact['school_id'])
        scope = F.read(self.paths['explicit_scope'])
        for fact in scope['positive_repairs']:
            self.positive[fact['site_id']]['explicit_positive_repair'].add(fact['school_id'])
        for fact in scope['negative_facts']:
            assert not fact['conflicts_with_prior_positive']
            self.negative[fact['site_id']].add(fact['school_id'])
        for site, groups in self.positive.items():
            assert not (set().union(*groups.values()) & self.negative[site]), site

    def apply(self, rows):
        """Rows must retain original decision roots, before late supplements."""
        result = []
        seen = set()
        for original in rows:
            row = copy.deepcopy(original)
            site = row['site_id']
            assert row['district'] == self.district and site not in seen
            seen.add(site)
            decision = row['decision']
            roots = set(decision['raw_school_ids']) | set(decision['positive_lower_bound'])
            before = set(row['after'])
            assert roots <= before
            after = before | set(expand_original(roots, self.graph))
            facts = {k: sorted(v) for k, v in self.positive[site].items()}
            for values in facts.values():
                after.update(values)
            removed = after & self.negative[site]
            after -= self.negative[site]
            if not after:
                raise ValueError(f'Explicit exclusions emptied prediction: {self.district}/{site}')
            row.update(after=sorted(after), cumulative_supplements=dict(
                original_root_school_ids=sorted(roots), source_positives=facts,
                explicit_exclusions=sorted(self.negative[site]), removed_school_ids=sorted(removed),
                added_school_ids=sorted(after-before), transitive_expansion=False,
                complete_school_set=False, enrollment_eligibility=False))
            result.append(row)
        assert set(self.positive) <= seen and set(self.negative) <= seen
        return result
