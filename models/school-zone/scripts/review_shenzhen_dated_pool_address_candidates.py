"""Record complete candidate review; facility addresses are not catchment truth.

All 52 dated novel address hits were visually checked against rendered PDF cells.
The exact locators stay distinct from confirmed whole-property memberships.
"""
from collections import Counter
from pathlib import Path
import shenzhen_full_inputs as F

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
SOURCE = ROOT / 'dated_address_inventory_v1'
OUT = ROOT / 'dated_address_review_v1'

# IDs are positions in the frozen 203-row queue, reviewed in full source order.
LOCATORS = {0, 8, 15, 25, 48, 55, 59, 76, 81, 85, 86, 89, 90, 101,
            104, 107, 109, 113, 115, 121, 127, 128, 132, 136, 139, 146, 149}
ADMIN_TOKENS = {7, 70, 75, 92}
PARENT_PHASES = {14, 18, 32, 130}
NAME_EXTENTS = {23, 28, 31, 35, 60, 77, 108, 110, 119, 129, 135, 151}
ADMIN_CONFLICTS = {56, 58, 93, 147}
SUBSTRING = {88}


def main():
    F.guard()
    if OUT.exists():
        raise FileExistsError('Preserve completed address review')
    manifest = F.read(SOURCE / 'manifest.json')
    for p, h in manifest['sources'].items():
        assert F.sha(p) == h, p
    for p, h in manifest['files'].items():
        assert F.sha(SOURCE / p) == h, p
    queue = F.read(SOURCE / 'review_queue.json')['rows']
    groups = [LOCATORS, ADMIN_TOKENS, PARENT_PHASES, NAME_EXTENTS, ADMIN_CONFLICTS, SUBSTRING]
    manual = set.union(*groups)
    assert len(manual) == sum(map(len, groups)) == 52
    eligible = {i for i, r in enumerate(queue) if not r['previous_accepted_identity']
        and r['explicit_geography'] and r['literal_in_address']
        and r['source_row']['temporal_status'] == 'dated_source_candidate'}
    assert eligible == manual
    priors = [f for sub in ['community_membership_sources', 'municipal_residential_membership_sources']
              for section in ['facts', 'pending']
              for f in F.read(ROOT / sub / 'review_v1/membership_facts.json').get(section, [])]
    registry = F.read(ROOT / 'public_property_memberships_review/current_membership_review.json')['rows']
    ledger = []
    for i, r in enumerate(queue):
        if i in LOCATORS:
            decision = 'dated_exact_property_locator_only'
            reason = 'The address explicitly locates a facility in the named property or exact phase. Whole-property administrative extent and the dataset representative point still require confirmation.'
        elif i in ADMIN_TOKENS:
            decision = 'reject_administrative_name_token'
            reason = 'The matched string belongs to the administrative address prefix; it is not a separate property name.'
        elif i in PARENT_PHASES:
            decision = 'pending_parent_vs_phase_extent'
            reason = 'The source names a phase; the dataset hit is the unspecific parent property. Do not copy phase membership to the whole parent.'
        elif i in NAME_EXTENTS:
            decision = 'pending_full_entity_name'
            reason = 'The source uses a longer property name. Do not delete a suffix or infer an alias from substring overlap.'
        elif i in ADMIN_CONFLICTS:
            decision = 'pending_administrative_identity_conflict'
            reason = 'Street/community version or identity conflicts with other official locality evidence; no automatic replacement.'
        elif i in SUBSTRING:
            decision = 'reject_interior_substring'
            reason = '祥苑 is only an interior substring of 雅豪祥苑; no exact property identity is established.'
        elif r['previous_accepted_identity']:
            decision = 'existing_accepted_membership_review_separately'
            reason = 'Not a new property reference; retain separate address evidence and do not overwrite an existing fact.'
        elif r['source_row']['temporal_status'] != 'dated_source_candidate':
            decision = 'pending_post_cutoff_attachment'
            reason = 'Current attachment was modified after the historical cutoff; page title and publication date cannot backdate it.'
        elif not r['explicit_geography']:
            decision = 'pending_no_explicit_street_community'
            reason = 'The address lacks a jointly explicit district, street and community chain.'
        else:
            assert not r['literal_in_address']
            decision = 'pending_unit_name_only'
            reason = 'The candidate name is present only in the operator name; this is not geographical identity proof.'
        previous = [dict(street=p.get('street'), community=p.get('community'),
                         status=p.get('status'), reason=p.get('reason')) for p in priors
                    if (p['district'], p['site_id']) == (r['district'], r['site_id'])]
        current = [dict(street=p['current_query'][0]['item']['StreetOfficeName'],
                        community=p['current_query'][0]['item']['ResidentsCommitteeName']) for p in registry
                   if (p['district'], p['site_id']) == (r['district'], r['site_id'])]
        if i == 56:
            reason += ' Dated address: 六约社区; current exact registry: 六约北社区.'
        elif i == 58:
            reason += ' Dated address: 六约北社区; current exact registry: 六约社区.'
        elif i == 93:
            reason += ' Address pairs 龙城街道 with 五联社区, while the official community inventory pairs 五联社区 with 龙岗街道.'
        elif i == 147:
            reason += ' Address uses 福民社区; dated 丹湖 profile names 华盛观荟. The relationship to 华盛观荟名庭 needs exact entity/version confirmation.'
        ledger.append(dict(candidate_id=i, district=r['district'], site_id=r['site_id'], site_names=r['site_names'],
            decision=decision, reason=reason, source_row=r['source_row'],
            explicit_geography=r['explicit_geography'], previous_review=previous, current_registry=current,
            pdf_cells_visually_reviewed=i in manual, historical_membership_admission=False,
            school_relations_generated=0, model_features_generated=0))
    panels = sorted((SOURCE / 'visual_review').glob('candidates_*.png'))
    assert len(panels) == 7
    summary = dict(total_pairs=len(ledger), decisions=dict(Counter(r['decision'] for r in ledger)),
        visually_reviewed_novel_dated_pairs=52, dated_exact_property_locators=27,
        confirmed_new_whole_property_memberships=0, new_school_relations=0,
        target_gold_read=False, target_polygons_read=False, model_inputs_changed=False,
        next_action='Seek dated project-level membership corroboration for every exact locator, then freeze a same-population experiment; retain partial extent and conflicting records.')
    F.write(OUT / 'review.json', dict(rows=ledger, summary=summary))
    sources = {str(p): F.sha(p) for p in [Path(__file__), SOURCE / 'manifest.json',
        SOURCE / 'review_queue.json', ROOT / 'public_property_memberships_review/current_membership_review.json'] + panels}
    sources.update(manifest['sources'])
    sources.update({str(ROOT / sub / 'review_v1/membership_facts.json'):
        F.sha(ROOT / sub / 'review_v1/membership_facts.json') for sub in
        ['community_membership_sources', 'municipal_residential_membership_sources']})
    F.write(OUT / 'manifest.json', dict(sources=sources, files={'review.json': F.sha(OUT / 'review.json')}))
    print('ADDRESS_REVIEW', summary, flush=True)


if __name__ == '__main__':
    main()
