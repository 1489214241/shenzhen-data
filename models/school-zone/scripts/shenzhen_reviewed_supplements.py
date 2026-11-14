"""Round31 cumulative evidence plus the four reviewed round23 identities.

The extra relations are positive lower bounds. They never trigger another
scope-graph expansion and never establish a complete school set.
"""
from collections import defaultdict
import copy
from shenzhen_current_supplements import CurrentSupplements,F,R

IDENTITIES=R.ROOT/'administrative_identity_review_v1'


class ReviewedSupplements(CurrentSupplements):
    def __init__(self,district):
        super().__init__(district)
        manifest=F.read(IDENTITIES/'manifest.json')
        self.hashes[str(IDENTITIES/'manifest.json')]=F.sha(IDENTITIES/'manifest.json')
        for rel,h in manifest['files'].items():
            path=IDENTITIES/rel;assert F.sha(path)==h;self.hashes[str(path)]=h
        for path,h in manifest.get('sources',{}).items():
            assert F.sha(path)==h;self.hashes[path]=h
        self.extra=defaultdict(set)
        for fact in F.read(IDENTITIES/'school_facts'/f'{district}.json')['facts']:
            kinds={x['kind'] for x in fact['proof']}
            assert {'official_community_membership','reviewed_administrative_identity'}<=kinds
            assert fact['district']==district and not fact['complete_school_set'] and not fact['enrollment_eligibility']
            site,school=fact['site_id'],fact['school_id']
            assert school not in self.negative[site]
            self.extra[site].add(school)
            self.positive[site]['reviewed_administrative_identity'].add(school)

    def augment_historical(self,rows):
        """The historical rows already have round31 facts and exclusions.

Do not re-run its pre-exclusion root assertion on already excluded outputs.
Only append the four extra facts, with no graph expansion.
"""
        result=[];seen=set()
        for original in rows:
            row=copy.deepcopy(original);site=row['site_id']
            assert row['district']==self.district and site not in seen;seen.add(site)
            old=set(row['after']);extra=self.extra.get(site,set())
            assert not (extra&self.negative.get(site,set()))
            row['after']=sorted(old|extra)
            row['reviewed_identity_completion']=dict(source_positive_ids=sorted(extra),added_school_ids=sorted(extra-old),
                transitive_expansion=False,complete_school_set=False,enrollment_eligibility=False)
            result.append(row)
        assert set(self.extra)<=seen
        return result
