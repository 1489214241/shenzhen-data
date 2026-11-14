"""Explicit, dated administrative identities for three-valued scope proof.

Only reviewed coextensive community/sole-committee relations are accepted.
This is not suffix stripping, parent-to-child inference, or a school alias table.
"""
from datetime import date
from shenzhen_community_facts import MembershipScopeProof


class ReviewedAdministrativeScopeProof(MembershipScopeProof):
    def __init__(self, inventory, asts, profiles, district, identities, cutoff='2026-06-01'):
        super().__init__(inventory, asts, profiles, district)
        self.district = district
        self.cutoff = date.fromisoformat(cutoff)
        self.reviewed = {}
        for identity in identities:
            if identity['district'] != district:
                continue
            if identity['relation'] != 'community_and_sole_committee_coextensive':
                raise ValueError('Only explicitly coextensive identities can be substituted')
            if identity['decision'] != 'accepted_dated_identity' or not identity['evidence']:
                raise ValueError('Reviewed dated evidence is required')
            observed = date.fromisoformat(identity['version_not_before'])
            if observed > self.cutoff:
                raise ValueError('Identity source postdates experiment cutoff')
            for evidence in identity['evidence']:
                if date.fromisoformat(evidence['publication_date']) > self.cutoff:
                    raise ValueError('Evidence postdates experiment cutoff')
                if not all(evidence.get(k) for k in ['quote', 'url', 'sha256']):
                    raise ValueError('Missing provenance')
            term = identity['scope_term']
            if term in self.reviewed:
                raise ValueError('Ambiguous administrative identity')
            expected = (identity['street'], identity['community'])
            if self.identities.get(identity['community']) != {expected}:
                raise ValueError('Community must have a unique district-scoped identity')
            # An identity already recognised by literal lookup cannot be overwritten.
            if term in self.identities and self.identities[term] != {expected}:
                raise ValueError('Mapping contradicts an existing literal identity')
            self.reviewed[term] = identity

    def evaluate(self, ast, fact, stack=()):
        if ast['op'] == 'membership' and ast['text'] in self.reviewed:
            identity = self.reviewed[ast['text']]
            if (fact['district'], fact['street'], fact['community']) != (
                    self.district, identity['street'], identity['community']):
                return -1, []
            # Never move a pre-split property automatically into a post-split child.
            dates = [date.fromisoformat(v[:10]) for v in fact.get('source_dates', [])]
            lower = date.fromisoformat(identity['version_not_before'])
            if not any(lower <= day <= self.cutoff for day in dates):
                return -1, []
            return 1, [dict(kind='official_community_membership',
                community=fact['community'], street=fact['street'], site_id=fact['site_id']),
                dict(kind='reviewed_administrative_identity', scope_term=ast['text'],
                    community=fact['community'], street=fact['street'],
                    version_not_before=identity['version_not_before'], evidence=identity['evidence'])]
        return super().evaluate(ast, fact, stack)
