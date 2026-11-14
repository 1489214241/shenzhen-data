"""Three-valued, source-only proof of school scope from reviewed membership.

Only literal district/street/community identities are admitted. Unobserved
membership, roads, exclusions and household conditions stay unknown. A result
must include a community-source witness; this does not claim a complete school
set, residence eligibility, or a negative label for any unlisted school.
"""
from collections import defaultdict


class MembershipScopeProof:
    def __init__(self, inventory, asts, profiles, district):
        self.inventory=inventory
        self.asts=asts
        self.identities=defaultdict(set)
        for p in profiles:
            if p['district']==district:
                self.identities[p['community']].add((p['street'],p['community']))

    def evaluate(self, ast, fact, stack=()):
        op=ast['op']
        if op=='admin':
            if ast['text']==fact['street']:
                return 1,[dict(kind='official_profile_street_identity',street=fact['street'],site_id=fact['site_id'])]
            return -1,[]
        if op=='membership':
            identity=(fact['street'],fact['community'])
            if self.identities.get(ast['text'])=={identity}:
                return 1,[dict(kind='official_community_membership',community=ast['text'],
                               street=fact['street'],site_id=fact['site_id'])]
            return -1,[]
        if op in ('and','or'):
            vals=[self.evaluate(c,fact,stack) for c in ast['children']]
            if op=='and':
                if any(v==0 for v,_ in vals):return 0,[]
                if any(v<0 for v,_ in vals):return -1,[]
            else:
                if not any(v==1 for v,_ in vals):
                    return (0 if all(v==0 for v,_ in vals) else -1),[]
            return 1,[w for v,ws in vals if v==1 for w in ws]
        if op=='minus':
            b,bw=self.evaluate(ast['children'][0],fact,stack)
            e,_=self.evaluate(ast['children'][1],fact,stack)
            if b==0 or e==1:return 0,[]
            if b==1 and e==0:return 1,bw
            return -1,[]
        if op=='ref':
            j=ast['school']
            if j in stack or self.inventory[j]['status'].endswith('_pending'):return -1,[]
            return self.evaluate(self.asts[j],fact,stack+(j,))
        # In particular, no office proxy, arbitrary region geometry, name
        # substring, household status, admin-parent collapse or policy closure.
        return -1,[]

    def derive(self,facts):
        out=[]
        for fact in facts:
            for j,(row,ast) in enumerate(zip(self.inventory,self.asts)):
                if row['status'].endswith('_pending'):continue
                val,witnesses=self.evaluate(ast,fact,(j,))
                if val==1 and witnesses:
                    out.append(dict(district=fact['district'],site_id=fact['site_id'],
                        building_names=fact['site_names'],school_id=row['school_id'],school_name=row['school_name'],
                        community=fact['community'],street=fact['street'],
                        evidence_kind='official_profile_and_official_school_scope',
                        school_scope_text=row['base_scope_text'],school_sources=row['evidence'],
                        membership_sources=fact['pre_enrollment_evidence'],proof=witnesses,
                        complete_school_set=False,enrollment_eligibility=False))
        return out
