"""Compose dated community facts with independently sourced road predicates.

Geographical conclusions remain estimates. Unknown atoms stay unknown; a
negative road predicate is never manufactured from a missing road or membership.
"""
from shenzhen_administrative_identity import ReviewedAdministrativeScopeProof


class JointGeographicProof(ReviewedAdministrativeScopeProof):
    def __init__(self, inventory, asts, profiles, district, identities, road_predicate,
                 cutoff='2026-06-01'):
        super().__init__(inventory, asts, profiles, district, identities, cutoff)
        self.road_predicate = road_predicate

    def evaluate(self, ast, fact, stack=()):
        op = ast['op']
        if op == 'road':
            values = [self.road_predicate(name, direction,
                ast.get('explicit_current_names', {}).get(name), fact)
                for name, direction in ast['pairs']]
            if any(v == 0 for v, _ in values):
                return 0, [w for v, ws in values if v == 0 for w in ws]
            if not values or any(v < 0 for v, _ in values):
                return -1, []
            return 1, [w for _, ws in values for w in ws]
        if op in ('and', 'or'):
            values = [self.evaluate(c, fact, stack) for c in ast['children']]
            if op == 'and':
                if any(v == 0 for v, _ in values):
                    return 0, [w for v, ws in values if v == 0 for w in ws]
                if not values or any(v < 0 for v, _ in values):
                    return -1, []
                return 1, [w for _, ws in values for w in ws]
            if any(v == 1 for v, _ in values):
                return 1, [w for v, ws in values if v == 1 for w in ws]
            if values and all(v == 0 for v, _ in values):
                return 0, [w for _, ws in values for w in ws]
            return -1, []
        if op == 'minus':
            base, bw = self.evaluate(ast['children'][0], fact, stack)
            excluded, ew = self.evaluate(ast['children'][1], fact, stack)
            if base == 0:
                return 0, bw
            if excluded == 1:
                return 0, [dict(kind='proved_exclusion', evidence=ew)]
            if base == 1 and excluded == 0:
                return 1, bw + [dict(kind='proved_outside_excluded_region', evidence=ew)]
            return -1, []
        return super().evaluate(ast, fact, stack)


def contains_witness(proof, kind):
    return any(w['kind'] == kind or contains_witness(w.get('evidence', []), kind)
               for w in proof if isinstance(w, dict) and 'kind' in w)
