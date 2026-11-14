"""Staged repair: a road-qualified residents' group is not every nearby home."""
import re
from shenzhen_scoped_ring import ScopedRingParser
from shenzhen_aligned_policy import node


class ResidentGroupParser(ScopedRingParser):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        m=re.fullmatch(r'([^()、，,；;。]+以[东西南北])(?:的)?居民小组',t)
        if m:
            # Preserve the road expression for a future independently verified
            # identity predicate; current evaluators must leave it unknown.
            return node('conditional',text=text,condition='resident_group_membership',
                        reason='road_side_does_not_prove_resident_group_identity',
                        children=[super().parse(m[1])])
        return super().parse(text)
