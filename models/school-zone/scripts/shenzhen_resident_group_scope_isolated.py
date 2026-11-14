"""Resident-group restriction only; deliberately excludes staged ring changes."""
import re
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_aligned_policy import node


class QualifiedResidentGroupParser(QualifiedParser):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        m=re.fullmatch(r'([^()、，,；;。]+以[东西南北])(?:的)?居民小组',t)
        if m:
            return node('conditional',text=text,condition='resident_group_membership',
                reason='road_side_does_not_prove_resident_group_identity',children=[super().parse(m[1])])
        return super().parse(text)
