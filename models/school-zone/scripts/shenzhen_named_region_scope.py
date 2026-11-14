"""Preserve explicitly included residential witnesses to a complete region.

No geographic identity is inferred from a school prediction. Only a positive
outer clause R(含 A,B) supplies named witnesses to R. Negated, conditional,
partially parsed and disjunctive base regions do not supply these witnesses.
Qualified road names remain literal; no general road-alias substitution.
"""
import re
from shenzhen_boundary_conjunction_v2 import BoundaryConjunctionParserV2
from shenzhen_aligned_policy import combine, node, walk
from shenzhen_scoped_ring import top_segments
from shenzhen_scope_implication import canonical, Unsupported


def trailing_parenthesis(t):
    if not t.endswith(')'):return None
    depth=0
    for i in range(len(t)-1,-1,-1):
        if t[i]==')':depth+=1
        elif t[i]=='(':
            depth-=1
            if depth==0:return t[:i],t[i+1:-1]
    return None


def region_only(ast):
    return all(n['op'] in ('and','membership','admin','road','ring') for n in walk(ast))


class NamedRegionParser(BoundaryConjunctionParserV2):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        # A complete community/road-side branch followed by a separate
        # hyphen-delimited ring. The separator is outside all parentheses.
        parts=[v.strip() for v,_,_ in top_segments(t) if v.strip()]
        if len(parts)==2:
            prefix,tail=parts
            m=re.fullmatch(r'(.+?)(?:围合|围成|合围)(?:的)?(?:区域|范围)',tail)
            if m and re.fullmatch(r'[^、，,；;。]+(?:社区|居委会)(?:范围内|的)?[^、，,；;。]+以[东西南北](?:片区|范围|住宅)?',prefix):
                boundaries=re.split('[—－-]',m[1])
                legal=all(re.fullmatch(r'[^()、，,；;。]+(?:铁路|大道|路|街|河|高速)(?:\([^()]+\))?',s) for s in boundaries)
                if len(boundaries)>=3 and legal and not re.search(r'不含|除外|居民小组|户籍|深户|以[东西南北]|往|延长|方向',m[1]):
                    return combine('or',[self.parse(prefix),node('ring',names=boundaries,text=tail)])
        result=super().parse(text)
        pair=trailing_parenthesis(t)
        if not pair:return result
        base,tail=pair
        inc=re.fullmatch(r'(?:含|包括|包含)[：:]?(.+)',tail)
        if not inc or re.search(r'不含|除外|以外|另含|另外|户籍|深户|居民小组',inc[1]):return result
        # Require one whole conjunctive region, not several independent clauses.
        region=super().parse(base)
        if not region_only(region):return result
        try:canonical(region)
        except Unsupported:return result
        names=self.parse(re.sub(r'等(?:楼盘|住宅区域|区域)?$','',inc[1]))
        if not all(n['op'] in ('or','named') for n in walk(names)):return result
        terms=[n['text'] for n in walk(names) if n['op']=='named']
        if not terms:return result
        result=dict(result,included_region_witness=dict(source_clause=text,base_text=base,
            included_terms=terms,region_ast=region))
        return result


def positive_witnesses(ast):
    """Never extract assertions below an exclusion or a conditional ancestor."""
    if ast['op'] not in ('or','named','road','ring','membership','admin','and'):return []
    result=[ast['included_region_witness']] if 'included_region_witness' in ast else []
    if ast['op']=='or':
        for child in ast['children']:result.extend(positive_witnesses(child))
    return result
