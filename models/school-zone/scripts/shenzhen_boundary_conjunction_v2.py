"""Repair only faulty fallback OR groups; preserve already-conjunctive scopes."""
import re
from shenzhen_resident_group_scope import ResidentGroupParser
from shenzhen_aligned_policy import combine,node,split_top
from shenzhen_boundary_conjunction import boundary_operand,bare_road_list


class BoundaryConjunctionParserV2(ResidentGroupParser):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        ast=super().parse(text)
        if ast['op']!='or':return ast
        if len(split_top(t,separators='；;。\n'))>1 or re.search(r'范围[一二三四五六七八九十]+[：:]|(?:^|[；;。\n])\s*\d+[.、．]',t):return ast
        if '或' in t:return ast
        out=[];run=[]
        def flush():
            if len(run)>1 and any(a['op']=='unknown' for a in run):out.append(combine('and',list(run)))
            else:out.extend(run)
            run.clear()
        for child in ast['children']:
            directed_admin=child['op']=='admin' and re.match(r'^[东西南北]{1,2}(?:和[东西南北])?至',child.get('source_text',''))
            if directed_admin and run:
                run.append(node('unknown',text=child['source_text'],reason='diagonal_boundary_in_fallback_requires_independent_line'))
            elif boundary_operand(child):run.append(child)
            elif run and bare_road_list(child):
                run.append(node('unknown',text=child['text'],reason='continuation_of_unresolved_directional_boundary'))
            else:flush();out.append(child)
        flush()
        return combine('or',out)
