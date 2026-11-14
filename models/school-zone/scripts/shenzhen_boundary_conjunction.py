"""Keep incomplete boundary conjunctions conjunctive instead of broad unions.

The parent retains complete native road parsing. This guard only repairs
fallback OR runs containing unresolved spatial operands. Unknown geometry is
not guessed. Explicit numbered/semicolon alternatives remain alternatives.
"""
import re
from shenzhen_resident_group_scope import ResidentGroupParser
from shenzhen_aligned_policy import combine,node,split_top


def boundary_operand(ast):
    if ast['op']=='road':return True
    text=ast.get('source_text',ast.get('text','')).strip()
    if ast['op']=='unknown':
        return bool(re.match(r'^[东西南北]{1,2}(?:和[东西南北])?至',text) or re.search(r'以[东西南北]$',text))
    return False


def bare_road_list(ast):
    if ast['op']!='named':return False
    return bool(re.fullmatch(r'[^()、，,；;。]+(?:路|街|大道|高速)(?:[和及][^()、，,；;。]+(?:路|街|大道|高速))+',ast.get('text','')))


class BoundaryConjunctionParser(ResidentGroupParser):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        # A directional administrative BORDER is not membership of the entire
        # named administrative area, especially a diagonal limiting side.
        if re.fullmatch(r'[东西南北]{1,2}(?:和[东西南北])?至[^()、，,；;。]+街道(?:行政)?区域界线',t):
            return node('unknown',text=text,reason='directed_admin_border_requires_independent_line')
        ast=super().parse(text)
        if ast['op']!='or':return ast
        if len(split_top(t,separators='；;。\n'))>1 or re.search(r'范围[一二三四五六七八九十]+[：:]|(?:^|[；;。\n])\s*\d+[.、．]',t):
            return ast
        if re.search('或者|或是|或',t):return ast
        children=ast['children'];out=[];run=[]
        def flush():
            if len(run)>1 and any(a['op']=='unknown' for a in run):out.append(combine('and',list(run)))
            else:out.extend(run)
            run.clear()
        for child in children:
            if boundary_operand(child):run.append(child)
            elif run and bare_road_list(child):
                run.append(node('unknown',text=child['text'],reason='continuation_of_unresolved_directional_boundary'))
            else:flush();out.append(child)
        flush()
        return combine('or',out)
