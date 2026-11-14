"""Keep parenthetical phase inclusions/exclusions local to a residential term.

No change to frozen parsers. Geographical conjunctions, qualification clauses,
and mixed unknown scopes remain under the conservative parent parser.
"""
import re
from shenzhen_named_region_scope import NamedRegionParser, trailing_parenthesis
from shenzhen_aligned_policy import combine, node, split_top


def outside_parentheses(text):
    depth=0;out=[]
    for ch in text:
        if ch in '([（【': depth+=1
        elif ch in ')]）】':
            depth-=1
            if depth<0: return None
        elif depth==0:out.append(ch)
    return ''.join(out) if depth==0 else None


def residential_head(text):
    return bool(text.strip()) and not re.search(
        r'[():：；;。\n]|社区|街道|居委会|居民小组|招生|围合|围成|合围|以[东西南北]|[东西南北]至|户籍|深户|村民子女|原住民|统筹|^\d+[.、．]',text)


class ScopedParenthesisParser(NamedRegionParser):
    def parse(self,text):
        t=text.replace('（','(').replace('）',')').strip(' 。;；\n')
        outer=outside_parentheses(t)
        if outer is None:
            return node('unknown',text=text,reason='unbalanced_parentheses')
        # A single named project with an explicit finite phase list and a
        # local exclusion. Keep the exclusion attached; never broaden to P.
        pair=trailing_parenthesis(t)
        if pair and len(split_top(t,separators='、，,'))==1:
            base,tail=pair
            m=re.fullmatch(r'(?:包括|包含|含)([一二三四五六七八九十0-9]+(?:、[一二三四五六七八九十0-9]+)+期)[，,；;](?:不包含|不包括|不含)(.+)',tail)
            if m and residential_head(base) and residential_head(m[2]) and outside_parentheses(m[2])==m[2]:
                return node('minus',children=[node('named',text=base+m[1]),self.parse(m[2])],text=text)
        # A project list is an outer union. Parentheses inside one member must
        # not turn siblings into its excluded or included operands.
        parts=split_top(t,separators='、，,')
        has_local_operator=bool(re.search(r'\([^()]*?(?:不含|不包括|不包含|包括|包含|除外)[^()]*\)',t))
        if len(parts)>1 and has_local_operator and all(
                outside_parentheses(p) is not None and residential_head(outside_parentheses(p)) for p in parts):
            return combine('or',[self.parse(p) for p in parts])
        return super().parse(text)
