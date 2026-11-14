"""Staged source-only repair for scoped road rings; no frozen inputs changed.

Recognize an explicit ring as one boundary expression before generic list
splitting. A community in '社区中由 ... 围合' remains an intersection. Road
aliases, directed segments, named partial rings and qualification ambiguity
are deliberately left to the conservative parent parser.
"""
import re

from shenzhen_aligned_policy import combine, node
from shenzhen_qualified_scope import QualifiedParser


def top_segments(text):
    """Return segments and offsets, respecting all parentheses and conjunctions."""
    result = []
    depth = start = i = 0
    while i < len(text):
        c = text[i]
        if c in '([（【':
            depth += 1
        elif c in ')]）】':
            depth -= 1
        width = 0
        if depth == 0:
            if text.startswith('以及', i):
                width = 2
            elif c in '、，,及':
                width = 1
        if width:
            result.append((text[start:i], start, i))
            i += width
            start = i
        else:
            i += 1
    result.append((text[start:], start, len(text)))
    return result


def road_atom(text):
    # Only a complete, unqualified named boundary; never guess a road side.
    if re.search(r'社区|街道|以[东西南北]|方向|延长|交界|往|至|\d+[.．]|[()：:；;。]', text):
        return False
    return bool(re.fullmatch(r'[^、，,及—－\-\s]+(?:铁路|路|街|大道|河|高速)', text))


class ScopedRingParser(QualifiedParser):
    def ring_clause(self, text):
        match = re.fullmatch(r'(.+?)(?:围合|围成|合围)(?:的)?(?:区域|范围|片区|范围内的住宅区域)', text)
        if not match:
            return None
        # Hyphenated rings are already understood by the original parser.
        # Do not split a route with turns or road-segment qualifications.
        parts = top_segments(match[1])
        roads = []
        community = None
        start_index = len(parts)
        for j in range(len(parts) - 1, -1, -1):
            atom = parts[j][0].strip()
            scoped = re.fullmatch(r'([^()、，,及]+社区)中由(.+)', atom)
            candidate = scoped[2] if scoped else atom.removeprefix('由')
            if not road_atom(candidate):
                break
            roads.insert(0, candidate)
            start_index = j
            if scoped:
                community = scoped[1]
                break
            if atom.startswith('由'):
                break
        if len(roads) < 3:
            return None
        ring = node('ring', names=roads, text=text[parts[start_index][1]:])
        if community:
            ring = combine('and', [node('membership', text=community), ring])
        if start_index:
            prefix = text[:parts[start_index - 1][2]].strip()
            # The separator explicitly means union only for a community/list
            # prefix; an earlier road side might bind the following ring too.
            if re.search(r'以[东西南北]|[东西南北]至|围合|围成|合围|方向|交界|延长|沿|往', prefix):
                return None
            ring = combine('or', [self.parse(prefix), ring])
        return ring

    def parse(self, text):
        t = text.replace('（', '(').replace('）', ')').strip(' 。;；\n')
        if re.search('非深户|深户生|户籍|村民子女|原住民|原本地|统筹开班', t):
            return super().parse(text)
        # Preserve a shared phase suffix for the existing Changsha name matcher.
        # Generic comma splitting would invent the residential names '二','三期'.
        phases = re.fullmatch(r'([^()、，,；;。]+?)([一二三四五六七八九十0-9]+(?:、[一二三四五六七八九十0-9]+)+)期(?:\(即([^()]+)\))?', t)
        if phases:
            named = node('named', text=phases[1] + phases[2] + '期')
            return combine('or', [named, self.parse(phases[3])]) if phases[3] else named
        ring = self.ring_clause(t)
        if ring is not None:
            return ring
        # Keep a final include/exclude attached to the complete ring expression.
        if t.endswith(')'):
            depth = 0
            for i in range(len(t) - 1, -1, -1):
                if t[i] == ')':
                    depth += 1
                elif t[i] == '(':
                    depth -= 1
                    if depth == 0:
                        base, tail = t[:i], t[i + 1:-1]
                        ring = self.ring_clause(base)
                        if ring is None:
                            break
                        # On a mixed list, the parenthesis binds to its final
                        # ring branch, not to the unrelated preceding communities.
                        target = ring['children'][-1] if ring['op'] == 'or' else ring
                        inc = re.fullmatch(r'(?:包括|包含|含|即)[：:]?(.+)', tail)
                        ex = re.fullmatch(r'(?:不含|不包括|不包含)(.+)', tail)
                        if not ex:
                            ex = re.fullmatch(r'(.+)(?:除外|以外)', tail)
                        if ex:
                            exclusion = re.sub(r'^.+?招生范围中的', '', ex[1])
                            target = node('minus', children=[target, self.parse(exclusion)], text=text)
                        elif inc and not re.search('不含|不包括|不包含|除外|以外', inc[1]):
                            target = combine('or', [target, self.parse(re.sub(r'等(?:楼盘|住宅区域|区域)?$', '', inc[1]))])
                        else:
                            break
                        if ring['op'] == 'or':
                            return combine('or', ring['children'][:-1] + [target])
                        return target
        return super().parse(text)
