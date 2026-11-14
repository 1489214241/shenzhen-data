"""Reviewed local qualification syntax, staged for a separate input experiment.

Explicit bracketed household conditions bind only to the enclosed range.
Unbracketed or nested ambiguous qualifications retain the old unknown result.
No current frozen policy package or running experiment is modified here.
"""
import re
from shenzhen_aligned_policy import Parser, combine, node

QUALIFIED=re.compile(r'\[([^\[\]]+)\]的(非深户生|深户生)')


class QualifiedParser(Parser):
    def parse(self,text):
        matches=list(QUALIFIED.finditer(text))
        if not matches:return super().parse(text)
        # Only clauses at the outer scope level can be split independently.
        for m in matches:
            prefix=text[:m.start()]
            if sum(prefix.count(c) for c in '（(')!=sum(prefix.count(c) for c in '）)'):
                return super().parse(text)
            if m.start() and text[m.start()-1] not in '，,、；;。\n ':
                return super().parse(text)
            if m.end()<len(text) and text[m.end()] not in '，,、；;。\n ':
                return super().parse(text)
        children=[];start=0
        for m in matches:
            prefix=text[start:m.start()].strip('，,、；;。\n ')
            if prefix:children.append(super().parse(prefix))
            children.append(node('conditional',text=m.group(0),
                reason='explicit_bracket_scope_requires_household_status',
                condition='non_shenzhen_hukou' if m[2]=='非深户生' else 'shenzhen_hukou',
                children=[super().parse(m[1])]))
            start=m.end()
        suffix=text[start:].strip('，,、；;。\n ')
        if suffix:children.append(super().parse(suffix))
        return combine('or',children)
