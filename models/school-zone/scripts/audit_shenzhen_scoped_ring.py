"""Enumerate every source AST change before admitting a staged ring parser."""
from collections import Counter
from pathlib import Path
import shenzhen_full_inputs as F
from build_shenzhen_aligned_inputs import parse_scope
from shenzhen_aligned_policy import walk
from shenzhen_qualified_scope import QualifiedParser
from shenzhen_scoped_ring import ScopedRingParser

ROOT = F.REPO / 'outputs/shenzhen_resolution_20261112'
INPUT = F.REPO / 'data/深圳市/局部条件修复输入_20261112'
OUT = ROOT / 'scoped_ring_source_audit_v2'


def signature(a):
    if a['op'] in ('or', 'and'):
        flat = []
        for c in a['children']:
            value = signature(c)
            flat.extend(value[1] if value[0] == a['op'] else [value])
        return a['op'], tuple(sorted(flat, key=repr))
    if 'children' in a:
        return a['op'], tuple(signature(c) for c in a['children'])
    if a['op'] == 'ring':
        return a['op'], tuple(a['names'])
    return tuple((k, repr(v)) for k, v in sorted(a.items()) if k != 'source_text')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve source-only diagnostic snapshot')
    changes = []
    summaries = {}
    for district in F.SZ.DISTRICTS:
        policy = F.read(INPUT / district / 'reviewed_policy.json')
        inventory = policy['school_scope_inventory']
        old = QualifiedParser(inventory, parse_scope)
        new = ScopedRingParser(inventory, parse_scope)
        counts = Counter()
        for row in inventory:
            before = old.parse(row['base_scope_text'])
            after = new.parse(row['base_scope_text'])
            counts['schools'] += 1
            for key, a in [('before', before), ('after', after)]:
                counts[key + '_with_unknown'] += any(v['op'] == 'unknown' for v in walk(a))
                counts[key + '_rings'] += sum(v['op'] == 'ring' for v in walk(a))
            if signature(before) != signature(after):
                counts['semantic_changes'] += 1
                changes.append(dict(district=district, school_id=row['school_id'],
                    school_name=row['school_name'], source_text=row['base_scope_text'],
                    evidence=row['evidence'], before=before, after=after,
                    unknown_before=[v for v in walk(before) if v['op'] == 'unknown'],
                    unknown_after=[v for v in walk(after) if v['op'] == 'unknown'],
                    admitted=False))
        summaries[district] = dict(counts)
    F.write(OUT / 'report.json', dict(summary=summaries, changes=changes,
        target_gold_read=False, target_polygon_read=False, input_snapshot_changed=False,
        adopted=False, limitation='AST syntax only; ring existence and independent geometry remain unverified.'))
    source = [Path(__file__), F.REPO / 'scripts/shenzhen_scoped_ring.py',
              F.REPO / 'tests/test_shenzhen_scoped_ring.py']
    F.write(OUT / 'manifest.json', dict(source_sha256={str(p):F.sha(p) for p in source},
        inputs={str(INPUT/d/'reviewed_policy.json'):F.sha(INPUT/d/'reviewed_policy.json') for d in F.SZ.DISTRICTS},
        report_sha256=F.sha(OUT/'report.json')))
    print(summaries, flush=True)
    for r in changes:
        print(r['school_name'], 'unknown',len(r['unknown_before']), '->',len(r['unknown_after']), flush=True)


if __name__ == '__main__':
    main()
