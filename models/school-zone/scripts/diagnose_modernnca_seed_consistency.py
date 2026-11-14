"""Descriptive unique-query stability of the previously frozen LN diagnosis."""
import json
from pathlib import Path
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'科研推进/ModernNCA针对性改进_20261113'


def main():
    rows=json.loads((DOC/'基线诊断逐题.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for row in rows:groups[(row['batch'],row['district'],row['q'])].append(row)
    report={}
    for (batch,d,q),rs in groups.items():
        assert len(rs)==3 and len({r['seed'] for r in rs})==3
        for name in ['两城',rs[0]['city']]:
            c=report.setdefault(f'{batch}/{name}',Counter());c['unique_queries']+=1
            wrong=[r['pred']!=r['truth'] for r in rs]
            c['all_three_correct']+=int(not any(wrong));c['all_three_wrong']+=int(all(wrong))
            c['mixed_correctness']+=int(any(wrong) and not all(wrong))
            if len({r['category'] for r in rs})==1:c['same_category/'+rs[0]['category']]+=1
            if all(wrong) and len({r['pred'] for r in rs})==1:
                c['same_wrong_school']+=1
                if len({r['category'] for r in rs})==1:c['same_wrong_school_and_category/'+rs[0]['category']]+=1
    out=dict(note='Unique-truth queries only. Counts describe three individual-seed predictions, not the category of the probability-averaged ensemble. No significance or independence across draws claimed.',counts={k:dict(v) for k,v in report.items()})
    (DOC/'基线错误跨种子稳定性.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
