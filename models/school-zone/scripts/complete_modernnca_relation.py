"""Run the already frozen experiment sequentially; no GPU concurrency."""
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo / 'outputs/modernnca_relation_20261112'
out.mkdir(parents=True, exist_ok=True)
jobs = [('selection', ['select'])]
jobs += [(f'fit_k{k}_s{s}', ['fit', '--k', str(k), '--seed', str(s)])
         for k, s in [(0, 1), (1, 1), (1, 2), (1, 3)]]
jobs.append(('score', ['score']))
for name, args in jobs:
    print('START', name, flush=True)
    with open(out / f'{name}.log', 'a', encoding='utf-8') as log:
        r = subprocess.run([sys.executable, '-X', 'utf8', 'scripts/run_modernnca_relation.py', *args],
                           cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f'{name} failed: {r.returncode}; see {out / (name + ".log")}')
    print('DONE', name, flush=True)
