"""Sequential GPU workers; seal and score only after all districts finish.

Two-worker scheduling was attempted before any target scoring, but this Windows
driver rejected a concurrent CUDA context. Keep the failure logs; resume serially.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = repo / 'outputs/full_evidence_20261112'
groups = [['天心区', '芙蓉区', '雨花区', '开福区'], ['拱墅区', '萧山区', '富阳区', '临安区']]
lock = out / 'scheduler.lock'
out.mkdir(parents=True, exist_ok=True)
fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.write(fd, str(os.getpid()).encode()); os.close(fd)
children = []
try:
    phases = [('select', ['select'])] + [(f'fit_k{k}_s{s}', ['fit', '--k', str(k), '--seed', str(s)]) for k, s in [(0,1),(1,1),(1,2),(1,3)]]
    for phase, args in phases:
        children = []
        for i, districts in enumerate(groups):
            with open(out / f'{phase}_group{i}.log', 'ab') as log:
                command = [sys.executable, '-u', '-X', 'utf8', 'scripts/run_full_evidence.py', *args, '--district', *districts]
                p = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                children = [p]
                print('START', phase, i, p.pid, flush=True)
                code = p.wait()
                if code: raise RuntimeError(f'{phase} group {i} failed: {code}')
        print('DONE', phase, flush=True)
    for script, args in [('run_full_evidence.py', ['score']), ('audit_full_evidence.py', []), ('report_full_evidence.py', [])]:
        p = subprocess.run([sys.executable, '-u', '-X', 'utf8', 'scripts/' + script, *args], cwd=repo)
        if p.returncode: raise RuntimeError(script)
    print('ALL COMPLETE', flush=True)
finally:
    for p in children:
        if p.poll() is None:
            p.terminate(); p.wait()
    lock.unlink(missing_ok=True)
