"""Recover an interrupted, unfinished round36 checkpoint without changing math.

Finished seeds and the original frozen training script stay untouched. Only
checkpoint serialization is redirected to a temporary file followed by atomic
replacement. A corrupt unfinished checkpoint is archived before that seed is
recomputed from its original round35 parent under the same fixed budget.
"""
import os
from pathlib import Path
from datetime import datetime,timezone
import psutil
import torch
import experiment_shenzhen_threshold_continuation as E
F,R=E.F,E.R


def main():
    for proc in psutil.process_iter(['pid','name']):
        if proc.pid==os.getpid() or 'python' not in proc.info['name'].lower():continue
        try:args=proc.cmdline()
        except psutil.NoSuchProcess:continue
        if any('resume_shenzhen_round36_atomic.py' in arg for arg in args) or (
            any('experiment_shenzhen_threshold_continuation.py' in arg for arg in args) and 'train' in args):
            raise RuntimeError(f'Existing round36 training process {proc.pid}; do not duplicate it')
    c=E.verify();history=F.read(E.OUT/'training_history.json')
    if (E.OUT/'prediction_freeze.json').exists():R.check(E.OUT,'prediction_freeze.json');return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=E.OUT/'recovery'/stamp;folder.mkdir(parents=True,exist_ok=False)
    archived=[];unchanged={}
    for key,h in history.items():
        arm,seed=key.split('/');p=E.OUT/'checkpoints'/f'{arm}_{seed}.pt'
        assert F.sha(p)==h['checkpoint_sha256'];unchanged[str(p)]=F.sha(p)
    for seed in c['seeds']:
        for arm in E.T.ARMS:
            key=f'{arm}/{seed}';p=E.OUT/'checkpoints'/f'{arm}_{seed}.pt'
            if key in history or not p.exists():continue
            try:torch.load(p,map_location='cpu',weights_only=False)
            except RuntimeError as error:
                if 'PytorchStreamReader' not in str(error):raise
                sha=F.sha(p);size=p.stat().st_size;dest=folder/p.name
                assert p.resolve().is_relative_to(E.OUT.resolve()) and dest.resolve().is_relative_to(E.OUT.resolve())
                p.rename(dest);assert F.sha(dest)==sha
                archived.append(dict(key=key,original=str(p),archive=str(dest),sha256=sha,bytes=size,error=str(error),
                    action='Recompute only unfinished seed/arm from frozen round35 parent with identical original budget.'))
    F.write(folder/'receipt.json',dict(archived=archived,completed_checkpoints=unchanged,
        numerical_protocol_unchanged=True,original_script_unchanged=True,wrapper_sha256=F.sha(__file__),status='resuming'))
    original_save=torch.save
    def atomic_save(obj,destination,*args,**kwargs):
        if isinstance(destination,(str,os.PathLike)):
            p=Path(destination)
            if p.parent.resolve()==(E.OUT/'checkpoints').resolve():
                temporary=E.OUT/'recovery/atomic_write'/p.name
                temporary.parent.mkdir(parents=True,exist_ok=True)
                original_save(obj,temporary,*args,**kwargs)
                os.replace(temporary,p)
                return
        return original_save(obj,destination,*args,**kwargs)
    torch.save=atomic_save
    print('ROUND36_RECOVERY',archived,'COMPLETED_PRESERVED',len(unchanged),flush=True)
    try:E.train()
    finally:torch.save=original_save
    for p,h in unchanged.items():assert F.sha(p)==h,p
    F.write(folder/'completion.json',dict(all_previous_completed_checkpoints_unchanged=True,
        prediction_manifest_sha256=F.sha(E.OUT/'prediction_freeze.json'),wrapper_sha256=F.sha(__file__)))


if __name__=='__main__':main()
