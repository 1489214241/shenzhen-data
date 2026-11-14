"""Reuse the audited geometry-sharing boundary estimator with school support masks."""
import importlib.util
from pathlib import Path
import numpy as np
from zoning import evidence as E

LEGACY=Path(__file__).resolve().parents[2]/'科研推进/参照缺失与政策反证验证_20261107/experiment.py'


def compute(dist,known,ref_sites,hidden_schools):
    spec=importlib.util.spec_from_file_location('frozen_reference_missing_geometry',LEGACY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Do not call setup/run or read historical labels/configuration.
    module.np=np;module.evidence=E
    masks={}
    for name,hidden in hidden_schools.items():
        keep=np.ones(dist.n,bool)
        for q,ss in known.items():
            if any(hidden[s] for s in ss):keep[q]=False
        masks[name]=keep
    return module.boundaries_many(E.lines(dist.lines_key or dist.city),dist.coords,dist.groups,known,ref_sites,masks)
