"""Numerically identical CPU matrix construction/cache for sealed set inference.

The original frozen training/model/decoder code is untouched. This runner only
avoids hundreds of thousands of tiny GPU indexing calls and repeated JSON reads.
Partial outputs from the interrupted original runner must replay bitwise.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import experiment_modernnca_set_learning as E

CACHE={}


def fast_site_matrices(d):
    if d in CACHE:return CACHE[d]
    meta=E.read(E.M.CURRENT/'metadata'/f'{d}.json');ids=[s['school_id'] for s in meta['schools']];index={s:j for j,s in enumerate(ids)}
    original=E.read(E.X.OUT/'site_candidates'/f'{d}.json')
    sites=[dict(site_id=s['site_id'],building_name=s['building_name'],
        scope_cell_candidates=[dict(school_ids=r['school_ids']) for r in s['scope_cell_candidates']]) for s in original]
    del original
    limit=max(len(s['scope_cell_candidates']) for s in sites)
    matrix=np.zeros((len(sites),limit,len(ids)),dtype=np.float32);valid=np.zeros((len(sites),limit),dtype=bool)
    for q,s in enumerate(sites):
        for j,cc in enumerate(s['scope_cell_candidates']):matrix[q,j,[index[a] for a in cc['school_ids']]]=1;valid[q,j]=True
        matrix[q,len(s['scope_cell_candidates']):]=matrix[q,0]
    CACHE[d]=(sites,torch.from_numpy(matrix).to(E.DEVICE),torch.from_numpy(valid).to(E.DEVICE))
    return CACHE[d]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['infer','audit']);phase=ap.parse_args().phase
    receipt=E.OUT/'runtime_acceleration.json';digest=E.sha(Path(__file__))
    if not receipt.exists():
        assert phase=='infer'
        files={str(p.relative_to(E.OUT)):E.sha(p) for p in (E.OUT/'target_probabilities').rglob('*.npy')}
        E.write(receipt,dict(code_path=str(Path(__file__).resolve()),code_sha256=digest,
            original_partial_probability_hashes=files,change='CPU construction of identical candidate incidence matrices plus reuse within process. Candidate order, schools, padding and valid mask unchanged.',
            reason='Original runner interrupted during repeated fine-grained GPU matrix indexing; no target scores read.'))
    data=E.read(receipt);assert data['code_sha256']==digest
    E.site_matrices=fast_site_matrices
    if phase=='infer':
        E.infer()
        for p,h in data['original_partial_probability_hashes'].items():assert E.sha(E.OUT/p)==h,p
        E.write(E.OUT/'runtime_acceleration_audit.json',dict(original_partial_arrays_bitwise_identical=len(data['original_partial_probability_hashes']),
            code_sha256=digest,original_model_and_training_code_unchanged=True))
        print('ACCELERATION PARITY PASSED',len(data['original_partial_probability_hashes']),flush=True)
    else:
        import audit_modernnca_set_learning as A
        A.main()
        for p,h in data['original_partial_probability_hashes'].items():assert E.sha(E.OUT/p)==h,p


if __name__=='__main__':main()
