import numpy as np
import torch
import experiment_modernnca_partial_sets as P
from zoning.partial_set_learning import compatible_sets


def main():
    P.install();E=P.E
    import audit_modernnca_set_learning as A
    A.main()
    counts={}
    for d in E.M.TARGETS:
        y=dict(np.load(E.M.OLD/'inputs'/f'{d}_labels.npz'));labels=y['labels'][y['train_ids']]
        candidates=E.read(P.OUT/'target_training_candidates'/f'{d}.json')
        mem=E.membership(candidates,labels.shape[1],device='cpu');mask=compatible_sets(torch.tensor(labels),mem).numpy()
        for q,row in enumerate(labels):
            required=set(np.flatnonzero(row==1));forbidden=set(np.flatnonzero(row==0))
            expected=[required<=set(c['school_ids']) and not(forbidden&set(c['school_ids'])) for c in candidates]
            assert np.array_equal(mask[q],expected)
        assert (labels!=0).all() and not (labels>=0).all(1).any()
        counts[d]=dict(partial_rows=len(labels),three_positive_rows=int(((labels==1).sum(1)==3).sum()),no_complete_labels_created=True)
    audit=E.read(P.OUT/'audit.json');audit['partial_label_compatibility_independently_checked']=counts;E.write(P.OUT/'audit.json',audit)
    print('PARTIAL SUPERVISION AUDIT PASSED',counts,flush=True)


if __name__=='__main__':main()
