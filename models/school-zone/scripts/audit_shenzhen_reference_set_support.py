"""Audit source multi-label supervision versus actual reference-set support.

Target membership is never scored here. Full source labels are used only for
diagnostics of legal source training and validation; partial sets stay partial.
"""
from collections import defaultdict,Counter
from pathlib import Path
import pickle
import numpy as np
import experiment_shenzhen_reference_retraining as R

F=R.F
OUT=R.ROOT/'reference_set_training_support_audit'


def main():
    F.guard()
    if OUT.exists():raise FileExistsError('Preserve source support audit')
    cfg=R.verify();xs,ys=R.load(cfg,'new_references_labels');summary={};records=[]
    for d,x in xs.items():
        groups=defaultdict(set);sites=defaultdict(set)
        for q,s in zip(x.ref_sites,x.ref_schools):groups[str(x.groups[q])].add(int(s));sites[str(x.groups[q])].add(int(q))
        info=dict(sites=x.n,schools=x.school_n,reference_groups=len(groups),reference_school_coverage=len(set(x.ref_schools)),
            observed_reference_set_sizes=dict(Counter(map(len,groups.values()))),
            partial_reference_sets_not_assumed_complete=True)
        if d in cfg['sources']:
            for split in ['train','valid']:
                ids=ys[d]['train_ids'] if split=='train' else ys[d]['valid_ids']
                labels=ys[d]['labels'][ids] if split=='train' else ys[d]['valid_labels']
                rows=[]
                for q,y in zip(ids,labels):
                    if not (y>=0).all() or (y==1).sum()<2:continue
                    truth=set(np.flatnonzero(y==1));own=str(x.groups[q]);available={g:ss for g,ss in groups.items() if g!=own}
                    # Positive co-observation supports the pair; absence never
                    # establishes a negative or a complete reference label set.
                    both=[g for g,ss in available.items() if truth<=ss]
                    individually=all(any(s in ss for ss in available.values()) for s in truth)
                    nearest={}
                    for s in truth:
                        cols=np.flatnonzero((x.ref_schools==s)&~x.self_mask[q])
                        nearest[str(s)]=float(x.pair[q,cols,x.pair_blocks['distance'].start].min()*300) if len(cols) else None
                    row=dict(district=d,split=split,site_index=int(q),source_gold_schools=[int(s) for s in sorted(truth)],
                        reference_support_for_all_positive_schools=bool(individually),positive_coobserved_reference_groups=both,
                        nearest_positive_reference_distance_m=nearest)
                    rows.append(row);records.append(row)
                info[split+'_multi']=dict(n=len(rows),all_positive_schools_have_reference=sum(r['reference_support_for_all_positive_schools'] for r in rows),
                    positive_set_seen_together_in_reference=sum(bool(r['positive_coobserved_reference_groups']) for r in rows),
                    missing_positive_reference=sum(not r['reference_support_for_all_positive_schools'] for r in rows))
        else:
            assert not (ys[d]['labels']==0).any()
            info['training_positive_only_sites']=len(ys[d]['train_ids'])
            info['target_complete_labels_read']=False
        summary[d]=info;print(d,info,flush=True)
    totals={split:{k:sum(v[split+'_multi'][k] for d,v in summary.items() if d in cfg['sources'])
                   for k in ['n','all_positive_schools_have_reference','positive_set_seen_together_in_reference','missing_positive_reference']}
            for split in ['train','valid']}
    F.write(OUT/'source_multi_rows.json',records)
    F.write(OUT/'report.json',dict(districts=summary,source_totals=totals,target_gold_read=False,target_polygon_read=False,
        interpretation='Availability diagnostic only. Coobserved positive references are not complete answers; no new model or target predictions generated.'))
    F.write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()},
        sources={str(R.OUT/'input_freeze.json'):F.sha(R.OUT/'input_freeze.json'),str(Path(__file__)):F.sha(__file__)}))
    print('TOTALS',totals,flush=True)


if __name__=='__main__':main()
