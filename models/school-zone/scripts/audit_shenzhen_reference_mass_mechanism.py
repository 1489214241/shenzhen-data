"""Input-only diagnosis of dilution; does not inspect Shenzhen evaluation labels."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import experiment_shenzhen_reference_retraining as E
from shenzhen_schoolwise_reference_mass import schoolwise_bias,batch
from zoning import model as M


def stats(x):
    groups=x.groups[x.ref_sites]
    observed={}
    for g,s in zip(groups,x.ref_schools):observed.setdefault(str(g),set()).add(int(s))
    corrected=schoolwise_bias(x.groups,x.ref_sites,x.ref_schools)
    return dict(reference_pairs=len(groups),reference_properties=len(observed),
        observed_relation_degree=dict(sorted(Counter(len(v) for v in observed.values()).items())),
        diluted_pairs=int((corrected-x.group_bias>1e-6).sum()),
        max_correction_factor=float(np.exp(corrected-x.group_bias).max()) if len(groups) else 1)


def main():
    E.F.guard();c=E.verify();out=E.ROOT/'reference_mass_mechanism_audit'
    if out.exists():raise FileExistsError('Preserve audit')
    old,_=E.load(c,'continued_rank');new,_=E.load(c,'new_references_labels')
    reports={};integration={}
    for d,x in new.items():
        reports[d]=dict(old=stats(old[d]),new=stats(x))
        oq={(int(q),int(s)):k for k,(q,s) in enumerate(zip(old[d].ref_sites,old[d].ref_schools))}
        nq={(int(q),int(s)):k for k,(q,s) in enumerate(zip(x.ref_sites,x.ref_schools))}
        delta=np.array([x.group_bias[nq[k]]-old[d].group_bias[v] for k,v in oq.items()])
        reports[d]['existing_pairs_diluted_by_new_relations']=int((delta < -1e-6).sum())
        reports[d]['min_existing_pair_weight_ratio']=float(np.exp(delta).min()) if len(delta) else 1.
        ids=np.array([0,x.n//2,x.n-1]);native=M.batch(x,ids)
        t0=batch(x,ids,'property_mass');t1=batch(x,ids,'school_mass');t2=batch(x,ids,'school_mass_context')
        for k in ['pair','anchor','valid','px','ax','gb','mem']:
            torch.testing.assert_close(native[k],t0[k],rtol=0,atol=0)
        for k in ['pair','anchor','valid','px','ax','mem']:
            torch.testing.assert_close(native[k],t1[k],rtol=0,atol=0)
        for t in [t1,t2]:
            np.testing.assert_array_equal(t['valid'].cpu().numpy(),~x.self_mask[ids])
            assert torch.isfinite(t['px']).all() and torch.isfinite(t['ax']).all()
        unchanged_context_error=None
        if not reports[d]['new']['diluted_pairs']:
            torch.testing.assert_close(native['gb'],t2['gb'],rtol=0,atol=0)
            # Native context is made on CPU; this arm recomputes the identical
            # formula on the batch device. Record, do not hide, float32 drift.
            unchanged_context_error=max(float((native[k]-t2[k]).abs().max()) for k in ['px','ax'])
            for k in ['px','ax']:torch.testing.assert_close(native[k],t2[k],rtol=3e-6,atol=1e-6)
        integration[d]=dict(control_exact=True,bias_only_other_features_unchanged=True,self_mask_preserved=True,
                            context_finite=True,unchanged_bias_context_max_abs_error=unchanged_context_error,
                            queries_checked=ids.tolist())
        print(d,reports[d],flush=True)
    numeric=[]
    # Fixed pair evidence exp(0), fixed anchor exp(-2): adding an observed school
    # changes native existing-school mass despite no change to its evidence.
    for k in [1,2,3,6]:
        old_mass=1/k+np.exp(-2);new_mass=1+np.exp(-2)
        numeric.append(dict(observed_relations=k,native_probability=float(old_mass/(1+old_mass)),
                            schoolwise_probability=float(new_mass/(1+new_mass))))
    E.F.write(out/'report.json',dict(districts=reports,integration=integration,fixed_evidence_probe=numeric,
        background_url='https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/',
        background_read='More than one class, lines88-101: multiple nonexclusive labels use per-class sigmoid and need not sum to one.',
        derivation='Native pair mass=exp(pair_score)/count(all records for property); proposed divisor=count(records for same property and school).',
        scope='Input and toy mechanism audit, not a causal accuracy estimate; training and model-context interactions require matched experiments.',
        target_gold_read=False,target_polygon_read=False,target_cardinality_used=False))
    E.F.write(out/'manifest.json',dict(files={str(p.relative_to(out)):E.F.sha(p) for p in out.rglob('*') if p.is_file()},
        code={str(p):E.F.sha(p) for p in [Path(__file__),E.F.REPO/'scripts/shenzhen_schoolwise_reference_mass.py']},
        frozen_input_links_sha256=E.F.sha(E.OUT/'input_links.json')))


if __name__=='__main__':main()
