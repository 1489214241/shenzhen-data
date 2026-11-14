"""Checkpoint replay and independent output/metric audit for learned set head."""
import numpy as np
import torch
import experiment_modernnca_set_learning as E
from zoning.modernnca_set_learning import LearnedSetReadout,evidence_latents


def main():
    c=E.verify()
    for name in ['cache_freeze.json','selection_freeze.json','weights_freeze.json','prediction_freeze.json']:E.check(name)
    assert E.read(E.OUT/'protocol_receipt.json')['protocol_sha256']==E.sha(E.OUT/'protocol.json')
    arrays=0;rows_checked=0;teacher_rows=0
    xs,ys=E.M.inputs(E.read(E.M.OUT/'protocol.json'))
    for seed in c['seeds']:
        m=LearnedSetReadout(seed,width=c['width']).to(E.DEVICE).eval()
        m.load_state_dict(torch.load(E.OUT/'weights'/f'{seed}.pt',map_location=E.DEVICE,weights_only=True))
        zero=LearnedSetReadout(seed,width=c['width']).to(E.DEVICE).eval()
        teacher=E.M.MultiLabelModernNCA(41,27,seed,32).to(E.DEVICE).eval()
        teacher.load_state_dict(torch.load(E.M.OUT/'weights/categorical'/f'{seed}.pt',map_location=E.DEVICE,weights_only=True))
        for d,x in xs.items():
            cache=torch.load(E.OUT/'cache'/str(seed)/f'{d}.pt',map_location=E.DEVICE,weights_only=True)
            q=np.arange(min(64,x.n));f,p=evidence_latents(teacher,E.M.MD.batch(x,q))
            torch.testing.assert_close(cache['features'][q],f,atol=0,rtol=0)
            torch.testing.assert_close(cache['probabilities'][q],p,atol=0,rtol=0);teacher_rows+=len(q)
        for d in c['targets']:
            sites,mem,valid=E.site_matrices(d);cache=torch.load(E.OUT/'cache'/str(seed)/f'{d}.pt',map_location=E.DEVICE,weights_only=True)
            for arm,model in [('learned_set_readout',m),('untrained_set_readout',zero)]:
                saved=np.load(E.OUT/'target_probabilities'/arm/f'{seed}_{d}.npy');parts=[]
                with torch.no_grad():
                    for lo in range(0,len(sites),64):parts.append(model(cache['features'][lo:lo+64],mem[lo:lo+64]).masked_fill(~valid[lo:lo+64],-torch.inf).softmax(1).cpu().numpy())
                np.testing.assert_array_equal(np.concatenate(parts),saved);arrays+=1
    for label in ['ensemble']+list(map(str,c['seeds'])):
        for d in c['targets']:
            sites=E.read(E.X.OUT/'site_candidates'/f'{d}.json')
            for arm in E.ARMS:
                rows=E.read(E.OUT/'predictions'/arm/label/f'{d}.json')
                if arm=='categorical_policy':
                    old=E.read(E.M.OUT/'predictions/categorical_policy'/label/f'{d}.json')
                    assert rows==old;rows_checked+=len(rows);continue
                pp=[np.load(E.OUT/'target_probabilities'/arm/f'{s}_{d}.npy') for s in c['seeds']]
                p=np.stack(pp).mean(0) if label=='ensemble' else pp[c['seeds'].index(int(label))]
                for q,(row,site) in enumerate(zip(rows,sites)):
                    assert row['site_id']==site['site_id']
                    expected=site['scope_cell_candidates'][int(np.argmax(p[q]))]['school_ids']
                    assert row['after']==expected
                    assert not row['complete_school_set'] and not row['enrollment_eligibility']
                    rows_checked+=1
    result=E.read(E.OUT/'results.json')
    for arm in E.ARMS:
        rr=E.read(E.OUT/'scored_rows'/f'{arm}.json');correct=fp=fn=0
        assert len(rr)==1761 and sum(r['semantic_pending'] for r in rr)==242
        for r in rr:
            g=set(r['truth']);p=set(r['prediction']);assert r['correct']==(p==g)
            correct+=int(p==g);fp+=len(p-g);fn+=len(g-p)
        m=result['metrics'][arm]['全部'];assert (correct,fp,fn)==(m['correct'],m['fp'],m['fn'])
    chosen=E.read(E.OUT/'selected.json');trials=[E.read(E.OUT/'selection'/f'trial{j}/trial.json') for j in range(len(c['grid']))]
    assert chosen['trial']==max(range(len(trials)),key=lambda j:E.rank(trials[j]['best']))
    # Recount every source final ensemble, including the singleton safeguard.
    source=E.read(E.OUT/'source_ensemble.json')
    for arm in ['learned_set_readout','untrained_set_readout']:
        cs=cm=0
        for d in c['sources']:
            pp=[dict(np.load(E.OUT/'source_probabilities'/arm/f'{s}.npz'))[d] for s in c['seeds']]
            p=np.stack(pp).mean(0);y=ys[d]['valid_labels'];cand=E.read(E.OUT/'source_candidates'/f'{d}.json')
            for row,j in zip(y,p.argmax(1)):
                g=set(np.flatnonzero(row));ok=g==set(cand[j]['school_ids'])
                if len(g)>1:cm+=ok
                else:cs+=ok
        assert (cs,cm)==(source[arm]['single_correct'],source[arm]['multi_correct'])
    for d,h in result['gold_sha256'].items():assert E.sha(E.M.F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    E.write(E.OUT/'audit.json',dict(passed=True,target_probability_arrays_bitwise_replayed=arrays,
        output_rows_independently_recounted=rows_checked,teacher_latent_rows_bitwise_replayed=teacher_rows,
        source_selection_and_final_ensemble_recounted=True,hashes_verified=True,gold_unchanged=True,semantic_pending_retained=242))
    print('AUDIT PASSED',arrays,'arrays',rows_checked,'outputs',teacher_rows,'teacher rows',flush=True)


if __name__=='__main__':main()
