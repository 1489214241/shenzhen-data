"""Independently audit logit reconstruction, exact thresholds and source gate."""
import numpy as np
import torch
from scipy.special import logsumexp, log_expit
import experiment_shenzhen_logmass_decode as D
from shenzhen_logmass_decode import odds_ratio
from audit_shenzhen_round18 import transform
from audit_shenzhen_round16 import select,counts
from zoning.evaluate import bootstrap

F=D.F;R=D.R


def main():
    F.guard();c=D.verify();R.check(D.OUT,'logit_freeze.json')
    dest=R.ROOT/'round26_independent_completion_audit.json'
    if dest.exists():raise FileExistsError('Preserve audit')
    report=F.read(D.OUT/'source_report.json');_,ys=D.X.load(c,'all_multi_training')
    raw={a:[dict(np.load(D.OUT/'logits'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    old={a:[dict(np.load(D.D.OUT/'probabilities'/a/f'{s}.npz')) for s in c['seeds']] for a in c['arms']}
    collected=F.read(D.OUT/'collection_audit.json');scores={};numerics={};observed={};predictions={}
    for arm in c['arms']:
        for i,seed in enumerate(c['seeds']):
            assert set(raw[arm][i])=={d+s for d in c['sources'] for s in ['', '_train_multi','_stress_0','_stress_1']}
            assert F.sha(D.D.OUT/'checkpoints'/f'{arm}_{seed}.pt')==collected[f'{arm}/{seed}']['checkpoint_sha256']
            for key,z in raw[arm][i].items():
                actual=torch.sigmoid(torch.from_numpy(z).to(R.DEVICE)).cpu().numpy()
                np.testing.assert_array_equal(actual,old[arm][i][key])
        for mode in D.MODES:
            name=arm+'_'+mode;scores[name]={};maximum=0.
            for key in raw[arm][0]:
                if mode=='probability_ratio':s=transform(np.stack([p[key] for p in old[arm]]).mean(0),'relative')
                else:
                    z=np.stack([p[key] for p in raw[arm]]).astype(np.float64)
                    s=odds_ratio(z)
                    alternate=logsumexp(log_expit(z),axis=0)-logsumexp(log_expit(-z),axis=0)
                    alternate=np.exp(alternate-alternate.max(1,keepdims=True))
                    np.testing.assert_allclose(s,alternate,rtol=1e-12,atol=1e-13)
                    if s.size:maximum=max(maximum,float(np.max(np.abs(s-alternate))))
                scores[name][key]=s
            numerics[name]=dict(scipy_independent_mixture_max_abs_difference=maximum)
            claim=report['reports'][name];t=claim['selections']['balanced_exact']['threshold']
            cal=F.read(D.OUT/'calibration'/f'{name}.json');options=[]
            for row in cal['candidates']:
                ns=nm=0
                for d in c['sources']:
                    ct=counts(select(scores[name][d],row['threshold']),ys[d]['valid_labels'].astype(bool))
                    ns+=ct['single_correct'];nm+=ct['multi_correct']
                assert (ns,nm)==(row['single_correct'],row['multi_correct']);options.append((row['threshold'],ns,nm))
            for objective in ['micro_exact','balanced_exact']:
                best=max(options,key=lambda v:((v[1]+v[2]) if objective=='micro_exact' else v[1]*60+v[2]*1186,v[0]))
                selected=claim['selections'][objective];assert best==(selected['threshold'],selected['single_correct'],selected['multi_correct'])
            for view in ['full','train_multi','stress_0','stress_1']:
                total={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']};attainable=0
                if view=='full':predictions[name]={}
                for d in c['sources']:
                    labels=ys[d]['valid_labels'].astype(bool);suffix='' if view=='full' else '_'+view
                    if view=='train_multi':
                        z=ys[d]['labels'][ys[d]['train_ids']];labels=z[(z>=0).all(1)&((z==1).sum(1)>1)].astype(bool)
                    p=scores[name][d+suffix];pred=select(p,t)
                    for k,v in counts(pred,labels).items():total[k]+=v
                    for pr,tr in zip(p,labels):
                        if tr.sum()>1:attainable+=int(pr[tr].min()>pr[~tr].max())
                    if view=='full':predictions[name][d]=pred
                expected=claim['selections']['balanced_exact'] if view=='full' else claim['train_multi'] if view=='train_multi' else claim['stress'][view]
                assert all(expected[k]==v for k,v in total.items());observed[name+'/'+view]=dict(**total,attainable_multi=attainable)
            full=observed[name+'/full'];gate=c['source_gate']
            passed=full['single_correct']>=gate['min_single_correct'] and full['multi_correct']>=gate['min_multi_correct'] and full['attainable_multi']>=gate['min_attainable_multi']
            assert passed==claim['source_gate_passed'] and full['attainable_multi']==claim['attainable_multi']
            for i,seed in enumerate(c['seeds']):
                tot={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
                for d in c['sources']:
                    p=odds_ratio(raw[arm][i][d][None]) if mode=='odds_ratio' else transform(old[arm][i][d],'relative')
                    for k,v in counts(select(p,t),ys[d]['valid_labels'].astype(bool)).items():tot[k]+=v
                assert tot==claim['seeds'][str(seed)]
            print('AUDITED',name,full,flush=True)
    rows=[]
    for d in c['sources']:
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            truth=ys[d]['valid_labels'][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(z//1000) for z in xy[q])),multi=bool(truth.sum()>1),
                gold=np.flatnonzero(truth).tolist(),predicted={k:np.flatnonzero(v[d][i]).tolist() for k,v in predictions.items()},
                correct={k:bool((v[d][i]==truth).all()) for k,v in predictions.items()}))
    assert rows==F.read(D.OUT/'paired_source_rows.json')
    for arm in c['arms']:
        a=arm+'_odds_ratio';b=arm+'_probability_ratio'
        for label,subset in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in subset]
            measured=dict(n=len(subset),before_correct=sum(r['correct'][b] for r in subset),after_correct=sum(r['correct'][a] for r in subset),
                corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(subset,delta,seed=20261104,draws=3000))
            assert measured==report['paired'][arm][label]
    a=c['primary_candidate'];b=c['strong_control'];claims=report['reports']
    stress_pass=all(observed[a+'/'+s]['correct']>=observed[b+'/'+s]['correct'] for s in ['stress_0','stress_1'])
    assert stress_pass==report['primary_stress_gate_passed']
    selected=a if claims[a]['source_gate_passed'] and claims[a]['balanced_exact']>=claims[b]['balanced_exact'] and stress_pass else None
    assert selected==report['selected_target_candidate']
    F.write(dest,dict(raw_logit_files=20,original_sigmoid_rows_replayed=76640,scipy_mixture_checks=numerics,
        counts=observed,all_transition_counts_and_seeds_verified=True,all_paired_rows_and_intervals_verified=True,
        source_gate_recomputed=True,selected_target_candidate=selected,target_predictions_generated=False,
        independent_validation=False,adopted=False,code_sha256=F.sha(__file__)))
    print('ROUND26_AUDIT_COMPLETE',selected,flush=True)


if __name__=='__main__':main()
