"""Post-score target diagnostics only. Never read this output in training or calibration."""
from collections import defaultdict
import pickle
import numpy as np
import experiment_shenzhen_relative_multireference_target as W

T=W.T;F=T.F;R=T.R


def main():
    c=T.verify();assert (R.ROOT/'round17_target_completion_audit.json').exists()
    dest=T.OUT/'diagnostics_not_training/ranking_and_reference_support.json'
    if dest.exists():raise FileExistsError('Preserve diagnosis')
    for d,h in F.read(R.BASE/'results.json')['gold_sha256'].items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    gold={(r['district'],r['site_id']):r for r in F.read(R.BASE/'scored_cases.json')};rows=[]
    for d in R.TARGETS:
        with (T.X.TARGET/d/'inputs.pkl').open('rb') as h:x=pickle.load(h)
        meta=F.read(R.OLD/'inputs'/f'{d}.json');sids=[s['school_id'] for s in meta['schools']];byid={s:i for i,s in enumerate(sids)}
        banks={a:np.stack([np.load(T.OUT/'probabilities'/a/f'{s}.npz')[d] for s in c['seeds']]).mean(0) for a in c['arms']}
        np.testing.assert_array_equal(banks['all_multi_training_absolute'],banks['all_multi_training_relative'])
        refs=defaultdict(set)
        for q,s in zip(x.ref_sites,x.ref_schools):refs[str(x.groups[q])].add(int(s))
        preds={a:F.read(T.OUT/'predictions'/a/'balanced_exact/ensemble'/f'{d}.json') for a in c['arms']}
        for i,site in enumerate(meta['sites']):
            sid=site['coordinate_group_id'];g=gold[d,sid]
            if not g['has_map_gold']:continue
            truth=set(g['gold']);mask=np.array([s in truth for s in sids]);indices={byid[s] for s in truth};own=str(x.groups[i]);available=[ss for k,ss in refs.items() if k!=own]
            per={}
            for a,p in banks.items():
                r=preds[a][i];assert r['site_id']==sid
                stages=dict(raw=r['raw_model_set'],base=r['decision']['raw_school_ids'],policy=r['decision']['predicted_school_ids'],community=r['before_corridor'],final=r['after'])
                per[a]=dict(top1_correct=bool(mask[p[i].argmax()]),rank_separable=bool(p[i,mask].min()>(p[i,~mask].max() if (~mask).any() else -1)),
                    stages={k:dict(exact=set(v)==truth,fp=len(set(v)-truth),fn=len(truth-set(v))) for k,v in stages.items()})
            rows.append(dict(d=d,site_id=sid,size=len(truth),reference_all_positive_schools=indices<=set().union(*available),
                reference_coobserved_positive_set=any(indices<=ss for ss in available),arms=per))
    assert len(rows)==1761
    summary={};groups=[('all',rows),('single',[r for r in rows if r['size']==1]),('multi',[r for r in rows if r['size']>1]),
        ('pairs',[r for r in rows if r['size']==2]),('three_or_more',[r for r in rows if r['size']>=3])]
    groups.extend((d,[r for r in rows if r['d']==d]) for d in R.TARGETS)
    groups.extend((k,[r for r in rows if r['reference_all_positive_schools']==flag]) for k,flag in [('all_positive_ref_available',True),('positive_ref_missing',False)])
    for name,rr in groups:
        summary[name]=dict(n=len(rr),all_positive_refs=sum(r['reference_all_positive_schools'] for r in rr),coobserved_positive_refs=sum(r['reference_coobserved_positive_set'] for r in rr),
            arms={a:dict(top1_correct=sum(r['arms'][a]['top1_correct'] for r in rr),rank_separable=sum(r['arms'][a]['rank_separable'] for r in rr),
                stages={s:{k:sum(r['arms'][a]['stages'][s][k] for r in rr) for k in ['exact','fp','fn']} for s in ['raw','base','policy','community','final']}) for a in c['arms']})
    F.write(dest,dict(groups=summary,target_labels_used_only_after_scoring=True,not_training=True,
        interpretation='Ranking separability and true-school reference support are answer-dependent diagnostics, not deployable gates or achievable accuracy. Stage counts are descriptive, not new rule-removal experiments.'))
    F.write(dest.with_name('ranking_and_reference_support_rows.json'),rows)
    for k in ['all','single','multi','pairs','three_or_more','all_positive_ref_available','positive_ref_missing']:
        r=summary[k];print(k,dict(n=r['n'],all_positive_refs=r['all_positive_refs'],coobserved=r['coobserved_positive_refs'],
            arms={a:{v:r['arms'][a][v] for v in ['top1_correct','rank_separable']} for a in c['arms']}),flush=True)


if __name__=='__main__':main()
