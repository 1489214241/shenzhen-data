"""Per-seed source harm and false-positive/negative audit for rounds34/35."""
import argparse
import numpy as np
import experiment_shenzhen_adaptive_threshold as T
F,R,E=T.F,T.R,T.E


def run(number):
    out=R.ROOT/('round34_adaptive_threshold' if number==34 else 'round35_threshold_convergence')
    F.guard();R.check(out,'prediction_freeze.json')
    dest=out/'seed_paired_audit.json'
    if dest.exists():raise FileExistsError('Preserve seed audit')
    c=F.read(out/'protocol.json');_,ys=E.load(c,'features_labels')
    threshold=F.read(E.OUT/'source_report.json')['reports']['features_labels_relative']['selections']['balanced_exact']['threshold']
    old=F.read(out/'source_seed_metrics.json');results={}
    for seed in c['seeds']:
        payloads={a:dict(np.load(E.OUT/'probabilities/features_labels'/f'{seed}.npz' if a=='original_relative' else out/'margins'/a/f'{seed}.npz')) for a in ['original_relative']+T.ARMS}
        rows=[]
        for d in c['sources']:
            for i,y in enumerate(ys[d]['valid_labels']):
                truth=set(np.flatnonzero(y));pred={}
                for arm,p in payloads.items():
                    z=np.asarray(p[d][i] if arm=='original_relative' else p[d+'_valid'][i],float)
                    ids=np.flatnonzero(z/z.max()>=threshold) if arm=='original_relative' else np.flatnonzero(z>0)
                    if not len(ids):ids=np.array([np.argmax(z)])
                    pred[arm]=set(ids)
                rows.append(dict(d=d,q=int(ys[d]['valid_ids'][i]),multi=len(truth)>1,truth=truth,pred=pred))
        results[str(seed)]={}
        for a,b in [('adaptive_head','original_relative'),('adaptive_head','constant_head'),('constant_head','original_relative')]:
            groups={}
            for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
                ac=[r['pred'][a]==r['truth'] for r in rs];bc=[r['pred'][b]==r['truth'] for r in rs]
                value=dict(n=len(rs),before_correct=sum(bc),after_correct=sum(ac),corrected=sum(u and not v for u,v in zip(ac,bc)),harmed=sum(v and not u for u,v in zip(ac,bc)),
                    fp_before=sum(len(r['pred'][b]-r['truth']) for r in rs),fp_after=sum(len(r['pred'][a]-r['truth']) for r in rs),
                    fn_before=sum(len(r['truth']-r['pred'][b]) for r in rs),fn_after=sum(len(r['truth']-r['pred'][a]) for r in rs))
                if name=='all' or name in c['sources']:
                    for arm,side in [(a,'after'),(b,'before')]:
                        prior=old[arm][str(seed)]['all'] if name=='all' else old[arm][str(seed)]['districts'][name]
                        assert prior['correct']==value[side+'_correct'] and prior['fp']==value['fp_'+side] and prior['fn']==value['fn_'+side]
                groups[name]=value
            results[str(seed)][a+'_vs_'+b]=groups
    F.write(dest,dict(seeds=results,metric_groups=10*3*11,independently_recomputed=True,
        prediction_manifest_sha256=F.sha(out/'prediction_freeze.json'),target_predictions_generated=False,audit_code_sha256=F.sha(__file__)))
    print('SEED_PAIRS_VERIFIED',number,330,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('number',type=int,choices=[34,35]);run(p.parse_args().number)
