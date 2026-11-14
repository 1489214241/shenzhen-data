"""Post-score source-only threshold and logit-gradient diagnosis, no new fit."""
import numpy as np
import torch
import experiment_shenzhen_school_reference_mask as D
from shenzhen_zlpr import zlpr_rows

F=D.F; R=D.R; X=D.X


def main():
    F.guard(); c=D.verify(); R.check(D.OUT,'probability_freeze.json')
    dest=D.OUT/'diagnostics_not_training/loss_and_threshold.json'
    if dest.exists():raise FileExistsError('Preserve diagnosis')
    threshold={}
    for arm in c['arms']:
        for mode in c['decoders']:
            name=arm+'_'+mode; cal=F.read(D.OUT/'calibration'/f'{name}.json')
            feasible=[r for r in cal['candidates'] if r['single_correct']>=c['source_gate']['min_single_correct']]
            threshold[name]=dict(best_multi_with_single_floor=max((r['multi_correct'] for r in feasible),default=None),
                any_threshold_meets_count_gates=any(r['multi_correct']>=c['source_gate']['min_multi_correct'] for r in feasible),
                note='Diagnostic only. No replacement of the predeclared primary threshold.')
    xs,ys=X.load(c,'all_multi_training'); seed=c['seeds'][0]
    start=R.M.LinearRetrieval(next(iter(xs.values())))
    model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
    model.load_state_dict(torch.load(X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt',map_location=R.DEVICE,weights_only=False)['model']); model.eval()
    rows=[]
    for d in c['sources']:
        x=xs[d]; ids=ys[d]['train_ids']; z=ys[d]['labels'][ids]; ids=ids[(z>=0).all(1)]
        for lo in range(0,len(ids),64):
            q=ids[lo:lo+64]
            with torch.no_grad():s=model(R.M.batch(x,q))
            y=torch.tensor(ys[d]['labels'][q],device=R.DEVICE); w=torch.ones(len(q),device=R.DEVICE)
            g={}
            for name in ['bce_lsep','zlpr']:
                logits=s.detach().clone().requires_grad_()
                loss=R.masked_bce(logits,y,w)+R.masked_lsep(logits,y,w) if name=='bce_lsep' else zlpr_rows(logits,y).sum()
                gradient=torch.autograd.grad(loss,logits)[0]
                g[name]=(gradient.masked_fill(y!=0,0).sum(1).cpu().numpy(),-gradient.masked_fill(y!=1,0).sum(1).cpu().numpy())
            p=torch.sigmoid(s).cpu().numpy(); yy=y.cpu().numpy()
            for i,qi in enumerate(q):
                row=dict(d=d,q=int(qi),school_n=x.school_n,multi=bool((yy[i]==1).sum()>1))
                row.update({name:dict(negative=float(v[0][i]),positive=float(v[1][i])) for name,v in g.items()})
                row['bce_negative_probability_sum']=float(p[i,yy[i]==0].sum());rows.append(row)
        print('GRADIENT_CHECKED',d,len(ids),flush=True)
    groups={}
    for label,rr in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
        groups[label]=dict(n=len(rr),losses={})
        for loss in ['bce_lsep','zlpr']:
            n=np.array([r[loss]['negative'] for r in rr]);p=np.array([r[loss]['positive'] for r in rr])
            groups[label]['losses'][loss]=dict(negative_sum=float(n.sum()),positive_sum=float(p.sum()),
                ratio=float(n.sum()/p.sum()),negative_over_positive_rows=int((n>p).sum()),
                negative_quantiles=np.quantile(n,[0,.5,.9,1]).tolist(),positive_quantiles=np.quantile(p,[0,.5,.9,1]).tolist())
    F.write(dest,dict(threshold_tradeoff=threshold,gradient_groups=groups,rows=rows,
        checkpoint=str(X.OUT/'checkpoints'/f'all_multi_training_{seed}.pt'),seed=seed,
        gradient='Unweighted per-row logit gradients at a fixed completed source model, not parameter gradients or proof of causal harm.',
        used_for_new_model_selection=False,target_gold_read=False,target_polygon_read=False,code_sha256=F.sha(__file__)))
    print('THRESHOLD_DIAGNOSIS',threshold,flush=True)
    print('GRADIENT_ALL',groups['all'],'MULTI',groups['multi'],flush=True)


if __name__=='__main__':main()
