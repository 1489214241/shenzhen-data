"""Round34: source-trained stopping boundary with frozen school ranking."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
import experiment_shenzhen_explicit_negative_training as E
from shenzhen_adaptive_threshold import query_features, at_loss, ThresholdHead, choose
from experiment_shenzhen_relative_decode import relative, select
from zoning.evaluate import bootstrap

F,R,S=E.F,E.R,E.S
OUT=R.ROOT/'round34_adaptive_threshold'
ARMS=['constant_head','adaptive_head']


def prepare():
    F.guard();old=E.verify();R.check(E.OUT,'probability_freeze.json')
    if OUT.exists():raise FileExistsError('Preserve round34')
    c={k:old[k] for k in ['sources','targets','seeds','steps','lr','reg','width','chunk','source_gate']}
    c.update(arms=ARMS,base='round32/features_labels frozen scorer, no continued scorer training',
        training='Complete source training rows only; retain original weights; no partial labels used to learn stopping boundary.',
        input='35d: softmax-weighted33d anchor/context, normalized entropy, top2 logit gap. Train-only weighted mean/std.',
        initializer='TH=max(school_logits)-1 for both heads; adaptive residual initially zero.',
        loss='ATLOP virtual threshold two categorical terms, all observed positive/negative classes, fixed school logits.',
        primary='adaptive_head: mean per-seed school logit minus mean learned TH >0; no validation threshold fitting; empty falls back to argmax.',
        controls='Same-budget constant TH offset head and original round32 features_labels relative probability decoder.',
        source_gate_rule='Original single/multi/attainable gate AND balanced_exact >= both controls AND each stress exact >= both controls; only adaptive_head may advance.',
        checkpoint='40 steps, no early stopping or seed selection, Adam lr0.003, reg0.01, gradient clip5.',
        complete_population='Original1246source development validation; no target inference before pass. Later target trial must include all cumulative facts identically for controls.',
        target_gold_read=False,target_polygon_read=False,target_cardinality_used=False,
        target_predictions_generated=False,independent_validation=False,adopted=False)
    F.write(OUT/'protocol.json',c)
    links=dict(F.read(E.OUT/'input_links.json')['files'])
    for p in [E.OUT/'probability_freeze.json',E.OUT/'source_report.json',R.ROOT/'round32_independent_completion_audit.json']:
        links[str(p)]=F.sha(p)
    for rel,h in F.read(E.OUT/'probability_freeze.json')['files'].items():links[str(E.OUT/rel)]=h
    for d in c['sources']:
        p=E.OUT/'weights/features_labels'/f'{d}.npz';links[str(p)]=F.sha(p)
    for p in [Path(__file__),F.REPO/'scripts/shenzhen_adaptive_threshold.py',F.REPO/'tests/test_shenzhen_adaptive_threshold.py']:
        dest=OUT/'code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);links[str(p)]=F.sha(p)
    for p in (R.ROOT/'adaptive_threshold_method_review').iterdir():
        if p.is_file():links[str(p)]=F.sha(p)
    F.write(OUT/'input_links.json',dict(files=links))
    F.write(OUT/'input_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('PREPARED',flush=True)


def verify():
    R.check(OUT,'input_freeze.json')
    for p,h in F.read(OUT/'input_links.json')['files'].items():assert F.sha(p)==h,p
    return F.read(OUT/'protocol.json')


def scorer(xs,c,seed):
    start=R.M.LinearRetrieval(next(iter(xs.values())))
    model=R.BernoulliRetrieval(41,27,start,seed,c['width']).to(R.DEVICE)
    saved=torch.load(E.OUT/'checkpoints'/f'features_labels_{seed}.pt',map_location=R.DEVICE,weights_only=False)
    model.load_state_dict(saved['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def extract(model,x,ids,chunk):
    zs=[];fs=[];ps=[]
    with torch.no_grad():
        for lo in range(0,len(ids),chunk):
            t=R.M.batch(x,ids[lo:lo+chunk]);z=model(t)
            f=query_features(z,t['ax'])
            zs.append(z.cpu().numpy());fs.append(f.cpu().numpy());ps.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(zs),np.concatenate(fs),np.concatenate(ps)


def cache():
    F.guard();c=verify()
    if (OUT/'cache_freeze.json').exists():R.check(OUT,'cache_freeze.json');return
    xs,ys=E.load(c,'features_labels');audit={}
    for seed in c['seeds']:
        dest=OUT/'cache'/f'{seed}.npz'
        if dest.exists():raise FileExistsError('Unfrozen cache must be inspected before continuation')
        model=scorer(xs,c,seed);bank={};old=dict(np.load(E.OUT/'probabilities/features_labels'/f'{seed}.npz'))
        for d in c['sources']:
            x=xs[d];y=ys[d];ids=y['train_ids'];ids=ids[(y['labels'][ids]>=0).all(1)]
            assert not set(x.groups[ids])&set(x.groups[y['valid_ids']])
            np.testing.assert_array_equal(x.self_mask,x.groups[:,None]==x.groups[x.ref_sites][None])
            z,f,_=extract(model,x,ids,c['chunk'])
            bank[d+'_train_z']=z;bank[d+'_train_f']=f;bank[d+'_train_ids']=ids
            bank[d+'_train_y']=y['labels'][ids]
            bank[d+'_train_w']=np.load(E.OUT/'weights/features_labels'/f'{d}.npz')['weights'][ids]
            for view in ['valid','stress_0','stress_1']:
                vx=x if view=='valid' else S.view(x,d,view)
                z,f,p=extract(model,vx,y['valid_ids'],c['chunk'])
                np.testing.assert_array_equal(p,old[d if view=='valid' else d+'_'+view])
                key=d+'_'+view;bank[key+'_z']=z;bank[key+'_f']=f
            audit[d]=dict(complete_training_rows=len(ids),multi_training_rows=int((y['labels'][ids].sum(1)>1).sum()),valid_rows=len(y['valid_ids']))
        ff=np.concatenate([bank[d+'_train_f'] for d in c['sources']]).astype(np.float64)
        ww=np.concatenate([bank[d+'_train_w'] for d in c['sources']]).astype(np.float64)
        mu=np.average(ff,axis=0,weights=ww)
        sd=np.sqrt(np.average((ff-mu)**2,axis=0,weights=ww)).clip(1e-3)
        bank['mean']=mu.astype(np.float32);bank['scale']=sd.astype(np.float32)
        assert all(np.isfinite(v).all() for v in bank.values())
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**bank)
        print('CACHED',seed,flush=True)
    F.write(OUT/'cache_audit.json',dict(districts=audit,original_source_probabilities_bitwise_equal=True,
        normalization_training_only=True,own_property_mask_verified=True,target_predictions_generated=False))
    F.write(OUT/'cache_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for p in (OUT/'cache').glob('*.npz')}))


def train():
    F.guard();c=verify();R.check(OUT,'cache_freeze.json')
    if (OUT/'prediction_freeze.json').exists():R.check(OUT,'prediction_freeze.json');return
    history=F.read(OUT/'training_history.json') if (OUT/'training_history.json').exists() else {}
    for seed in c['seeds']:
        bank=dict(np.load(OUT/'cache'/f'{seed}.npz'))
        frames=[]
        for d in c['sources']:
            frames.append([torch.from_numpy(bank[d+'_train_'+k]).to(R.DEVICE) for k in ['z','f','y','w']])
        for arm in ARMS:
            key=f'{arm}/{seed}';ck=OUT/'checkpoints'/f'{arm}_{seed}.pt';dest=OUT/'margins'/arm/f'{seed}.npz'
            if key in history:
                assert F.sha(ck)==history[key]['checkpoint_sha256'] and F.sha(dest)==history[key]['margins_sha256'];continue
            head=ThresholdHead(bank['mean'],bank['scale'],arm=='adaptive_head',seed,c['width']).to(R.DEVICE)
            opt=torch.optim.Adam(head.parameters(),lr=c['lr']);losses=[]
            for step in range(c['steps']):
                opt.zero_grad();penalty=head.penalty(c['reg']);total=penalty
                for z,f,y,w in frames:total=total+at_loss(z,head(z,f),y,w)
                assert torch.isfinite(total);total.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5));opt.step()
                losses.append(dict(loss=float(total.detach()),penalty=float(penalty.detach()),gradient_norm=norm))
            payload={}
            with torch.no_grad():
                for d in c['sources']:
                    for view in ['train','valid','stress_0','stress_1']:
                        key2=d+'_'+view;z=bank[key2+'_z'];f=bank[key2+'_f']
                        th=head(torch.from_numpy(z).to(R.DEVICE),torch.from_numpy(f).to(R.DEVICE)).cpu().numpy()
                        payload[key2+'_threshold']=th
                        payload[key2]=z.astype(np.float64)-th.astype(np.float64)[:,None]
            ck.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=head.state_dict(),step=c['steps'],history=losses,protocol_hash=F.sha(OUT/'protocol.json')),ck)
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            history[key]=dict(checkpoint_sha256=F.sha(ck),margins_sha256=F.sha(dest),parameters=sum(p.numel() for p in head.parameters()),losses=losses)
            F.write(OUT/'training_history.json',history);print('TRAINED',key,losses[-1],flush=True)
    F.write(OUT/'prediction_freeze.json',dict(files={str(p.relative_to(OUT)):F.sha(p) for folder in ['checkpoints','margins'] for p in (OUT/folder).rglob('*') if p.is_file()},target_predictions_generated=False))


def count(ps,ys):
    out={k:0 for k in ['n','correct','single_correct','multi_correct','fp','fn']}
    for p,y in zip(ps,ys):
        y=y.astype(bool);correct=(p==y).all(1);multi=y.sum(1)>1
        for k,v in dict(n=len(y),correct=correct.sum(),single_correct=(correct&~multi).sum(),multi_correct=(correct&multi).sum(),fp=(p&~y).sum(),fn=(~p&y).sum()).items():out[k]+=int(v)
    return out


def report():
    F.guard();c=verify();R.check(OUT,'prediction_freeze.json')
    if (OUT/'source_report.json').exists():raise FileExistsError('Preserve report')
    _,ys=E.load(c,'features_labels');labels=[ys[d]['valid_labels'] for d in c['sources']]
    original=F.read(E.OUT/'source_report.json')['reports']['features_labels_relative']
    threshold=original['selections']['balanced_exact']['threshold']
    all_pred={};reports={};seed_stats={};means={}
    for arm in ['original_relative']+ARMS:
        payloads=[dict(np.load(E.OUT/'probabilities/features_labels'/f'{s}.npz' if arm=='original_relative' else OUT/'margins'/arm/f'{s}.npz')) for s in c['seeds']]
        mean={k:np.stack([p[k] for p in payloads]).mean(0) for k in payloads[0]};means[arm]=mean
        def predictions(p,view):
            return [select(relative(p[d if view=='valid' else d+'_'+view]),threshold) if arm=='original_relative' else choose(p[d+'_'+view]) for d in c['sources']]
        pred={v:predictions(mean,v) for v in ['valid','stress_0','stress_1']}
        full=count(pred['valid'],labels);balanced=full['single_correct']/1186/2+full['multi_correct']/60/2
        reports[arm]=dict(full=full,balanced_exact=balanced,stress={v:count(pred[v],labels) for v in ['stress_0','stress_1']})
        all_pred[arm]=pred['valid'];seed_stats[arm]={}
        for seed,p in zip(c['seeds'],payloads):
            preds=predictions(p,'valid');seed_stats[arm][str(seed)]=dict(all=count(preds,labels),districts={d:count([pr],[y]) for d,pr,y in zip(c['sources'],preds,labels)})
        print('SOURCE',arm,reports[arm],flush=True)
    assert reports['original_relative']['full']==original['full']
    rows=[]
    for di,d in enumerate(c['sources']):
        xy=np.load(F.REPO/'data'/d/'arrays.npz')['coords']
        for i,q in enumerate(ys[d]['valid_ids']):
            y=labels[di][i].astype(bool)
            rows.append(dict(d=d,q=int(q),block=str(tuple(int(v//1000) for v in xy[q])),multi=bool(y.sum()>1),gold=np.flatnonzero(y).tolist(),
                predicted={a:np.flatnonzero(p[di][i]).tolist() for a,p in all_pred.items()},correct={a:bool((p[di][i]==y).all()) for a,p in all_pred.items()}))
    paired={}
    for a,b in [('adaptive_head','original_relative'),('adaptive_head','constant_head'),('constant_head','original_relative')]:
        paired[a+'_vs_'+b]={}
        for name,rs in [('all',rows),('single',[r for r in rows if not r['multi']]),('multi',[r for r in rows if r['multi']])]+[(d,[r for r in rows if r['d']==d]) for d in c['sources']]:
            delta=[int(r['correct'][a])-int(r['correct'][b]) for r in rs]
            paired[a+'_vs_'+b][name]=dict(n=len(rs),before_correct=sum(r['correct'][b] for r in rs),after_correct=sum(r['correct'][a] for r in rs),corrected=delta.count(1),harmed=delta.count(-1),delta_pp=float(np.mean(delta)*100),delta_95_pp=bootstrap(rs,delta,seed=20261104,draws=3000))
    attainable=0
    for d,y in zip(c['sources'],labels):
        z=means['adaptive_head'][d+'_valid'];y=y.astype(bool);multi=y.sum(1)>1
        attainable+=int(((np.where(y,z,np.inf).min(1)>np.where(~y,z,-np.inf).max(1))&multi).sum())
    a=reports['adaptive_head'];g=c['source_gate']
    checks=dict(single=a['full']['single_correct']>=g['min_single_correct'],multi=a['full']['multi_correct']>=g['min_multi_correct'],attainable=attainable>=g['min_attainable_multi'])
    for b in ['original_relative','constant_head']:
        checks['balanced_vs_'+b]=a['balanced_exact']>=reports[b]['balanced_exact']
        for v in ['stress_0','stress_1']:checks[v+'_vs_'+b]=a['stress'][v]['correct']>=reports[b]['stress'][v]['correct']
    F.write(OUT/'paired_source_rows.json',rows);F.write(OUT/'source_seed_metrics.json',seed_stats)
    F.write(OUT/'source_report.json',dict(reports=reports,paired=paired,checks=checks,attainable_multi=attainable,
        selected_target_candidate='adaptive_head' if all(checks.values()) else None,threshold_fitted_on_validation=False,target_predictions_generated=False,adopted=False,independent_validation=False))
    print('TARGET_GATE',all(checks.values()),checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache','train','report']);globals()[p.parse_args().phase]()
