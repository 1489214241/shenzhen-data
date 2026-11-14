"""Frozen school-content readout pilot. Validation screen before target scoring."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
import run_full_evidence as base
import run_modernnca_interaction as common
from run_modernnca_relation import tasks,device_batch,predict,digest,save,cpu_task
from run_modernnca_support import val_task,training_audit
from zoning import DEVICE,evaluate,train
from zoning.model import RetrievalScorer
from zoning.modernnca_readout import ReadoutModernNCA

OUT=REPO/'outputs/modernnca_readout_20261113'
DOC=REPO.parent/'科研推进/ModernNCA学校证据读出尝试_20261113'
METHODS=['identity','summary','content','reference']
CHECKPOINTS=[20,40,80,160]
PAIRS=[('content','identity'),('content','summary'),('content','reference'),('summary','identity'),('reference','identity')]


def make(kind,x,start,seed):
    if kind=='reference':
        if start is None: start=train.warm_start(x)
        return RetrievalScorer(x.np_,x.na,start,seed,32).to(DEVICE)
    return ReadoutModernNCA(x.np_,x.na,seed,kind).to(DEVICE)


def configure_common():
    common.make=make
    common.CHECKPOINTS=CHECKPOINTS


def freeze():
    old=json.loads((REPO/'outputs/modernnca_interaction_20261113/frozen.json').read_text(encoding='utf8'))
    for p,h in old['source_hashes'].items(): assert digest(REPO/p)==h,p
    paths=list(dict.fromkeys([REPO/p for p in old['source_hashes']]+[Path(__file__),
        REPO/'zoning/modernnca_readout.py',REPO/'tests/test_modernnca_readout.py',
        REPO/'scripts/audit_modernnca_interaction_targets.py']))
    plan=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),methods=METHODS,grid=base.GRID,
        checkpoints=CHECKPOINTS,seeds=base.SEEDS,batches=base.BATCHES,validation=base.VALIDATION,
        input_hashes=old['input_hashes'],source_hashes={str(p.relative_to(REPO)):digest(p) for p in paths},
        architecture='Original LN encoder/query/Euclidean votes unchanged. content: shared residual school logits log(p)+MLP(ax, seven scalar summaries, vote-weighted reference-only raw px mean/variance and embedding mean32). summary: same common features, omits only new content pooling path. Each head ~10000 parameters, rounded hidden width. Both zero initialized, no dropout, preserve common encoder RNG. identity original LN. reference existing RetrievalScorer with original 100-step local linear warm start.',
        scalar_order=['log school probability','reference vote share','anchor vote share','log1p valid project-weighted reference count','no reference indicator','within-school vote-weighted learned distance mean','distance variance'],
        input='Same full legal references, all px/ax evidence, group mass, membership, candidates, masks and training labels. No SNS/top-k, no new labels or manual reliability flags. Unknown/policy channels only as existing inputs encode them. Summary ablation still receives all evidence via original encoder. Reference training includes source-district genuinely rule-uncovered labeled queries already.',
        training='All four models fresh: Adam, clip5, batch64, same task rows/labels/weights and draws. 4 lr/reg choices and fixed cap160 for all. Reference retains original local linear warm start, separately disclosed; LN variants none. Validation district excluded entirely from training. K1 inherits K0 selection.',
        selection='K0 validation accuracy then NLL then earlier step. 8 target-validation scenarios, six distinct validation districts. No extension or post-score retuning.',
        gate='Primary content must improve over LN in BOTH cities and >=5/8 scenarios AND beat summary and reference in pooled validation accuracy. If not, stop this declared candidate after validation, no target evaluation. Screening is not significance. Controls cannot trigger target expansion.',
        confirmation='If content passes, freeze all32 selections, all four methods on K0 and three K1 batches with three seeds; predict all before scoring. Five declared pair contrasts use within-batch Bonferroni99% spatial-block CI; content must beat LN,summary,reference in all K1 draws with no city95% confirmed harm. K0 separate. No automatic adoption.',
        diagnostic='Replay archived LN on validation only before fits; stratify errors by zero true-school references and low/high top-two probability margin. Observational hypothesis check only, not proof of information-loss mechanism. No target truth used for gate.',
        audits='Three unit tests on active heads, masks, labels, permutations, batch independence, zero/one refs, pooling and RNG; all128 trial artifacts, all32 selected replay; LN32 trials/128 checkpoints exact archive reproduction; if target run all384 seed artifacts +128 replay.',
        status='pre_v1 repeatedly reused development data, Hangzhou Gold unfinished, not independent blind validation. Single chosen content architecture. No literature reproduction or new theoretical expressivity claim.')
    plan=json.loads(json.dumps(plan));f=OUT/'frozen.json'
    if f.exists():
        prev=json.loads(f.read_text(encoding='utf8'));plan['created']=prev['created'];assert plan==prev,'Frozen sources changed'
    else:
        save(f,plan);save(DOC/'方案.json',plan)
        for p in paths:
            dst=OUT/'code'/p.relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    return plan


def diagnose():
    if (OUT/'diagnosis.json').exists():return
    _,D,R,X=base.loaded(0,1);rows=[]
    archive=REPO/'outputs/modernnca_interaction_20261113/validation/identity'
    for d in D:
        v=base.VALIDATION[d];t=val_task(d,D,R,X);m=make('identity',X[d],None,base.SEEDS[0])
        m.load_state_dict(torch.load(archive/f'{d}.pt',map_location=DEVICE,weights_only=True))
        P=predict(m,t);saved=np.load(archive/f'{d}.npz')
        assert np.array_equal(saved['ids'],t['ids']) and np.allclose(P,saved['P'],atol=3e-6)
        for st in range(0,len(P),128):
            b=device_batch(t,slice(st,st+128));mass=b['valid']*b['gb'].exp()[None]
            counts=(mass@b['mem'][:mass.shape[1]]).cpu().numpy()
            for j,y in enumerate(b['y'].tolist()):
                i=st+j;top=np.sort(P[i]);rows.append(dict(target=d,validation=v,city=D[v].city,
                    q=int(t['ids'][i]),truth=y,pred=int(P[i].argmax()),true_ref_mass=float(counts[j,y]),
                    pred_ref_mass=float(counts[j,P[i].argmax()]),margin=float(top[-1]-top[-2])))
        del m,t;gc.collect();torch.cuda.empty_cache()
    groups={}
    for key,rr in [('all',rows),('zero_true_ref',[r for r in rows if r['true_ref_mass']==0]),
                   ('with_true_ref',[r for r in rows if r['true_ref_mass']>0]),
                   ('margin_below_0.2',[r for r in rows if r['margin']<.2]),
                   ('margin_at_least_0.2',[r for r in rows if r['margin']>=.2])]:
        groups[key]=dict(n=len(rr),errors=sum(r['truth']!=r['pred'] for r in rr))
    save(OUT/'diagnosis.json',dict(groups=groups,rows=rows,note='Validation only; descriptive, no causal diagnosis.'))
    print('DIAGNOSIS',groups,flush=True)


def select():
    configure_common();_,D,R,X=base.loaded(0,1)
    for d in D:
        tr=tasks(d,D,R,X,exclude={base.VALIDATION[d]});val=val_task(d,D,R,X)
        audit=training_audit(tr);assert base.VALIDATION[d] not in audit
        assert audit[d]['ids']==sorted(int(q) for q in X[d].supervised)
        save(OUT/'training_audit'/f'{d}.json',audit)
        for kind in METHODS:
            dest=OUT/'selection'/f'{d}_{kind}.json'
            if dest.exists():continue
            trials=[]
            for i,hp in enumerate(base.GRID):
                path=OUT/'trials'/f'{d}_{kind}_{i}.json'
                prior=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
                if prior.get('complete'):
                    assert digest(path.with_suffix('.pt'))==prior['state_hash']
                    assert digest(path.with_suffix('.npz'))==prior['probability_hash']
                    info=prior
                else:info=common.fit_trial(kind,hp,d,X,tr,val,path)
                trials.append(info)
            i=max(range(len(trials)),key=lambda j:common.rank(trials[j]['best']));chosen=trials[i]
            src=OUT/'trials'/f'{d}_{kind}_{i}';dst=OUT/'validation'/kind/d;dst.parent.mkdir(parents=True,exist_ok=True)
            for ext in ['.pt','.npz']:dst.with_suffix(ext).write_bytes(src.with_suffix(ext).read_bytes())
            save(dest,dict(hp=chosen['hp'],steps=chosen['best']['step'],validation=base.VALIDATION[d],best=chosen['best'],trial=i,trials=trials))
            print('SELECTED',d,kind,chosen['best'],flush=True)
        del tr,val;gc.collect();torch.cuda.empty_cache()


def gate_and_audit():
    _,D,R,X=base.loaded(0,1);rows=[];replays={};maxerr=0.;curves=0
    for d in D:
        v=base.VALIDATION[d];t=val_task(d,D,R,X);pp={}
        for kind in METHODS:
            for i in range(len(base.GRID)):
                path=OUT/'trials'/f'{d}_{kind}_{i}.json';info=json.loads(path.read_text(encoding='utf8'))
                assert info['complete'] and digest(path.with_suffix('.pt'))==info['state_hash']
                assert digest(path.with_suffix('.npz'))==info['probability_hash']
                if kind=='identity':
                    old=json.loads((REPO/'outputs/modernnca_interaction_20261113/trials'/path.name).read_text(encoding='utf8'))
                    for a,b in zip(info['history'],old['history']):
                        assert all(a[k]==b[k] for k in ['step','accuracy','nll']), (d,i,a,b)
                        curves+=1
            f=OUT/'validation'/kind/d;m=make(kind,X[d],None,base.SEEDS[0])
            m.load_state_dict(torch.load(f.with_suffix('.pt'),map_location=DEVICE,weights_only=True))
            P=predict(m,t);saved=np.load(f.with_suffix('.npz'))
            assert np.array_equal(saved['ids'],t['ids']);err=float(np.abs(P-saved['P']).max());assert err<3e-6
            maxerr=max(maxerr,err);pp[kind]=P
            choice=json.loads((OUT/'selection'/f'{d}_{kind}.json').read_text(encoding='utf8'))
            assert float((P.argmax(1)==t['y'].numpy()).mean())==choice['best']['accuracy']
            replays[f'{d}_{kind}']=dict(max_error=err,parameters=sum(p.numel() for p in m.parameters()),
                added_parameters=sum(p.numel() for p in m.head.parameters()) if hasattr(m,'head') else 0,
                steps=choice['steps'],state_hash=digest(f.with_suffix('.pt')),probability_hash=digest(f.with_suffix('.npz')))
            del m
        for i,q in enumerate(t['ids']):rows.append(dict(target=d,validation=v,d=v,city=D[v].city,q=int(q),
            truth=int(t['y'][i]),block=evaluate.block(D[v],int(q)),predictions={k:int(P[i].argmax()) for k,P in pp.items()}))
        del t;gc.collect();torch.cuda.empty_cache()
    assert curves==128
    save(OUT/'validation_cases.json',rows)
    subsets=[('两城',rows)]+[(c,[r for r in rows if r['city']==c]) for c in ['长沙','杭州']]+[(d,[r for r in rows if r['target']==d]) for d in D]
    report={}
    for a,b in PAIRS:
        z={name:common.comparison(rr,a,b) for name,rr in subsets}
        delta=[int(r['predictions'][a]==r['truth'])-int(r['predictions'][b]==r['truth']) for r in rows]
        z['descriptive_postselection_ci']=common.interval(rows,delta,.01);report[f'{a}-{b}']=z
    z=report['content-identity'];positive=sum(z[d]['delta']>0 for d in D)
    yes=positive>=5 and all(z[c]['delta']>0 for c in ['长沙','杭州']) and all(report[f'content-{k}']['两城']['delta']>0 for k in ['summary','reference'])
    verdict=dict(promoted=bool(yes),positive_scenarios=positive,city_deltas={c:z[c]['delta'] for c in ['长沙','杭州']},
        control_deltas={k:report[f'content-{k}']['两城']['delta'] for k in ['summary','reference']})
    save(OUT/'gate.json',verdict)
    save(DOC/'验证结果.json',dict(comparisons=report,verdict=verdict))
    save(DOC/'完整性审计.json',dict(trial_artifacts=128,selected_replays=32,max_error=maxerr,archive_identical_checkpoints=curves,models=replays))
    save(OUT/'sealed_selection.json',{str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'selection').glob('*.json'))})
    print('GATE',json.dumps(verdict,ensure_ascii=False),flush=True)
    return verdict


def confirm(verdict):
    seal=json.loads((OUT/'sealed_selection.json').read_text(encoding='utf8'))
    assert len(seal)==32 and all(digest(OUT/p)==h for p,h in seal.items())
    base.OUT=OUT;base.make=make;base.METHODS=METHODS
    for k,s in base.BATCHES:base.run(k,s)
    results={}
    for k,s in base.BATCHES:
        _,D,R,X=base.loaded(k,s);folder=OUT/f'k{k}_s{s}';scored={};whole={}
        hashes=json.loads((folder/'prediction_hashes.json').read_text(encoding='utf8'))
        assert len(hashes)==32 and all(digest(folder/p)==h for p,h in hashes.items())
        for kind in METHODS:
            scored[kind]=[];whole[kind]=[]
            for d in D:
                P=np.load(folder/kind/'probs'/f'{d}.npz')['probs'].mean(0);er=evaluate.eval_rows(D[d],R[d].known)
                scored[kind]+=[dict(r,q=e['q'],truth=e['truth']) for r,e in zip(evaluate.correct(P,D[d],er),er)]
                whole[kind]+=[r for q,r in zip(sorted(D[d].answers),evaluate.overall(P,D[d],R[d])) if R[d].kind.get(q)!='verified']
        contrasts={}
        for a,b in PAIRS:
            z={}
            for city in ['两城','长沙','杭州']+list(D):
                def keep(r):return city=='两城' or r['city']==city or r['d']==city
                aa=[r for r in scored[a] if keep(r)];bb=[r for r in scored[b] if keep(r)]
                assert [(r['d'],r['q'],r['truth'],r['block']) for r in aa]==[(r['d'],r['q'],r['truth'],r['block']) for r in bb]
                delta=[int(x['ok'])-int(y['ok']) for x,y in zip(aa,bb)]
                z[city]=dict(n=len(aa),delta=float(np.mean(delta)*100),ci=common.interval(aa,delta,.01 if city=='两城' else .05),fixed=sum(x==1 for x in delta),broken=sum(x==-1 for x in delta))
            contrasts[f'{a}-{b}']=z
        results[f'k{k}_s{s}']=dict(model={m:evaluate.summary(rr) for m,rr in scored.items()},overall_excluding_verified={m:evaluate.summary(rr) for m,rr in whole.items()},contrasts=contrasts)
        save(DOC/'目标结果.json',results)
    pairs=['content-identity','content-summary','content-reference']
    decisions=dict(confirmed=bool(verdict['promoted'] and all(results[f'k1_s{s}']['contrasts'][p]['两城']['ci'][0]>0 for s in [1,2,3] for p in pairs)
        and not any(results[f'k1_s{s}']['contrasts'][p][city]['ci'][1]<0 for s in [1,2,3] for p in pairs for city in ['长沙','杭州'])))
    save(DOC/'目标判定.json',decisions)


def report():
    z=json.loads((DOC/'验证结果.json').read_text(encoding='utf8'));gate=z['verdict'];audit=json.loads((DOC/'完整性审计.json').read_text(encoding='utf8'))
    labels={'identity':'原ModernNCA-LN','summary':'摘要评分头','content':'证据内容评分头','reference':'RetrievalScorer'}
    target=(DOC/'目标判定.json').exists()
    verdict=json.loads((DOC/'目标判定.json').read_text(encoding='utf8')) if target else None
    lines=['ModernNCA学校证据读出：研究判断','',
        ('目标开发评价完成；确认提升。仍需独立地区/时段验证。' if verdict['confirmed'] else '目标开发评价完成；未确认提升，不替换研究基线。') if target else
        ('验证筛查通过，目标评价尚未完成。' if gate['promoted'] else '证据内容评分头未通过事先冻结的验证筛查；停止本具体结构的目标扩跑，不替换研究基线。'),
        '', '方法：保持原LN逐条编码、查询生成、欧氏距离和投票；新增共享学校评分log(p)+delta。',
        '内容头读取按原投票加权的参照原始证据均值/方差、编码均值，以及共同的学校直接证据、周边统计、票数/参照数量/距离摘要。',
        '摘要头保留共同信息，不接新增内容汇总；仍通过原编码器使用全部逐条证据。两头约1万新增参数，末层零初始化。',
        '这与上一轮编码前/距离前交互不同；不是TabR原版复现，也未证明原读出理论上无法表达证据组合。',
        '', '验证：8情景、6个不同验证区。每方法4组学习率/正则、20/40/80/160步，统一固定上限。共128次从头训练。',
        '训练题包含源区真实规则未判但有合法标签的样本；目标区仅允许的监督题。验证区完全排除训练。',
        'RetrievalScorer保留既有100步本区线性热启动；其他方法没有热启动。所有方法共享输入、训练题和选参规则。', '']
    for a,b in PAIRS:
        v=z['comparisons'][f'{a}-{b}'];c=v['两城']
        lines += [f"{labels[a]} - {labels[b]}: {c['accuracy_a']:.3f}% vs {c['accuracy_b']:.3f}%；差{c['delta']:+.3f}点；改对/误伤{c['fixed']}/{c['broken']}；99%选择后描述区间{v['descriptive_postselection_ci']}",
                  f"  长沙{v['长沙']['delta']:+.3f}点；杭州{v['杭州']['delta']:+.3f}点。"]
    lines+=['',f'筛查：{json.dumps(gate,ensure_ascii=False)}','筛查仅用于晋级，验证数据已用于选参，不能把分数或区间当成独立提升证明。','',
        f"审计：{audit['trial_artifacts']}组训练产物哈希核对；{audit['selected_replays']}组所选模型重放，最大误差{audit['max_error']}；原LN128个检查点准确率和NLL与上轮完全一致。",'参数与到上限情况：']
    for k in METHODS:
        mm=[v for n,v in audit['models'].items() if n.endswith('_'+k)]
        lines.append(f"{labels[k]}: 总参数{mm[0]['parameters']}，新增头参数{mm[0]['added_parameters']}，选择160步{sum(v['steps']==160 for v in mm)}/8。")
    if target:
        result=json.loads((DOC/'目标结果.json').read_text(encoding='utf8'))
        for batch,r in result.items():
            lines+=['',batch]+[f"{labels[k]} 模型{r['model'][k]['两城']['accuracy']:.3f}%；规则+模型{r['overall_excluding_verified'][k]['两城']['accuracy']:.3f}%" for k in METHODS]
            for p in ['content-identity','content-summary','content-reference']:lines.append(p+': '+json.dumps(r['contrasts'][p],ensure_ascii=False))
        if (DOC/'目标完整性审计.json').exists():
            ta=json.loads((DOC/'目标完整性审计.json').read_text(encoding='utf8'))
            lines.append(f"目标产物审计：{ta['artifacts']}组；重放{ta['replayed']}组；最大误差{ta['max_error']}。")
    lines+=['','边界：pre_v1重复使用的开发数据，杭州Gold未冻结；只测这个固定结构和160步预算，不代表全部内容读出无效。',
        '验证筛查失败不等于证明方法无效。若出现分城反向，不能选择性保留有利城市后宣称整体改进。',
        '默认流程、原基线和数据未修改。配置/种子/逐题概率/权重与训练题审计在独立outputs/modernnca_readout_20261113目录。']
    (DOC/'研究判断.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','diagnose','select','report','all']);args=parser.parse_args()
    freeze();configure_common()
    if args.phase in ['diagnose','all']:diagnose()
    if args.phase in ['select','all']:select()
    if args.phase in ['report','all']:
        verdict=gate_and_audit()
        if args.phase=='all':
            if verdict['promoted']:confirm(verdict)
            save(OUT/'completed.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),verdict=verdict,target_evaluated=verdict['promoted'],frozen_sha256=digest(OUT/'frozen.json')))
            if verdict['promoted']:
                import audit_modernnca_interaction_targets as target_audit
                target_audit.run=sys.modules[__name__]
                target_audit.main()
        report()


if __name__=='__main__':main()
