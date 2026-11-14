"""Input-only school evidence inventory and independent local-feature replay.

No Gold, polygons or model correctness is read to choose evidence gaps.
"""
import collections
import json
from pathlib import Path
import numpy as np
import experiment_shenzhen_school_membership_trees as T


def run():
    T.check('inputs_sha256.json')
    dest=T.REPO.parent/'科研推进/深圳整体方法核查与学校成员预测_20261114'
    dest.mkdir(parents=True,exist_ok=True)
    result=[];summary={};inputs={};replayed=0
    for d in ['龙华区','龙岗区']:
        root=T.INPUT/d
        x=T.M.pkl(root/'routed.pkl')
        np.testing.assert_array_equal(T.local_features(x),np.load(T.OUT/'cache'/f'{d}.npz')['local'])
        replayed+=1
        meta=T.read(root/'metadata.json');ledger=T.read(root/'evidence_ledger.json')
        scopes=T.read(root/'full_scope_inventory.json');scope={s['school_id']:s for s in scopes}
        states=np.load(root/'whole_scope.npz')['states']
        assert set(np.unique(states)) <= {-1,0,1}
        lookup=collections.defaultdict(list)
        for e in ledger:lookup[e['site_id'],e['school_id']].append(e)
        rows=[]
        for s,school in enumerate(meta['schools']):
            sid=school['school_id'];refrows=[]
            for col in np.flatnonzero(x.ref_schools==s):
                i=int(x.ref_sites[col]);site=meta['sites'][i]
                evidence=[e for e in lookup[site['coordinate_group_id'],sid] if e['positive']]
                kind='textual_proof' if any(e['routes']['evidence_type']=='textual_proof' for e in evidence) else 'estimate' if evidence else 'untraced_in_ledger'
                refrows.append(dict(site_id=site['coordinate_group_id'],name=site['building_name'],kind=kind))
            count=collections.Counter(r['kind'] for r in refrows)
            sc=scope[sid]
            row=dict(district=d,school_id=sid,school_name=school['school_name'],
                     reference_records=len(refrows),reference_groups=len({r['site_id'] for r in refrows}),
                     textual_reference_records=count['textual_proof'],estimated_reference_records=count['estimate'],
                     untraced_reference_records=count['untraced_in_ledger'],references=refrows,
                     source_text=sc['source_text'],official_sources=sc['evidence'],scope_status=sc['scope_status'],
                     scope_state_counts={str(k):int((states[:,s]==k).sum()) for k in [-1,0,1]},
                     missing_reference=len(refrows)==0,missing_textual_reference=count['textual_proof']==0,
                     next_action='核验该校范围内的小区归属，记录原文出处和适用条件；同一小区逐校审查，部分正例不得标作完整集合。' if not count['textual_proof'] else '核验范围边缘、排除条款及其他学校是否同时适用；保持大学区原地段。')
            rows.append(row)
        summary[d]=dict(schools=len(rows),no_reference=sum(r['missing_reference'] for r in rows),
                        no_textual_reference=sum(r['missing_textual_reference'] for r in rows),
                        reference_records=sum(r['reference_records'] for r in rows),
                        reference_types=dict(collections.Counter(r['kind'] for row in rows for r in row['references'])),
                        all_site_scope_states={str(k):int((states==k).sum()) for k in [-1,0,1]})
        result.extend(rows)
        for name in ['metadata.json','routed.pkl','evidence_ledger.json','full_scope_inventory.json','whole_scope.npz']:
            p=root/name;inputs[str(p)]=T.sha(p)
    result.sort(key=lambda r:(not r['missing_reference'],not r['missing_textual_reference'],r['district'],r['school_name']))
    T.write(dest/'逐校输入缺口_不使用评价答案.json',dict(summary=summary,schools=result,not_for_training=True,
            target_gold_read=False,target_polygon_read=False,priority='zero reference, no textual reference, alphabetical; never sorted by target errors',
            source_sha256=inputs,local_features_replayed=replayed))
    lines=['深圳逐校输入缺口：只依据已有输入，不读取评价答案或学区polygon。',
           '学校顺序：无参照优先，其次无文字证据参照，再按区名、校名字典序。',
           '参照含估计；有正关系不等于掌握完整学校集合。这里没有新增核验事实。',
           '范围状态 -1=未知，0=估计范围外，1=估计范围内；均不自动视为行政归属证明。',
           json.dumps(summary,ensure_ascii=False,indent=2),'']
    for r in result:
        lines += [f"{r['district']} | {r['school_name']} | 参照{r['reference_records']}条，其中文字证据{r['textual_reference_records']}条",
                  '招生原文：'+r['source_text']]
        for e in r['official_sources']:lines.append(str(e.get('source_url',''))+' '+str(e.get('source_position','')))
        lines += ['待办：'+r['next_action'],'']
    (dest/'逐校输入缺口与官方出处.txt').write_text('\n'.join(lines),encoding='utf-8')
    T.write(T.OUT/'independent_local_feature_audit.json',dict(passed=True,targets_recomputed=replayed,
            no_target_gold_or_polygon_read=True,script_sha256=T.sha(Path(__file__)),source_sha256=inputs))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
