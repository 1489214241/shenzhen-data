"""Record completed experiments, source ranking diagnosis and semantic audit."""
import collections
import numpy as np
import shenzhen_full_inputs as F
from shenzhen_exact_calibration import calibration_intervals


def main():
    root=F.REPO/'outputs/shenzhen_resolution_20261112'
    inputs=F.REPO/'data/深圳市/局部条件修复输入_20261112'
    folders=[(inputs,'manifest.json'),(root/'round04_reviewed_scopes','prediction_freeze.json'),
        (root/'round05_exact_source_calibration','prediction_freeze.json'),
        (root/'round03_partial_labels','probability_freeze.json')]
    integrity={}
    for folder,manifest in folders:
        files=F.read(folder/manifest)['files']
        for rel,h in files.items():assert F.sha(folder/rel)==h,(folder,rel)
        integrity[str(folder)]=dict(manifest_sha256=F.sha(folder/manifest),verified_files=len(files))
    base=F.REPO/'outputs/shenzhen_aligned_model_20261111'
    gold_hashes=F.read(base/'results.json')['gold_sha256']
    for d,h in gold_hashes.items():assert F.sha(F.SZ.DATA/d/'Gold/小区_学校Gold.json')==h
    old=root/'round03_partial_labels';c=F.read(old/'protocol.json');capacity={}
    for arm in c['arms']:
        loaded=[dict(np.load(old/'probabilities'/arm/f'{s}.npz')) for s in c['seeds']]
        p=[np.stack([r[d] for r in loaded]).mean(0) for d in c['sources']]
        y=[np.load(old/'inputs'/f'{d}_labels.npz')['valid_labels'] for d in c['sources']]
        intervals,sizes=calibration_intervals(p,y)
        attainable=collections.Counter('multi' if k>1 else 'single' for lo,hi,k in intervals)
        capacity[arm]=dict(source_rows=len(sizes),source_multi=int((sizes>1).sum()),attainable_raw_sets=dict(attainable),
            source_multi_unattainable=int((sizes>1).sum())-attainable['multi'],
            meaning='Per-row hypothetical threshold capacity of raw source probabilities with top1fallback, before any policy decoder. Not a deployable score or target-label-assisted method.')
    F.write(root/'round05_exact_source_calibration/source_ranking_diagnosis.json',capacity)
    rawpath=F.SZ.DATA/'原始资料/龙岗区/public_map_data.json';raw=F.read(rawpath)
    records=[{k:r[k] for k in ['ORG_ID','MC','XQ','GXSJ']}
        for r in raw['school_data'] if any(name in r['MC'] for name in ['贤义外国语','信义实验'])]
    audit=root/'scope_map_consistency_audit'
    F.write(audit/'official_group_and_visual_review.json',dict(raw_map=str(rawpath),raw_map_sha256=F.sha(rawpath),
        native_group_records=raw['groups'],selected_school_records=records,
        native_group_label_not_policy_classification=True,
        policy_classification='Official2026guide page3 calls both groups shared catchments; priority scope gives2.5additional points. Not university zones.',
        official_guide_url='https://www.lg.gov.cn/attachment/1/1722/1722313/12839768.pdf',
        visually_reviewed_pages=[3,9],rendered_page_sha256={p.name:F.sha(p) for p in audit.glob('official_page-*.png')},
        figure_visually_reviewed=True,figure_sha256=F.sha(audit/'贤义信义_原图路径与命中.png'),
        live_map_fetch=dict(url=raw['source_url'],requests_status=403,web_tool_page_readable=True,
            live_frontend_group_click_behavior_confirmed=False),
        conclusion='Native map-hit answers and the agreed shared-scope target are not equivalent for these38sites. Whether the paths encode priority zones requires frontend/source confirmation.',
        unmodified_gold=True,used_for_training=False))
    report=F.read(audit/'report.json');assert sum(r['unique_violating_sites'] for r in report['districts'].values())==63
    lines=['第四、五轮修复实验与文字—地图语义审计，2026-11-12','',
        '目标尚未解决；本轮没有替换默认模型，也没有修改原图Gold或评分分母。现用仍为龙华、龙岗。','',
        '一、本轮落地工作',
        '1. 新输入快照：data/深圳市/局部条件修复输入_20261112。所有202校重新扫描，仅4校观察发生变化。',
        '   水径、阳光只恢复括号外分支，括号内户籍条件仍未知。龙华两校整观湖街道范围有2026原表和2024官方方案依据。',
        '   地理范围不是入学资格。新实验只使用严格名称对应及独立行政面，不使用社区办公点推导新增事实。',
        '2. 第四轮：3个冻结模型版本×局部条件/街道范围/两者合用×10种子及集成。没有重新训练。',
        '3. 第五轮：保留第三轮20个模型，以来源区留出集完整搜索0—1阈值；同时预先规定单校/多校等权校准。',
        '   所有阈值先在来源区选择，全部深圳预测冻结后再评分。没有用深圳答案挑阈值或方法。','',
        '二、第四轮结果（与各自冻结版本比较）']
    r4=F.read(root/'round04_reviewed_scopes/results.json')['reports']
    for arm,components in r4.items():
        r=components['both']['ensemble']['全部'];a=r['after'];b=r['before']
        lines.append(f"{arm}: {b['correct']}→{a['correct']}/1761={a['exact_accuracy']:.2%}；改正/损害{r['corrected']}/{r['harmed']}；误报/漏报{a['fp']}/{a['fn']}；95%变化区间{r['delta_95_pp']}个百分点。")
    lines += ['局部条件的严格名称事实仅4条，原预测已经含这些学校，所以固定预测实验没有变化。',
        '街道范围减少84条地图口径漏报，同时新增29—30条地图口径误报，9个原先完整正确样本变错。未确认稳定改善，不采用。',
        '重建原生参照还会因多校冲突丢掉龙华两条旧参照；本轮补充过程保留旧预测，未把新部分集合当完整答案。','',
        '三、第五轮结果（与各自第三轮模型比较）']
    r5=F.read(root/'round05_exact_source_calibration/results.json')['reports']
    for arm,objectives in r5.items():
        cal=F.read(root/'round05_exact_source_calibration/source_calibration'/f'{arm}.json')
        for objective,reports in objectives.items():
            r=reports['ensemble']['全部'];a=r['after'];t=cal['selections'][objective]['threshold']
            lines.append(f"{arm}/{objective}: 阈值{t:.8f}；{a['correct']}/1761={a['exact_accuracy']:.2%}；改正/损害{r['corrected']}/{r['harmed']}；误报/漏报{a['fp']}/{a['fn']}；95%变化区间{r['delta_95_pp']}个百分点。")
    lines += ['完整微平均校准把阈值从0.90提高到约0.95，但深圳完整正确数没有增加；阈值上限不是当前低分的充分解释。',
        '来源区60个多校留出样本中，即使允许每例选自己的阈值，原始概率也只有15例（忽略未知）/20例（未知当负）可表达完整集合。',
        '这是解码前排序容量诊断；不能当部署准确率，不能概括所有多标签方法无效。它提示需要改进排序/多校监督，而非只调整一个全局阈值。','',
        '四、明确的新语义问题',
        '对已准入的76条直接文字范围关系全量检查：龙华19、龙岗44个小区的地图集合不满足这些关系，共63个。',
        '其中部分关系有优享条款或解析假设，不能把63个都直接改成政策Gold；原答案不变。',
        '最明确的一组：贤义外国语学校与信义实验小学。官方2026指南第3页明确称共享学区，并说明优享仅加2.5分；第9页两校基础范围相同、优享范围分东西两侧。',
        'https://www.lg.gov.cn/attachment/1/1722/1722313/12839768.pdf',
        '已目视核对第3、9页。原地图导出也有“贤义/信义”两校分组，并保留同一共享范围文字。',
        '但原图答案23个小区仅命中贤义、15个仅命中信义，共同命中0个。两条原始路径均存在自交，未擅自修复。',
        '因此，现有原图命中答案与“共享范围对应多校”的任务至少在这38例上不是同一种关系。',
        '地图路径可能表示优享范围，需要继续确认前端实际分组查询行为；不能仅凭图形推定。',
        '本机直接下载当前地图页面返回403；网页工具能读取页面，但尚未取得实时前端分组执行证据。原导出时间为2026-11-04。','',
        '五、验证与下一步',
        '71项相关测试通过，包括阈值区间算法与全部概率断点穷举结果核对、局部户籍条件隔离、旧正例保留和禁止大学区扩展。',
        '输入、冻结预测、原模型与概率哈希已复核，Gold哈希不变。所有评分过程已结束，无训练在运行。',
        '先完成共享/优享在原地图中的语义核验及全体不一致条款复核，保留原图评测单独报告；不通过改标签、删样本或选择最好种子制造提升。',
        '模型端下一步应围绕多校排序失败做受控实验，并用来源区单校/多校分别验证；来源区阈值校准的边界问题已排查。',
        '没有独立验证通过，目标保持进行中。']
    (root/'第四五轮结论与语义审计.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    F.write(root/'round04_05_completion_integrity.json',dict(verified_manifests=integrity,
        original_gold_hashes=gold_hashes,tests=dict(passed=71,failed=0),experiments_scored=True,
        audit_only_sources={str(p.relative_to(root)):F.sha(p) for p in audit.rglob('*') if p.is_file()},
        goal_solved=False,independent_validation=False,adopted=False))
    print('VERIFIED',integrity,flush=True)


if __name__=='__main__':main()
