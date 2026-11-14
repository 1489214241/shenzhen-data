"""Post-score diagnosis only. Never imported by input/training/prediction code."""
import collections
import pickle
from pathlib import Path
import shenzhen_full_inputs as F
from zoning import shenzhen_policy as SP, data as DATA

BASE = F.REPO / 'outputs/shenzhen_aligned_model_20261111'
ROUND = F.REPO / 'outputs/shenzhen_resolution_20261112/round01_set_decode'
OUT = F.REPO / 'outputs/shenzhen_resolution_20261112/capacity_diagnosis'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve diagnostic snapshot')
    freeze = F.read(ROUND / 'prediction_freeze.json')
    for rel, digest in freeze['files'].items():
        assert F.sha(ROUND / rel) == digest, rel
    rows = F.read(ROUND / 'diagnostic_cases_not_training.json')
    all_cases, reports, populations = [], {}, {}
    for district in ['龙华区', '龙岗区']:
        meta = F.read(BASE / 'inputs' / f'{district}.json')
        policy = SP.load_policy(district)
        names = {s['school_id']: s['school_name'] for s in meta['schools']}
        ids = list(names)
        observations = F.read(F.REPO / 'data/深圳市/规则对齐输入_20261111' / district / 'policy_observations.json')
        obs = {s['school_id']: s for s in observations}
        dummy = dict(coordinate_group_id='empty', building_name='', coordinate_group_building_names=[])
        roots = {frozenset(SP.apply(policy, dummy, [s])['predicted_school_ids']) for s in ids}
        refs = {frozenset(ids[s] for s in ss) for ss in meta['known'].values()}
        # Reference sets are rule estimates, not independently confirmed complete truth.
        applied_refs = {frozenset(SP.apply(policy, meta['sites'][int(q)], [ids[s] for s in ss])['predicted_school_ids'])
                        for q, ss in meta['known'].items()}
        declared_sets = {frozenset(r['school_ids']) for r in policy['relations'] if r['kind'] == 'shared'}
        categories, groups = collections.Counter(), collections.defaultdict(list)
        dr = [r for r in rows if r['district'] == district]
        for r in dr:
            gold = frozenset(r['gold'])
            if r['representable']:
                category = 'represented_correct' if set(r['after']) == gold else 'represented_selection_error'
            elif r['rule_known']:
                actual = set(r['after'])
                category = ('protected_incomplete' if actual < gold else
                            'protected_extra' if gold < actual else 'protected_mixed_error')
            elif gold in refs | applied_refs | declared_sets:
                category = 'absent_root_but_present_source_set'
            else:
                category = 'absent_all_enumerated_source_sets'
            categories[category] += 1
            relations = [p['rule_id'] for p in policy['relations'] if set(p['school_ids']) & gold
                         and p['kind'] in ['pending', 'eligibility', 'university']]
            all_cases.append(dict(r, category=category,
                root_without_protection=gold in roots,
                reference_raw_set=gold in refs, reference_applied_set=gold in applied_refs,
                declared_shared_set=gold in declared_sets, intersecting_pending_relations=relations))
            if not r['representable']:
                groups[tuple(sorted(gold))].append(r)
        group_rows = []
        for gold, cases in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            group_rows.append(dict(schools=[names[s] for s in gold], school_ids=list(gold), n=len(cases),
                protected=sum(r['rule_known'] for r in cases), semantic_pending=sum(r['semantic_pending'] for r in cases),
                source_set_available=frozenset(gold) in roots | refs | applied_refs | declared_sets,
                examples=[r['building_name'] for r in cases[:5]],
                official_sources=[dict(school_name=names[s], source_text=obs[s]['source_text'],
                    scope_status=obs[s]['scope_status'], evidence=obs[s]['evidence']) for s in gold]))
        reports[district] = dict(n=len(dr), categories=dict(categories),
            distinct_root_sets=len(roots), distinct_reference_sets=len(refs),
            distinct_applied_reference_sets=len(applied_refs), distinct_declared_shared_sets=len(declared_sets),
            unrestricted_source_set_coverage=sum(frozenset(r['gold']) in roots | refs | applied_refs | declared_sets for r in dr),
            unreachable_groups=group_rows)
    cache = F.REPO / 'outputs/shenzhen_full_source_cache_20261111'
    for filename, digest in F.read(cache / 'manifest.json')['files'].items():
        assert F.sha(cache / filename) == digest
        with (cache / filename).open('rb') as f:
            x, population = pickle.load(f)
        dist = DATA.with_list_gold(DATA.load(Path(filename).stem))
        city = dist.city
        c = populations.setdefault(city, dict(n=0, multi=0, references=0, schools=0, schools_with_reference=0))
        answers = [v for v in dist.answers.values() if len(v) and min(v) >= 0]
        c['n'] += len(answers); c['multi'] += sum(len(v) > 1 for v in answers)
        c['references'] += len(set(x.ref_sites)); c['schools'] += dist.school_n
        c['schools_with_reference'] += len(set(x.ref_schools))
    report = dict(purpose='Diagnostic only: map-set representability is not policy eligibility or deployable accuracy.',
        limitations='Rule reference sets may be incomplete. Set equality is mathematical coverage, not evidence of local applicability. Protected errors may be source/map semantic disagreement, not necessarily erroneous policy.',
        prediction_changes=False, gold_used_only_for_diagnosis=True, populations=populations, districts=reports)
    F.write(OUT / 'report.json', report)
    F.write(OUT / 'cases_not_training.json', all_cases)
    lines = ['深圳候选集合容量诊断（仅评分后分析，不作为训练输入）', '',
             '候选可表示不代表能选对；文字推导参照不一定是完整标签；地图命中不等于入学资格。', '']
    for city, c in populations.items():
        lines.append(f"{city}：多校 {c['multi']}/{c['n']}；参照学校 {c['schools_with_reference']}/{c['schools']}。")
    for district, report in reports.items():
        lines += ['', district, str(report['categories']),
                  f"取消固定输出约束、合并所有现有原始参照集合和明确共享集合后的数学覆盖：{report['unrestricted_source_set_coverage']}/{report['n']}。这不是预测结果。"]
        for g in report['unreachable_groups'][:8]:
            lines += [f"\n{g['n']}例：{'；'.join(g['schools'])}；固定输出{g['protected']}，语义待核{g['semantic_pending']}。",
                      '例：' + '、'.join(g['examples'])]
            for s in g['official_sources']:
                lines += [s['school_name'] + '：' + s['source_text']]
                lines += [e['source_url'] + ' ' + e['source_position'] for e in s['evidence']]
    (OUT / '候选缺口与官方原文.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    F.write(OUT / 'verification.json', dict(frozen_prediction_hashes_passed=True,
        all_scored_sites=len(all_cases), population_preserved=len(all_cases)==1761,
        source_cache_hashes_passed=True, inference_files_modified=False))
    print({d: {k:v for k,v in r.items() if k!='unreachable_groups'} for d,r in reports.items()})
    print(populations)


if __name__ == '__main__':
    main()
