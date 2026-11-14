"""执行深圳基础划片规则。可传入冻结模型结果，不调用旧S模型冒充现用模型。

python scripts/run_shenzhen_policy.py --predictions predictions.json --output outputs/shenzhen_policy_run
predictions.json: {district: {coordinate_group_id: [school_id, ...]}}
facts.json: {district: {coordinate_group_id: [独立适用事实, ...]}}
不传模型结果则输出政策可确认集合与弃权；该模式不报告“模型准确率”。
"""
import argparse
import json
import shutil
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zoning import shenzhen_sources, shenzhen_policy, shenzhen_policy_evaluate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path)
    ap.add_argument('--facts', type=Path)
    ap.add_argument('--output', type=Path, default=REPO/'outputs/shenzhen_policy_20261111')
    args = ap.parse_args()
    predictions = read(args.predictions) if args.predictions else {}
    facts = read(args.facts) if args.facts else {}
    if set(predictions) - set(shenzhen_sources.DISTRICTS) or set(facts) - set(shenzhen_sources.DISTRICTS):
        raise ValueError('只允许现用龙华、龙岗；旧区结果不能混用')
    if args.predictions and set(predictions) != set(shenzhen_sources.DISTRICTS):
        raise ValueError('模型结果必须提供两个现用区，不能静默漏区')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('输出目录非空，请使用新目录保留冻结结果')
    frozen = {}
    for d in shenzhen_sources.DISTRICTS:
        if args.predictions:
            keys = {r['coordinate_group_id'] for r in shenzhen_sources._load(d, '小区数据/坐标组.json')}
            if set(predictions[d]) != keys:
                raise ValueError('模型结果须覆盖全部候选坐标组（可显式空集），不能按Gold筛选或沿用旧版行号')
        frozen[d] = shenzhen_policy.predict_district(d, predictions.get(d), facts.get(d))
        write(out/d/'predictions.json', frozen[d])
    # 所有区预测均保存后，才允许读取评价数据。
    prediction_hashes = {d:shenzhen_policy.source_hash(out/d/'predictions.json') for d in frozen}
    summary = []
    for d, rows in frozen.items():
        policy = shenzhen_policy.load_policy(d)
        gold = shenzhen_sources.load_point_gold(d)
        audit = shenzhen_policy_evaluate.audit_gold(policy, gold)
        write(out/d/'基础标签适用性.json', audit)
        item = dict(district=d, candidate_sites=len(rows),
                    policy_supported_sites=sum(bool(r['policy_supported_school_ids']) for r in rows),
                    output_sites=sum(bool(r['predicted_school_ids']) for r in rows),
                    multischool_output_sites=sum(len(r['predicted_school_ids'])>1 for r in rows),
                    map_gold_groups=len(gold), semantic_pending=sum(bool(r['unresolved_reasons']) for r in audit))
        if args.predictions:
            result = shenzhen_policy_evaluate.score(rows, gold, audit)
            write(out/d/'evaluation.json', result)
            item['evaluation'] = {k:v for k,v in result.items() if k != 'rows'}
        summary.append(item)
    assert all(shenzhen_policy.source_hash(out/d/'predictions.json') == h for d,h in prediction_hashes.items())
    code_files = [Path(__file__), REPO/'zoning/shenzhen_policy.py', REPO/'zoning/shenzhen_policy_evaluate.py', REPO/'zoning/shenzhen_sources.py']
    for code in code_files:
        target = out/'code'/code.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code, target)
    write(out/'manifest.json', dict(schema='shenzhen_policy_execution_v1', mode='frozen_model_plus_rules' if args.predictions else 'policy_only_no_model_score',
          current_dataset=shenzhen_sources.CURRENT, prediction_hashes=prediction_hashes,
          predictions_input_sha256=shenzhen_policy.source_hash(args.predictions) if args.predictions else None,
          facts_input_sha256=shenzhen_policy.source_hash(args.facts) if args.facts else None,
          rules_sha256={d:shenzhen_policy.source_hash(shenzhen_sources.DATA/d/'政策规则/规则包.json') for d in frozen},
          gold_sha256={d:shenzhen_policy.source_hash(shenzhen_sources.DATA/d/'Gold/小区_学校Gold.json') for d in frozen},
          code_sha256={str(p.relative_to(REPO)):shenzhen_policy.source_hash(p) for p in code_files},
          prediction_frozen_before_evaluation=True, districts=summary, model_training_rerun=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
