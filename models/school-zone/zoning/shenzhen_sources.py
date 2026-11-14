"""读取深圳现用标准数据；边界、政策输入和评价标签分别读取。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/深圳市'
CURRENT = json.loads((ROOT / '当前数据集.json').read_text(encoding='utf-8'))
DATA = ROOT / CURRENT['dataset_path']
DISTRICTS = tuple(CURRENT['active_districts'])


def _load(district, filename):
    if district not in DISTRICTS:
        raise ValueError(f'{district}不在深圳现用区中；现用：{", ".join(DISTRICTS)}。福田、南山原始资料保留在data/深圳市/同名历史目录。')
    return json.loads((DATA / district / filename).read_text(encoding='utf-8'))


def load_schools(district):
    """学校目录；以school_id与polygon、文字关联。"""
    return _load(district, 'schools.json')


def load_texts(district):
    """原始政策文字与来源；空缺、候选和历史适用状态均保留。"""
    return _load(district, 'texts.json')


def load_polygons(district):
    """官方边界证据，原始BD-09约定；不作为预测模型输入。"""
    return _load(district, 'polygons.json')


def load_buildings(district):
    """全部独立候选小区；不依据评价标签筛选。"""
    return _load(district, '小区数据/buildings.json')


def load_point_gold(district):
    """仅评价用的原图点落面集合，禁止作为政策规则或预测输入。"""
    return _load(district, 'Gold/小区_学校Gold.json')


def load_policy_rules(district):
    """现用基础划片规则；含官方原文、大学区/共享/资格分层及适用条件。"""
    from .shenzhen_policy import load_policy
    return load_policy(district)


def predict_with_policy(district, predictions=None, facts=None):
    """执行规则；模型预测以坐标组ID连接，无预测时缺依据的小区保持未知。"""
    from .shenzhen_policy import predict_district
    return predict_district(district, predictions, facts)
