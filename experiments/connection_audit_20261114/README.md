# GitHub连接后核查记录（2026-11-14）

## 执行边界

本次独立核查读取两个固定提交：旧基线 `646c170db0a1a738833062efb90cf8b40e0e44f2` 和主分支新提交 `87fc8c0a6fcf6c1350e68b88b896c138634ee43b`。
本分支只增加核查工作流与记录，不修改main、Gold、默认模型或冻结预测。本次没有重新训练神经模型，没有产生新方法性能结果。

已完成：
- GitHub Actions在646c170实际运行evaluate_reference_trials.py和evaluate.py，全部通过。run 43599627960。
- GitHub Actions在87fc8c0实际运行evaluate_low_reference_training.py和evaluate_reference_trials.py，全部通过。run 43600517688。
- 下载两个提交的公开快照，在独立容器再次运行评分。
- 另写独立计分脚本，重新从两区公开Gold计算完整集合正确率、误报漏报及配对修正/误伤，与最新报告一致。
- 最新test_low_reference_training.py和test_modernnca_multilabel.py共10项功能测试通过。此前独立候选原型15项功能测试通过。功能测试不代表预测性能提升。

## 最新已提交实验，不是本次重新训练的结果

固定1095题；既有目标1的84个Gold模拟完整核验不变；相同政策处理。

| 方法 | 正确数 | 完整集合准确率 |
|---|---:|---:|
| 已有独立多标签结构，普通训练 |396|36.16%|
| 已有独立多标签结构，稀疏参照训练 |293|26.76%|
| 学校条件查询，普通训练 |479|43.74%|
| 学校条件查询，稀疏参照训练 |348|31.78%|
| 历史Modern微调，对照 |514|46.94%|
| 历史Modern不微调，对照 |533|48.68%|

配对复算：
- global_sparse_policy - global_full_policy：修正15、误伤118，-9.4064个百分点。
- local_sparse_policy - local_full_policy：修正66、误伤197，-11.9635个百分点。
- local_full_policy - global_full_policy：修正160、误伤77，+7.5799个百分点。
- local_sparse_policy - global_sparse_policy：修正76、误伤21，+5.0228个百分点。

稀疏训练后的global误报由1368增加到2053（+685条学校关系），漏报584降至583。这里只描述错误变化，不声称确定因果机制。
学校条件查询普通训练的最优阈值为1.0，raw输出均为单校；政策后多校不能说成模型已学会未见组合。
新四臂随机初始化、源区验证；历史两臂训练过程不同。因此与历史对照比较实际表现，不作单因素因果归因。

## 两项必须纠正的前提

1. 原目标1是“补到至少1”，不是每校总共只用1。按公布的可靠参照计数，202校中114所恰为1、86所大于1、2所为0；原弱参照还共同保留。最新稀疏训练也只限制来源区，深圳保留原池。因此未验证严格部署K=1协议。
2. 独立多标签输出并非未尝试。此前modernnca_shenzhen_multilabel_20261113在其1761题协议下，单校加政策50.99%、多标签加政策39.86%。不能把换成独立输出包装为已经有效的新办法。最新87fc8c0又检验了稀疏训练和学校条件查询，预设稀疏训练对照为负结果。

这并不证明所有少参照方法无效，也不是此前独立原型的逐行实现复现；但已不能在无新证据下继续推荐该具体训练改动为首选有效方案。

## 重新训练仍缺什么

公开仓库有代码、JSON、冻结逐题结果，但省略.pt权重、.pkl特征缓存和.npz数组。
旧646c170输入哈希清单53项：公开副本仅7项匹配、46项缺失。另核对的来源区重建依赖也未提供。
已实测：可构建9923参数的原模型；加载历史categorical/20261231.pt时报FileNotFoundError。训练入口在Linux还存在winreg顶层导入的可移植性问题。

连接GitHub解决了读取仓库和运行公开评分的问题，不会自动提供F盘未提交的文件。因此本次不称为“从零重训复现”，也不报告不存在的新准确率。

## 复核入口

在87fc8c0快照：
```bash
python experiments/evaluate_low_reference_training.py
python experiments/evaluate_reference_trials.py
cd models/school-zone
PYTHONPATH=. python -m pytest -q tests/test_low_reference_training.py tests/test_modernnca_multilabel.py
```

结果来源：experiments/modern_low_reference_training_20261114/实验结论.txt、results.json、scored_rows/；旧协议来源：experiments/modern_school_gap_reference_20261114/实验结论.txt及released_plans.json。
