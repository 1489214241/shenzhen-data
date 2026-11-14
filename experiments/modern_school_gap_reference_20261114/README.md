# 按学校补缺参照

[实验结论](实验结论.txt)给出完整方案、结果及限制。每校1/2/3是**已有参照加新增参照的总目标**，不是每校额外增加。

- `verified_partial_references.json`：已有391条文字证据正关系；没有被冒充为完整学校集合。
- `released_plans.json`：三档嵌套的Gold模拟核验清单、剩余查询、共同查询及未补齐学校。
- `budget_summary.json`、`逐校补缺与官方证据.json`：去重成本和逐校证据。
- `results.json`：共同1095点主对照以及每档自然剩余队列。
- `scored_rows/`：所有方法的完整逐题预测，可用仓库两区Gold重算。
- `protocol.json`、`training_selection.json`、`independent_audit.json`：预定方案、内部验证及独立计分检查。
- `post_score_coverage_diagnostic.json`：评分后才计算的组合覆盖诊断，未进入训练或选择。
- `zero_budget_control/`：看到主结果后补做的零新增核验固定对照，无新调参。

从仓库根目录运行：

```bash
python experiments/evaluate_reference_trials.py
```

训练源码在 `models/school-zone/scripts/experiment_school_gap_reference.py`，主分支在 `models/school-zone/zoning/modernnca_complete_reference.py`。原源码和哈希保留本地路径溯源；此公开包提供成绩复核所需的数据与预测，不包含历史训练缓存、预训练权重或本轮训练权重，不能称为零配置从零重训。

学校标签参与模拟选点。新增核验尚未真实采集。Gold表示地图点落面关系，不等于实际入学资格。主实验不支持低预算直接采用完整组合分支，默认模型未变。
