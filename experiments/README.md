# 实验结果

- [固定参照的低参照训练与逐校查询对照](modern_low_reference_training_20261114/实验结论.txt)：固定84个参照与1095题，4个版本×3种子，未替换默认模型。复核：`python experiments/evaluate_low_reference_training.py`。

各轮配置、汇总成绩、审计和中文报告；最新四组实验另附逐题评分输出。训练缓存、权重和重复中间结果不上传。失败与草稿保留原状态。

新增两轮完整参照实验，含模拟核验名单和逐题预测：

- [按学校补缺：每校总目标1/2/3](modern_school_gap_reference_20261114/实验结论.txt)：两区新增84/167/255个核验小区，三档共同测试1095点；低预算主分支未达预期，不采用。
- [固定512点完整参照](modern_complete_reference_20261114/实验结论.txt)：两区各256个，剩余1249点；改造Modern含政策处理76.22%，开发候选。

在仓库根目录执行 `python experiments/evaluate_reference_trials.py`，无需训练缓存即可独立复算两轮成绩并检查预算、项目组隔离。这个命令不重训神经模型。完整训练脚本位于 `models/school-zone/scripts/`，仍依赖原历史输入和预训练权重；公开包不包含这些大型缓存与权重。

| 实验 | 文件数 |
|---|---:|
| [modern_conditional_admission_20261113](modern_conditional_admission_20261113/) | 3 |
| [modern_conditional_admission_hybrid_20261113](modern_conditional_admission_hybrid_20261113/) | 27 |
| [modern_shared_school_prior_20261114](modern_shared_school_prior_20261114/) | 2 |
| [modern_shenzhen_full_prior_20261114](modern_shenzhen_full_prior_20261114/) | 2 |
| [modern_shenzhen_full_prior_repaired_20261114](modern_shenzhen_full_prior_repaired_20261114/) | 8 |
| [modernnca_conditional_sets_20261113](modernnca_conditional_sets_20261113/) | 3 |
| [modernnca_evidence_v2_20261113](modernnca_evidence_v2_20261113/) | 1 |
| [modernnca_evidence_v2_extended_20261113](modernnca_evidence_v2_extended_20261113/) | 3 |
| [modernnca_learned_sets_20261113](modernnca_learned_sets_20261113/) | 3 |
| [modernnca_partial_sets_20261113](modernnca_partial_sets_20261113/) | 3 |
| [modernnca_shenzhen_multilabel_20261113](modernnca_shenzhen_multilabel_20261113/) | 5 |
| [shenzhen_aligned_model_20261111](shenzhen_aligned_model_20261111/) | 7 |
| [shenzhen_aligned_model_20261111_未评分草稿1](shenzhen_aligned_model_20261111_未评分草稿1/) | 3 |
| [shenzhen_aligned_model_20261111_未评分草稿2](shenzhen_aligned_model_20261111_未评分草稿2/) | 3 |
| [shenzhen_current_model_20261111](shenzhen_current_model_20261111/) | 4 |
| [shenzhen_full_model_20261111](shenzhen_full_model_20261111/) | 6 |
| [shenzhen_membership_trees_20261114](shenzhen_membership_trees_20261114/) | 17 |
| [shenzhen_resolution_20261112](shenzhen_resolution_20261112/) | 177 |
| [shenzhen_road_policy_trial_20261114](shenzhen_road_policy_trial_20261114/) | 8 |

[中文报告](报告/)。在仓库根目录运行 `python experiments/evaluate.py`，用两区Gold重算最新13组成绩。
