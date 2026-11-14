# 深圳实验代码

- `zoning/`：ModernNCA及检索模型、政策规则、特征、训练、预测和评价。
- `scripts/`：数据处理和各轮实验入口。
- `configs/`：原实验配置。
- `tests/`：已有测试。
- `third_party/`：引用的模型代码及原许可。
- `data/`：迁移实验使用的长沙、杭州基础输入；深圳指针读取仓库根目录的龙华、龙岗数据，不复制第二套深圳主数据。

最新全学校成员预测试验入口为 `scripts/experiment_shenzhen_school_membership_trees.py`；当前条件补校实验入口为 `scripts/experiment_conditional_admission_hybrid.py`。详细参数、来源验证和采用状态见根目录 `experiments/` 对应目录。

原脚本使用Windows工作区、阶段快照和哈希校验；本次按长沙形式公开源码、数据和结果，省略大型训练缓存、权重与重复派生输入。历史绝对路径未伪装成可跨机器直接运行的命令，重训前需重建相应输入并设置新的输出目录。

只核对公开预测结果时，在仓库根目录运行 `python experiments/evaluate.py`，无需安装模型依赖。
