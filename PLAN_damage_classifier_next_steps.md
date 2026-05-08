# 模拟损伤光斑图模型后续计划

本文档用于记录当前基线模型之后的优化方向，便于后续继续训练、调参和复盘。

## 当前基线

当前系统采用三模型两阶段分类流程：

- `cascade_level`: 判断损伤位于 `第一级` 或 `第二级`。
- `severity_stage1`: 判断第一级损伤程度 `S1-S4`。
- `severity_stage2`: 判断第二级损伤程度 `S1-S4`。

程度分档仍按模拟序列编号等宽划分：

- `S1`: `img_001` - `img_030`
- `S2`: `img_031` - `img_060`
- `S3`: `img_061` - `img_090`
- `S4`: `img_091` - `img_121`

## 已知问题

- 程度模型的验证集 Top1 波动较大，说明模型对相邻程度档的边界还不稳定。
- 当前样本数量较少，每个级联只有 121 张图，验证集更小，单个样本会明显影响曲线。
- `S1-S4` 是按编号等宽划分，未必完全对应真实物理损伤程度的视觉变化边界。
- 当前只使用模拟图训练，暂不保证对真实实验图的泛化能力。

## 下一步建议

1. 继续保留两阶段结构，先优化 `cascade_level`，再分别优化两个程度模型。
2. 重点查看 `confusion_matrix.png`，确认错误是否主要发生在相邻程度档。
3. 对程度模型增加“相邻档位可接受准确率”作为汇报指标，不只看 Top1。
4. 尝试更稳的训练设置，例如降低学习率、增加 patience、减少过强数据增强。
5. 如果后续有真实图，先单独建立真实图验证集，不要直接混入训练集。
6. 如发现 `S2/S3` 或 `S3/S4` 长期混淆，可考虑重新定义程度分档边界。
7. 每次新训练后保留 `damage_artifacts\archives` 中的历史模型和曲线，避免覆盖可用版本。

## 推荐复盘材料

每次训练后建议至少保存和查看：

- `damage_artifacts\runs\*\results.png`
- `damage_artifacts\runs\*\results.csv`
- `damage_artifacts\runs\*\confusion_matrix.png`
- `damage_artifacts\models\*\train_summary.json`
- `damage_artifacts\metadata\evaluation_latest.json`
- `output\training_analysis` 下的中文分析图和汇总表
