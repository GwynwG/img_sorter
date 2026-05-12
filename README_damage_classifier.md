<<<<<<< HEAD
﻿# 模拟损伤光斑图分级识别
=======
# 模拟损伤光斑图分级识别
>>>>>>> 6ed2eb1deaef4a6128758f9b78052bf38af78ac2

这套新流程不会改你原来的 `app.py`，而是单独使用下面 4 个入口：

```powershell
python prepare_damage_dataset.py
python train_damage_models.py --models all --epochs 40
python evaluate_damage_models.py
python launch_damage_gradio.py
```

## 功能

- 输入：`模拟图` 下的 `损伤光斑图`
- 输出：
  - `第一级` / `第二级`
  - `S1` / `S2` / `S3` / `S4`
  - 两项置信度
  - 按编号区间给出的损伤程度说明

## 目录

- 原始数据：`模拟图`
- 生成数据集：`damage_artifacts/datasets`
- 训练结果：`damage_artifacts/runs`
- 导出模型：`damage_artifacts/models`
- 元数据：`damage_artifacts/metadata`

## 默认分档

- `S1`: `img_001` - `img_030`
- `S2`: `img_031` - `img_060`
- `S3`: `img_061` - `img_090`
- `S4`: `img_091` - `img_121`

## 说明

- 默认模型骨架使用 `yolo11n-cls.yaml`
- 如果你本地已经有 `yolo11n-cls.pt`，可以在训练时加：

```powershell
python train_damage_models.py --model-source D:\你的路径\yolo11n-cls.pt
```
