# JG-AI

模拟损伤光斑图分级识别基线项目。

当前这套实现面向 `D:\project\img_sorter\模拟图` 下的模拟数据，目标是只根据 `损伤光斑图` 完成两阶段分类：

- 先判断损伤发生在 `第一级` 还是 `第二级`
- 再判断损伤程度落在 `S1` 到 `S4` 的哪个粗略区间
- 提供本地 `Gradio` 页面，支持单张图片上传和结果查看

这套流程与旧的 `app.py` 分开维护，不复用旧业务入口。

## 当前能力

- 自动从本地 `模拟图` 目录构建分类数据集
- 自动生成三套分类训练数据：
  - 级联位置分类器
  - 第一级程度分类器
  - 第二级程度分类器
- 提供训练、评估、单图推理和网页交互入口
- 支持把损伤程度按编号区间映射为 `S1-S4`

## 项目结构

- `damage_classifier/`: 数据构建、预处理、训练、推理、评估、Gradio 页面
- `prepare_damage_dataset.py`: 从本地 `模拟图` 目录生成分类数据集
- `train_damage_models.py`: 训练三套分类模型
- `evaluate_damage_models.py`: 评估两阶段推理链路
- `launch_damage_gradio.py`: 启动本地网页界面
- `start_ui.bat`: 一键启动本地网页界面
- `README_damage_classifier.md`: 当前这套分类流程的补充说明
- `USER_GUIDE_CN.md`: 面向新手的完整使用说明
- `PLAN_damage_classifier_next_steps.md`: 后续训练和改模型的备忘计划
- `tests/`: 基础单元测试

## 数据规则

模型输入只使用：

- `第一级/损伤光斑图`
- `第二级/损伤光斑图`

当前默认标签规则如下：

- 级联位置标签：直接使用目录名 `第一级` / `第二级`
- 程度标签：根据图片编号按 4 档粗分

默认程度分档为：

- `S1`: `img_001` - `img_030`
- `S2`: `img_031` - `img_060`
- `S3`: `img_061` - `img_090`
- `S4`: `img_091` - `img_121`

## 快速开始

先进入项目并激活虚拟环境：

```powershell
cd D:\project\img_sorter
.\.venv\Scripts\activate
```

### 1. 构建数据集

```powershell
python prepare_damage_dataset.py --clean
```

### 2. 训练模型

CPU 训练示例：

```powershell
python train_damage_models.py --models all --epochs 40 --device cpu
```

如果本机 GPU 可用，可以改成：

```powershell
python train_damage_models.py --models all --epochs 40 --device 0
```

### 3. 评估模型

```powershell
python evaluate_damage_models.py --device cpu
```

### 4. 启动网页界面

```powershell
python launch_damage_gradio.py --device cpu
```

启动后默认访问地址：

```text
http://127.0.0.1:7860
```

## 训练输出位置

训练和导出结果默认会写到：

- `damage_artifacts/datasets`: 预处理后的分类数据集
- `damage_artifacts/runs`: Ultralytics 训练输出
- `damage_artifacts/models`: 导出的 `best.pt` / `last.pt`
- `damage_artifacts/metadata`: 样本元数据与数据集摘要

默认导出模型路径：

- `damage_artifacts/models/cascade_level/best.pt`
- `damage_artifacts/models/severity_stage1/best.pt`
- `damage_artifacts/models/severity_stage2/best.pt`

## 依赖

当前核心依赖包括：

- `ultralytics`
- `gradio`
- `pandas`

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 当前状态说明

这套实现已经完成本地冒烟验证，说明训练和推理链路是通的；但当前只是第一版基线，不代表最终精度已经达到最好。

如果后面要继续优化模型，建议先看：

- `USER_GUIDE_CN.md`
- `README_damage_classifier.md`
- `PLAN_damage_classifier_next_steps.md`

其中 `USER_GUIDE_CN.md` 说明了从准备数据、训练模型、查看结果到打开网页界面的完整操作流程；计划文件专门记录了后续准备怎么继续训练、调参、改分档和决定是否需要人工标注。
