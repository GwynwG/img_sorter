# 模拟损伤光斑图识别系统使用说明

本文档面向第一次使用本项目的人员，说明如何准备训练数据、训练模型、查看训练结果，以及如何打开网页界面进行单张图片识别。

## 1. 项目位置

项目目录：

```powershell
D:\project\img_sorter
```

进入项目目录：

```powershell
cd D:\project\img_sorter
```

## 2. 训练数据放在哪里

默认训练数据目录为：

```text
D:\project\img_sorter\模拟图
```

目录必须保持下面的结构：

```text
模拟图
├── 第一级
│   ├── 损伤图
│   └── 损伤光斑图
└── 第二级
    ├── 损伤图
    └── 损伤光斑图
```

当前训练只使用 `损伤光斑图`，`损伤图` 只用于数据核对和后续解释参考。

每个 `损伤光斑图` 文件夹下应有 `121` 张图片，命名格式建议为：

```text
img_001.jpg
img_002.jpg
...
img_121.jpg
```

标签规则如下：

- `第一级` / `第二级` 目录名表示损伤发生的级联位置。
- 图片编号表示损伤由轻到重的演化顺序。
- `S1`: `img_001` - `img_030`
- `S2`: `img_031` - `img_060`
- `S3`: `img_061` - `img_090`
- `S4`: `img_091` - `img_121`

## 3. 构建训练数据集

训练前需要先把原始图片转换成 Ultralytics 分类任务需要的目录结构。

使用当前 `.venv` 环境：

```powershell
cd D:\project\img_sorter
.\.venv\Scripts\activate
python prepare_damage_dataset.py --clean
```

生成的数据集会放在：

```text
D:\project\img_sorter\damage_artifacts\datasets
```

会生成三套数据集：

```text
damage_artifacts\datasets\cascade_level
damage_artifacts\datasets\severity_stage1
damage_artifacts\datasets\severity_stage2
```

含义如下：

- `cascade_level`: 级联位置分类器数据集，用于判断 `第一级` / `第二级`。
- `severity_stage1`: 第一级损伤程度分类器数据集，用于判断 `S1-S4`。
- `severity_stage2`: 第二级损伤程度分类器数据集，用于判断 `S1-S4`。

## 4. 训练模型

### 4.1 使用 CPU 训练

CPU 训练环境简单，但速度较慢：

```powershell
cd D:\project\img_sorter
.\.venv\Scripts\activate
python train_damage_models.py --models all --epochs 40 --device cpu
```

如果需要训练更久，例如 `200` 轮：

```powershell
python train_damage_models.py --models all --epochs 200 --device cpu
```

### 4.2 使用 GPU 训练

本机可用的 GPU Python 环境为：

```text
D:\ANACONDA\envs\yolov11_gpu\python.exe
```

使用 GPU 训练 `200` 轮：

```powershell
cd D:\project\img_sorter
D:\ANACONDA\envs\yolov11_gpu\python.exe train_damage_models.py --models all --epochs 200 --device 0
```

参数说明：

- `--models all` 表示训练三套模型。
- `--epochs 200` 表示最多训练 200 轮。
- `--device 0` 表示使用第 1 张 NVIDIA GPU。
- 如果训练过程中触发 Early Stopping，没有跑满 200 轮也属于正常情况。

## 5. 训练结果在哪里

训练结果主要分为三类。

### 5.1 训练过程结果

路径：

```text
D:\project\img_sorter\damage_artifacts\runs
```

三个模型分别在：

```text
damage_artifacts\runs\cascade_level
damage_artifacts\runs\severity_stage1
damage_artifacts\runs\severity_stage2
```

每个目录下重点查看：

- `results.png`: Ultralytics 默认训练曲线图。
- `results.csv`: 每一轮训练的原始指标数据。
- `confusion_matrix.png`: 混淆矩阵。
- `confusion_matrix_normalized.png`: 归一化混淆矩阵。
- `weights\best.pt`: 验证集效果最好的模型。
- `weights\last.pt`: 最后一轮模型。

一般优先使用 `best.pt`，不要优先使用 `last.pt`。

### 5.2 导出的最终模型

路径：

```text
D:\project\img_sorter\damage_artifacts\models
```

默认推理使用这三个模型：

```text
damage_artifacts\models\cascade_level\best.pt
damage_artifacts\models\severity_stage1\best.pt
damage_artifacts\models\severity_stage2\best.pt
```

含义如下：

- `cascade_level\best.pt`: 判断损伤发生在第一级还是第二级。
- `severity_stage1\best.pt`: 判断第一级损伤程度。
- `severity_stage2\best.pt`: 判断第二级损伤程度。

### 5.3 按日期归档的历史结果

每次新训练完成后，系统会自动保留一份带日期后缀的历史结果。

路径：

```text
D:\project\img_sorter\damage_artifacts\archives
```

常见目录：

```text
damage_artifacts\archives\runs
damage_artifacts\archives\models
damage_artifacts\archives\summaries
damage_artifacts\archives\evaluations
```

示例：

```text
damage_artifacts\archives\models\cascade_level_20260508
damage_artifacts\archives\runs\severity_stage2_20260508
damage_artifacts\archives\evaluations\evaluation_20260508.json
```

同一天多次训练时，会自动生成 `_v2`、`_v3` 等版本，避免覆盖旧结果。

## 6. 评估模型

训练完成后，运行评估脚本。

CPU 评估：

```powershell
cd D:\project\img_sorter
.\.venv\Scripts\activate
python evaluate_damage_models.py --device cpu
```

GPU 评估：

```powershell
cd D:\project\img_sorter
D:\ANACONDA\envs\yolov11_gpu\python.exe evaluate_damage_models.py --device 0
```

评估结果会显示：

- `val_samples`: 验证集图片数量。
- `cascade_accuracy`: 级联位置识别准确率。
- `severity_top1_accuracy`: 损伤程度 Top1 准确率。
- `severity_relaxed_accuracy`: 相邻档位也算可接受时的宽松准确率。

最新评估结果会保存到：

```text
D:\project\img_sorter\damage_artifacts\metadata\evaluation_latest.json
```

历史评估结果会保存到：

```text
D:\project\img_sorter\damage_artifacts\archives\evaluations
```

## 7. 查看中文分析图

项目提供了重新绘制后的中文分析图。

路径：

```text
D:\project\img_sorter\output\training_analysis
```

示例：

```text
output\training_analysis\20260401\三模型精度趋势分析_20260401.png
output\training_analysis\20260401\三模型损失曲线分析_20260401.png
output\training_analysis\20260401\训练结果汇总快照_20260401.png
output\training_analysis\20260401\稳定性与回落风险分析_20260401.png
```

对应的汇总数据在：

```text
output\training_analysis\20260401\训练结果汇总_20260401.csv
output\training_analysis\20260401\训练结果汇总_20260401.json
```

如果需要重新生成分析图，运行：

```powershell
cd D:\project\img_sorter
.\.venv\Scripts\activate
python output\jupyter-notebook\generate_training_analysis_cn.py
```

分析 notebook 在：

```text
D:\project\img_sorter\output\jupyter-notebook\training-results-analysis-cn.ipynb
```

## 8. 打开网页可视化界面

最简单的启动方式是双击项目目录下的：

```text
D:\project\img_sorter\start_ui.bat
```

双击后等待终端出现本地地址，再在浏览器打开：

```text
http://127.0.0.1:7860
```

也可以手动启动：

```powershell
cd D:\project\img_sorter
D:\project\img_sorter\.venv\Scripts\python.exe launch_damage_gradio.py --device cpu
```

注意：

- 网页界面建议使用 `.venv` 启动，因为这个环境已经安装 `gradio`。
- `D:\ANACONDA\envs\yolov11_gpu` 主要用于 GPU 训练，默认没有安装 `gradio`。
- 如果用 `yolov11_gpu` 直接启动网页，可能会出现 `ModuleNotFoundError: No module named 'gradio'`。

如果确实要用 GPU 环境启动网页，需要先给 GPU 环境安装 `gradio`：

```powershell
D:\ANACONDA\envs\yolov11_gpu\python.exe -m pip install gradio
D:\ANACONDA\envs\yolov11_gpu\python.exe launch_damage_gradio.py --device 0
```

网页功能：

- 上传单张 `损伤光斑图`。
- 输出级联位置：`第一级` 或 `第二级`。
- 输出损伤程度：`S1`、`S2`、`S3` 或 `S4`。
- 输出级联位置置信度和损伤程度置信度。
- 输出损伤程度文字说明。

## 9. 在网页中导入新模型

网页界面中有一个折叠区域：

```text
模型选择（不选择则使用默认最新模型）
```

展开后可以选择三个模型文件：

```text
级联位置模型 best.pt
第一级程度模型 best.pt
第二级程度模型 best.pt
```

如果不选择模型，系统会默认使用：

```text
damage_artifacts\models\cascade_level\best.pt
damage_artifacts\models\severity_stage1\best.pt
damage_artifacts\models\severity_stage2\best.pt
```

如果要测试历史模型，可以从归档目录选择对应的 `best.pt`：

```text
damage_artifacts\archives\models\cascade_level_日期\best.pt
damage_artifacts\archives\models\severity_stage1_日期\best.pt
damage_artifacts\archives\models\severity_stage2_日期\best.pt
```

建议一次选择成套的三个模型，避免级联位置模型和程度模型来自不同训练版本，导致结果不一致。

## 10. 推荐的新手操作顺序

第一次使用时，建议按下面顺序操作：

1. 确认数据放在 `D:\project\img_sorter\模拟图`。
2. 运行 `python prepare_damage_dataset.py --clean`。
3. 运行训练命令，例如 GPU 训练 `200` 轮。
4. 运行 `evaluate_damage_models.py` 查看评估结果。
5. 查看 `damage_artifacts\runs` 下的 `results.png` 和 `confusion_matrix.png`。
6. 查看 `output\training_analysis` 下的中文分析图。
7. 启动 `launch_damage_gradio.py` 打开网页界面。
8. 上传一张损伤光斑图，查看识别结果。
9. 如需比较不同模型，在网页中的模型选择区域导入对应的 `best.pt`。

## 11. 常见问题

### 11.1 找不到模型文件

如果网页启动后提示找不到 `best.pt`，说明还没有训练模型，或模型文件不在默认位置。

先运行训练命令，或者在网页中手动选择三个 `.pt` 模型文件。

### 11.2 端口被占用

默认端口是 `7860`。如果端口被占用，可以换一个端口：

```powershell
python launch_damage_gradio.py --device cpu --port 7861
```

然后打开：

```text
http://127.0.0.1:7861
```

### 11.3 GPU 命令不能运行

先确认使用的是 GPU 环境：

```powershell
D:\ANACONDA\envs\yolov11_gpu\python.exe -c "import torch; print(torch.cuda.is_available())"
```

如果输出为 `True`，说明 GPU 环境可用。

### 11.4 Top5 一直是 1.0

当前程度分类只有 `4` 个类别，`Top5` 没有参考意义。分析模型效果时主要看：

- `metrics/accuracy_top1`
- `train/loss`
- `val/loss`
- `confusion_matrix.png`
