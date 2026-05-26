# Training Workbench Design

## Context

The project is a local Gradio application for simulated damage spot image classification. The current app supports single-image inference, batch inference, model path selection, and a severity position visual. Training, evaluation, and artifact review are available through command-line scripts and generated files under `damage_artifacts`, but they are not first-class workflows in the web UI.

Current model flow:

- `cascade_level` predicts whether damage belongs to `第一级` or `第二级`.
- `severity_stage1` predicts `S1-S4` for first-level damage.
- `severity_stage2` predicts `S1-S4` for second-level damage.

Training artifacts already include `results.csv`, `results.png`, confusion matrices, `args.yaml`, exported `best.pt` and `last.pt`, and `train_summary.json`.

## Goals

Add training, hyperparameter tuning, and neural-network visualization to the existing interface without weakening the current inference experience.

The feature should:

- Keep the single-image recognition flow familiar.
- Allow one training run to be started from the web UI.
- Show live training status, logs, and useful metrics while training runs.
- Show training artifacts and lightweight experiment comparisons after training.
- Explain the three-model two-stage neural-network flow visually.
- Summarize the current YOLO classification model files in a way that is easy to read and screenshot.

## Non-Goals

This version will not implement a full experiment tracking platform, distributed training, editable neural-network architecture design, or automatic hyperparameter search. It will provide a solid single-run training workbench and a light history comparison layer that can grow later.

## UI Structure

The Gradio app will move to top-level tabs:

- `单张识别`: preserves the existing image upload, prediction result, model path selection, and severity position visual.
- `批量识别`: moves the current batch upload, batch table, and Excel export into its own tab.
- `训练工作台`: provides parameter presets, manual training parameters, start/status controls, live logs, and latest metric previews.
- `模型分析`: reads training artifacts and shows result summaries, curves, confusion matrices, model paths, and light historical comparison.
- `神经网络结构`: shows the two-stage pipeline diagram and a readable summary of the selected YOLO classification models.

This keeps inference and training separate while making both workflows visible.

## Training Workbench

The training tab will support one active training job at a time.

User controls:

- Model target: `全部模型`, `级联位置分类器`, `第一级损伤程度分类器`, or `第二级损伤程度分类器`.
- Presets: `快速验证`, `常规训练`, and `较长训练`.
- Editable parameters: `epochs`, `imgsz`, `batch`, `device`, `workers`, and `model-source`.

Before starting a run, the app checks that the required dataset directories exist. If they do not, the user sees a clear message to run dataset preparation first.

While training runs, the page shows:

- Job state: idle, running, finished, or failed.
- Current training target.
- Live log tail from the subprocess.
- Latest available epoch metrics parsed from `results.csv`.
- Preview paths for `results.png` and confusion matrices when they appear.

After training finishes, the page shows:

- Exported `best.pt` and `last.pt`.
- Run directory.
- Summary JSON path.
- `results.csv`, `results.png`, and confusion matrix paths.
- A refresh action for the model analysis tab to pick up the new run.

## Model Analysis

The model analysis tab will read artifacts from the current run directories and archive directories.

For each of the three model names, it will show:

- Display name and model key.
- Training arguments from `args.yaml`.
- Latest or best available Top1 accuracy.
- Final validation loss.
- Epoch count.
- Current `best.pt` path.
- Current `results.csv` path.
- `results.png`, `confusion_matrix.png`, and `confusion_matrix_normalized.png` if present.

Historical comparison will start as a compact table using archived summaries and archived run directories. It should help compare recent training outputs without becoming a full experiment database.

Missing artifacts are handled per model. One missing file should not break the whole page; it should render as `暂无结果` or `未找到文件`.

## Neural-Network Visualization

The structure tab has two layers.

First, a fixed two-stage pipeline visual:

`输入损伤光斑图 -> 级联位置分类器 -> 第一级/第二级程度分类器 -> S1/S2/S3/S4`

Second, model summaries for the selected `.pt` files:

- Model file path.
- Input image size used by the app.
- Class labels.
- Layer/module count when available.
- Parameter count when available.
- A plain-language description of what the model contributes to the pipeline.

If a YOLO model file cannot be loaded, the flow diagram remains visible and the model card displays a clear parse failure message.

## Code Organization

The implementation should keep responsibilities separated:

- `damage_classifier/training_manager.py`: starts training subprocesses, stores current job state, tails logs, prevents concurrent runs, and exposes status snapshots.
- `damage_classifier/training_artifacts.py`: reads `results.csv`, `args.yaml`, `train_summary.json`, confusion matrices, and archive summaries into stable Python data structures.
- `damage_classifier/model_visualization.py`: builds the two-stage pipeline representation and extracts YOLO model summaries.
- `damage_classifier/gradio_app.py`: owns the UI composition and wires tabs to the helper modules.

The existing prediction pipeline and preprocessing modules should stay focused on inference and image preparation.

## Data Flow

Training flow:

1. User selects target models and parameters in `训练工作台`.
2. UI asks `training_manager` to start a training subprocess using the existing CLI entrypoint.
3. `training_manager` records the job metadata and log file location.
4. UI polls `training_manager` for status and `training_artifacts` for newly written metrics.
5. After completion, existing training code exports `best.pt`, `last.pt`, summaries, and archives.
6. `模型分析` reads the same artifact locations and refreshes its summaries.

Analysis flow:

1. `training_artifacts` scans the known model keys from `MODEL_SPECS`.
2. It reads available CSV, YAML, JSON, and image paths.
3. It returns one model summary per model plus historical rows.
4. Gradio renders summaries, images, and tables with missing-file fallbacks.

Visualization flow:

1. `model_visualization` builds a static two-stage graph from `MODEL_SPECS`.
2. It loads selected model files only when needed.
3. It returns a readable summary for UI cards.

## Error Handling

The UI should use clear Chinese messages for common failures:

- Dataset directory missing.
- A training task is already running.
- Model source path does not exist.
- Training subprocess failed.
- Artifact file is not available yet.
- YOLO model summary could not be loaded.

Failures should be isolated to the relevant tab or model card.

## Testing Strategy

Tests will be added before implementation.

Unit tests should cover:

- Training command construction and parameter validation.
- Concurrent training prevention.
- Parsing of `results.csv`, `args.yaml`, and `train_summary.json`.
- Missing artifact fallbacks.
- Two-stage visualization data.
- YOLO model summary fallback for missing or invalid files.

Existing tests for severity visualization and core config behavior should continue passing.

UI wiring can be covered with focused tests for helper functions and a smoke test that `build_demo` still creates a Gradio Blocks app.

## Repository Hygiene

The repository currently contains existing conflict-marker text in documentation-related files. This design does not require a broad documentation rewrite. If those markers interfere with the implementation or user-facing instructions, clean only the minimum affected lines and keep that work separate from the training workbench logic.
