# Training Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gradio tabs for training, lightweight hyperparameter control, training artifact review, and neural-network structure visualization.

**Architecture:** Keep inference code stable and add focused helper modules for training jobs, artifact parsing, and model visualization. `gradio_app.py` will become the UI composer that wires the existing prediction flow plus new tabs to those helpers.

**Tech Stack:** Python stdlib, unittest, Gradio, pandas/openpyxl already present, Ultralytics YOLO already present.

**User Preference:** Do not commit implementation changes. Replace commit steps with `git status` checkpoints so the user can commit later.

---

### Task 1: Training Artifact Reader

**Files:**
- Create: `damage_classifier/training_artifacts.py`
- Test: `tests/test_training_artifacts.py`

- [ ] **Step 1: Write failing tests for current/missing artifact summaries**

Add tests that create temporary model artifact directories, write a small `results.csv`, `args.yaml`, `train_summary.json`, and assert that a summary contains epoch count, final metrics, image paths, and missing-file fallbacks.

Run: `python -m unittest tests.test_training_artifacts -v`
Expected: FAIL because `damage_classifier.training_artifacts` does not exist.

- [ ] **Step 2: Implement artifact dataclasses and parsing**

Create stable dataclasses:

```python
@dataclass(frozen=True)
class ModelArtifactSummary:
    model_name: str
    display_name: str
    run_dir: Path
    model_dir: Path
    args: dict[str, str]
    epoch_count: int
    top1_accuracy: float | None
    val_loss: float | None
    best_path: Path | None
    last_path: Path | None
    results_csv: Path | None
    results_png: Path | None
    confusion_matrix: Path | None
    confusion_matrix_normalized: Path | None
    train_summary_path: Path | None
    status: str
```

Add helpers `summarize_model_artifacts(model_name)` and `summarize_all_artifacts()`.

- [ ] **Step 3: Run artifact tests until green**

Run: `python -m unittest tests.test_training_artifacts -v`
Expected: PASS.

### Task 2: Training Manager

**Files:**
- Create: `damage_classifier/training_manager.py`
- Test: `tests/test_training_manager.py`

- [ ] **Step 1: Write failing tests for request validation and command construction**

Test that a request for `all` maps to `--models all`, a single model maps to its model key, and parameters are rendered as CLI arguments for `train_damage_models.py`.

Run: `python -m unittest tests.test_training_manager -v`
Expected: FAIL because `damage_classifier.training_manager` does not exist.

- [ ] **Step 2: Write failing tests for concurrency guard**

Patch `subprocess.Popen` with a fake running process and assert that a second `start_training()` call raises a clear error while the first job is still running.

Run: `python -m unittest tests.test_training_manager -v`
Expected: FAIL until the manager exists.

- [ ] **Step 3: Implement the manager**

Create:

```python
@dataclass(frozen=True)
class TrainingRequest:
    model_target: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    workers: int
    model_source: str | None = None

@dataclass(frozen=True)
class TrainingSnapshot:
    state: str
    model_target: str | None
    command: list[str]
    log_path: Path | None
    returncode: int | None
    message: str
    started_at: str | None
    finished_at: str | None
```

Use `subprocess.Popen` to run `sys.executable train_damage_models.py ...`, redirect stdout/stderr to a timestamped log under `damage_artifacts/training_logs`, and provide `snapshot()` and `tail_log()`.

- [ ] **Step 4: Run manager tests until green**

Run: `python -m unittest tests.test_training_manager -v`
Expected: PASS.

### Task 3: Model Visualization

**Files:**
- Create: `damage_classifier/model_visualization.py`
- Test: `tests/test_model_visualization.py`

- [ ] **Step 1: Write failing tests for the two-stage pipeline graph**

Assert that the graph includes input, cascade model, both severity models, and `S1-S4` outputs with clear Chinese labels.

Run: `python -m unittest tests.test_model_visualization -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 2: Write failing tests for missing model summary fallback**

Assert that summarizing a missing `.pt` file returns `available=False` and a readable message instead of raising.

Run: `python -m unittest tests.test_model_visualization -v`
Expected: FAIL until implemented.

- [ ] **Step 3: Implement graph and model summary helpers**

Create dataclasses for graph nodes, graph edges, and model summaries. Load Ultralytics YOLO only inside `summarize_yolo_model()` so tests for missing files do not need a real model.

- [ ] **Step 4: Run visualization tests until green**

Run: `python -m unittest tests.test_model_visualization -v`
Expected: PASS.

### Task 4: Gradio Tabs and UI Wiring

**Files:**
- Modify: `damage_classifier/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Add a failing smoke test for tabbed demo construction**

Extend `tests/test_gradio_app.py` with a test that creates a `DamagePredictionPipeline` using placeholder paths and verifies `build_demo()` returns a Gradio `Blocks` object without loading model weights.

Run: `python -m unittest tests.test_gradio_app -v`
Expected: PASS may already occur; if so, keep it as a regression smoke test before UI edits.

- [ ] **Step 2: Refactor current UI into `单张识别` and `批量识别` tabs**

Move the existing single-image controls into the first tab and batch controls into the second tab. Keep existing prediction functions, outputs, CSS classes, and Excel export behavior.

- [ ] **Step 3: Add `训练工作台` tab**

Wire controls to a module-level `TrainingManager` instance. Add a start button, refresh button, status text, log textbox, latest artifact table, and clear messages for missing datasets or concurrent training.

- [ ] **Step 4: Add `模型分析` tab**

Use `summarize_all_artifacts()` to render one row per model, plus image previews for `results.png` and confusion matrices when available.

- [ ] **Step 5: Add `神经网络结构` tab**

Use `build_pipeline_graph()` and `summarize_default_models()` to render the two-stage flow and model summary cards.

- [ ] **Step 6: Run Gradio smoke tests**

Run: `python -m unittest tests.test_gradio_app -v`
Expected: PASS.

### Task 5: Full Verification

**Files:**
- No new files unless tests reveal a narrow fix.

- [ ] **Step 1: Run all unit tests**

Run: `python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 2: Start the Gradio app for a manual smoke check**

Run: `python launch_damage_gradio.py --device cpu`
Expected: the app starts and exposes a local URL.

- [ ] **Step 3: Inspect git state**

Run: `git status --short`
Expected: only intentional implementation/test/plan changes are listed. No commit is made.
