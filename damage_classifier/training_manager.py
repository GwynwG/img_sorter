from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import MODEL_SPECS, PROJECT_ROOT, RUNS_DIR

TRAINING_TARGET_LABELS = {
    "全部模型": "all",
    "级联位置分类器": "cascade_level",
    "第一级损伤程度分类器": "severity_stage1",
    "第二级损伤程度分类器": "severity_stage2",
}

TRAINING_PRESETS = {
    "快速验证": {"epochs": 5, "imgsz": 320, "batch": 8, "workers": 0},
    "常规训练": {"epochs": 40, "imgsz": 320, "batch": 16, "workers": 0},
    "较长训练": {"epochs": 200, "imgsz": 320, "batch": 16, "workers": 0},
}


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


def normalize_model_target(model_target: str) -> str:
    normalized = TRAINING_TARGET_LABELS.get(model_target, model_target)
    allowed = set(MODEL_SPECS) | {"all"}
    if normalized not in allowed:
        raise ValueError(f"未知训练目标: {model_target}")
    return normalized


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def validate_training_request(request: TrainingRequest) -> None:
    normalize_model_target(request.model_target)
    _validate_positive_int("epochs", request.epochs)
    _validate_positive_int("imgsz", request.imgsz)
    _validate_positive_int("batch", request.batch)
    if request.workers < 0:
        raise ValueError("workers 不能小于 0")
    if not request.device.strip():
        raise ValueError("device 不能为空")


def build_training_command(
    request: TrainingRequest,
    *,
    python_executable: str | None = None,
    script_path: Path | None = None,
) -> list[str]:
    validate_training_request(request)
    target = normalize_model_target(request.model_target)
    command = [
        python_executable or sys.executable,
        str(script_path or PROJECT_ROOT / "train_damage_models.py"),
        "--models",
        target,
        "--epochs",
        str(request.epochs),
        "--imgsz",
        str(request.imgsz),
        "--batch",
        str(request.batch),
        "--device",
        request.device,
        "--workers",
        str(request.workers),
    ]
    if request.model_source:
        command.extend(["--model-source", request.model_source])
    return command


class TrainingManager:
    def __init__(
        self,
        *,
        log_dir: Path | None = None,
        validate_datasets: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir or RUNS_DIR.parent / "training_logs")
        self.validate_datasets = validate_datasets
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._snapshot = TrainingSnapshot(
            state="idle",
            model_target=None,
            command=[],
            log_path=None,
            returncode=None,
            message="暂无训练任务",
            started_at=None,
            finished_at=None,
        )

    def _required_model_names(self, model_target: str) -> list[str]:
        normalized = normalize_model_target(model_target)
        return list(MODEL_SPECS) if normalized == "all" else [normalized]

    def _ensure_datasets_exist(self, request: TrainingRequest) -> None:
        if not self.validate_datasets:
            return
        if request.model_source and not Path(request.model_source).exists():
            raise FileNotFoundError(f"模型源文件不存在: {request.model_source}")
        missing = [
            str(MODEL_SPECS[model_name]["dataset_dir"])
            for model_name in self._required_model_names(request.model_target)
            if not MODEL_SPECS[model_name]["dataset_dir"].exists()
        ]
        if missing:
            raise FileNotFoundError("找不到训练数据集，请先构建数据集: " + "; ".join(missing))

    def _refresh_snapshot_unlocked(self) -> TrainingSnapshot:
        if self._process is None or self._snapshot.state != "running":
            return self._snapshot

        returncode = self._process.poll()
        if returncode is None:
            return self._snapshot

        state = "finished" if returncode == 0 else "failed"
        message = "训练完成" if returncode == 0 else f"训练失败，退出码 {returncode}"
        self._snapshot = TrainingSnapshot(
            state=state,
            model_target=self._snapshot.model_target,
            command=self._snapshot.command,
            log_path=self._snapshot.log_path,
            returncode=returncode,
            message=message,
            started_at=self._snapshot.started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return self._snapshot

    def snapshot(self) -> TrainingSnapshot:
        with self._lock:
            return self._refresh_snapshot_unlocked()

    def start_training(self, request: TrainingRequest) -> TrainingSnapshot:
        with self._lock:
            current = self._refresh_snapshot_unlocked()
            if current.state == "running":
                raise RuntimeError("已有训练任务正在运行，请等待完成后再启动新的训练")

            self._ensure_datasets_exist(request)
            command = build_training_command(request)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_target = normalize_model_target(request.model_target)
            log_path = self.log_dir / f"training_{model_target}_{timestamp}.log"

            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            self._process = process
            self._snapshot = TrainingSnapshot(
                state="running",
                model_target=model_target,
                command=command,
                log_path=log_path,
                returncode=None,
                message="训练已启动",
                started_at=datetime.now().isoformat(timespec="seconds"),
                finished_at=None,
            )
            return self._snapshot

    def tail_log(self, line_count: int = 80) -> str:
        snapshot = self.snapshot()
        if snapshot.log_path is None or not snapshot.log_path.exists():
            return ""
        lines = snapshot.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-line_count:])
