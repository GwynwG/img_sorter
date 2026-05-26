from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import MODEL_SPECS, MODELS_DIR, RUNS_DIR


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


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _clean_yaml_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_simple_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _clean_yaml_value(value)
    return values


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_result_metrics(path: Path) -> tuple[int, float | None, float | None]:
    if not path.exists():
        return 0, None, None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return 0, None, None

    top1_values = [
        value
        for value in (_float_or_none(row.get("metrics/accuracy_top1")) for row in rows)
        if value is not None
    ]
    top1_accuracy = max(top1_values) if top1_values else None
    val_loss = _float_or_none(rows[-1].get("val/loss"))
    return len(rows), top1_accuracy, val_loss


def summarize_model_artifacts(
    model_name: str,
    *,
    display_name: str | None = None,
    run_dir: Path | None = None,
    model_dir: Path | None = None,
) -> ModelArtifactSummary:
    spec = MODEL_SPECS.get(model_name, {})
    resolved_display_name = display_name or str(spec.get("display_name", model_name))
    resolved_run_dir = Path(run_dir or RUNS_DIR / model_name)
    resolved_model_dir = Path(model_dir or spec.get("artifact_dir", MODELS_DIR / model_name))

    args = _read_simple_yaml(resolved_run_dir / "args.yaml")
    results_csv = _existing_path(resolved_run_dir / "results.csv")
    epoch_count, top1_accuracy, val_loss = _read_result_metrics(results_csv) if results_csv else (0, None, None)

    best_path = _existing_path(resolved_model_dir / "best.pt")
    last_path = _existing_path(resolved_model_dir / "last.pt")
    train_summary_path = _existing_path(resolved_model_dir / "train_summary.json")
    status = "available" if results_csv or best_path or train_summary_path else "missing"

    return ModelArtifactSummary(
        model_name=model_name,
        display_name=resolved_display_name,
        run_dir=resolved_run_dir,
        model_dir=resolved_model_dir,
        args=args,
        epoch_count=epoch_count,
        top1_accuracy=top1_accuracy,
        val_loss=val_loss,
        best_path=best_path,
        last_path=last_path,
        results_csv=results_csv,
        results_png=_existing_path(resolved_run_dir / "results.png"),
        confusion_matrix=_existing_path(resolved_run_dir / "confusion_matrix.png"),
        confusion_matrix_normalized=_existing_path(resolved_run_dir / "confusion_matrix_normalized.png"),
        train_summary_path=train_summary_path,
        status=status,
    )


def summarize_all_artifacts() -> list[ModelArtifactSummary]:
    return [summarize_model_artifacts(model_name) for model_name in MODEL_SPECS]
