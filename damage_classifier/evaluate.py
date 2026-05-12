from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .archive import archive_json
from .config import METADATA_DIR, SEVERITY_BINS, default_model_paths
from .inference import DamagePredictionPipeline


def _severity_distance(left: str, right: str) -> int:
    order = {name: idx for idx, name in enumerate(SEVERITY_BINS.keys())}
    return abs(order[left] - order[right])


def evaluate_pipeline(metadata_csv: Path, pipeline: DamagePredictionPipeline) -> dict[str, float]:
    total = 0
    cascade_correct = 0
    severity_correct = 0
    severity_relaxed = 0

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["split"] == "val"]

    for row in rows:
        total += 1
        prediction = pipeline.predict(row["raw_path"])
        if prediction.cascade_level == row["cascade_level"]:
            cascade_correct += 1
        if prediction.severity_bin == row["severity_bin"]:
            severity_correct += 1
        if _severity_distance(prediction.severity_bin, row["severity_bin"]) <= 1:
            severity_relaxed += 1

    if total == 0:
        raise ValueError("No validation rows were found in metadata CSV")

    return {
        "val_samples": total,
        "cascade_accuracy": round(cascade_correct / total, 4),
        "severity_top1_accuracy": round(severity_correct / total, 4),
        "severity_relaxed_accuracy": round(severity_relaxed / total, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估两阶段模拟损伤光斑图分类模型")
    parser.add_argument("--metadata", default=str(METADATA_DIR / "samples.csv"), help="数据集元数据 CSV")
    defaults = default_model_paths()
    parser.add_argument("--cascade-model", default=str(defaults["cascade_level"]), help="级联位置模型路径")
    parser.add_argument("--stage1-model", default=str(defaults["severity_stage1"]), help="第一级程度模型路径")
    parser.add_argument("--stage2-model", default=str(defaults["severity_stage2"]), help="第二级程度模型路径")
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu 或 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DamagePredictionPipeline(
        cascade_model_path=args.cascade_model,
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        device=args.device,
    )
    metrics = evaluate_pipeline(Path(args.metadata), pipeline)
    metrics["metadata_csv"] = args.metadata
    metrics["cascade_model"] = args.cascade_model
    metrics["stage1_model"] = args.stage1_model
    metrics["stage2_model"] = args.stage2_model

    latest_path = METADATA_DIR / "evaluation_latest.json"
    latest_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["archived_evaluation_path"] = str(archive_json(metrics, "evaluations", "evaluation"))
    latest_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
