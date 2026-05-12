<<<<<<< HEAD
﻿from __future__ import annotations
=======
from __future__ import annotations
>>>>>>> 6ed2eb1deaef4a6128758f9b78052bf38af78ac2

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "模拟图"
ARTIFACTS_DIR = PROJECT_ROOT / "damage_artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
RUNS_DIR = ARTIFACTS_DIR / "runs"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"

DAMAGE_DIRNAME = "损伤图"
SPOT_DIRNAME = "损伤光斑图"
CASCADE_LEVELS = ("第一级", "第二级")

DEFAULT_CROP_SIZE = 1200
DEFAULT_IMAGE_SIZE = 320
DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 16
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL = "yolo11n-cls.yaml"

SEVERITY_BINS = {
    "S1": range(1, 31),
    "S2": range(31, 61),
    "S3": range(61, 91),
    "S4": range(91, 122),
}

SEVERITY_RANGE_TEXT = {
    "S1": "轻度损伤，对应模拟序列约 001-030 段",
    "S2": "较轻到中度损伤，对应模拟序列约 031-060 段",
    "S3": "中度到较重损伤，对应模拟序列约 061-090 段",
    "S4": "较重到重度损伤，对应模拟序列约 091-121 段",
}

SPLIT_RANGES = {
    "train": ((1, 24), (31, 54), (61, 84), (91, 114)),
    "val": ((25, 30), (55, 60), (85, 90), (115, 121)),
}

MODEL_SPECS = {
    "cascade_level": {
        "dataset_dir": DATASETS_DIR / "cascade_level",
        "artifact_dir": MODELS_DIR / "cascade_level",
        "display_name": "级联位置分类器",
        "classes": CASCADE_LEVELS,
    },
    "severity_stage1": {
        "dataset_dir": DATASETS_DIR / "severity_stage1",
        "artifact_dir": MODELS_DIR / "severity_stage1",
        "display_name": "第一级损伤程度分类器",
        "classes": tuple(SEVERITY_BINS.keys()),
    },
    "severity_stage2": {
        "dataset_dir": DATASETS_DIR / "severity_stage2",
        "artifact_dir": MODELS_DIR / "severity_stage2",
        "display_name": "第二级损伤程度分类器",
        "classes": tuple(SEVERITY_BINS.keys()),
    },
}


def ensure_runtime_dirs() -> None:
    for path in (ARTIFACTS_DIR, DATASETS_DIR, MODELS_DIR, METADATA_DIR, RUNS_DIR, YOLO_CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def severity_bin_for_index(index: int) -> str:
    for name, members in SEVERITY_BINS.items():
        if index in members:
            return name
    raise ValueError(f"Unsupported image index: {index}")


def split_for_index(index: int) -> str:
    for split_name, ranges in SPLIT_RANGES.items():
        for start, end in ranges:
            if start <= index <= end:
                return split_name
    raise ValueError(f"Image index {index} does not belong to any configured split")


def severity_range_text(bin_name: str) -> str:
    if bin_name not in SEVERITY_RANGE_TEXT:
        raise KeyError(f"Unknown severity bin: {bin_name}")
    return SEVERITY_RANGE_TEXT[bin_name]


def default_model_paths() -> dict[str, Path]:
    return {name: spec["artifact_dir"] / "best.pt" for name, spec in MODEL_SPECS.items()}
