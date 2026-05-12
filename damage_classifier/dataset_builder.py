from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from .config import (
    CASCADE_LEVELS,
    DAMAGE_DIRNAME,
    DATASETS_DIR,
    DEFAULT_CROP_SIZE,
    DEFAULT_IMAGE_SIZE,
    METADATA_DIR,
    MODEL_SPECS,
    RAW_DATA_DIR,
    SPOT_DIRNAME,
    ensure_runtime_dirs,
    severity_bin_for_index,
    split_for_index,
)
from .preprocessing import parse_image_index, save_preprocessed_image


def _sorted_images(directory: Path) -> list[Path]:
    images = sorted(directory.glob("img_*.jpg"))
    return sorted(images, key=parse_image_index)


def _validate_level_directory(level_dir: Path) -> list[Path]:
    damage_dir = level_dir / DAMAGE_DIRNAME
    spot_dir = level_dir / SPOT_DIRNAME
    if not damage_dir.is_dir():
        raise FileNotFoundError(f"Missing damage image directory: {damage_dir}")
    if not spot_dir.is_dir():
        raise FileNotFoundError(f"Missing spot image directory: {spot_dir}")

    damage_images = _sorted_images(damage_dir)
    spot_images = _sorted_images(spot_dir)
    if len(damage_images) != 121 or len(spot_images) != 121:
        raise ValueError(f"{level_dir.name} must contain 121 damage images and 121 spot images")

    damage_names = [path.name for path in damage_images]
    spot_names = [path.name for path in spot_images]
    if damage_names != spot_names:
        raise ValueError(f"{level_dir.name} damage and spot images are not one-to-one matched")

    expected = [f"img_{i:03d}.jpg" for i in range(1, 122)]
    if spot_names != expected:
        raise ValueError(f"{level_dir.name} image sequence must be continuous from img_001.jpg to img_121.jpg")
    return spot_images


def _reset_dataset_dirs() -> None:
    for dataset_spec in MODEL_SPECS.values():
        dataset_dir = dataset_spec["dataset_dir"]
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
    if METADATA_DIR.exists():
        shutil.rmtree(METADATA_DIR)


def build_datasets(
    source_dir: Path = RAW_DATA_DIR,
    crop_size: int = DEFAULT_CROP_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    clean: bool = False,
) -> dict[str, object]:
    ensure_runtime_dirs()
    if clean:
        _reset_dataset_dirs()

    rows: list[dict[str, object]] = []
    counts = defaultdict(int)

    for level_name in CASCADE_LEVELS:
        level_dir = source_dir / level_name
        if not level_dir.is_dir():
            raise FileNotFoundError(f"Missing cascade level directory: {level_dir}")

        for spot_path in _validate_level_directory(level_dir):
            index = parse_image_index(spot_path)
            split = split_for_index(index)
            severity = severity_bin_for_index(index)
            base_name = f"{level_name}_{spot_path.name}"

            cascade_out = DATASETS_DIR / "cascade_level" / split / level_name / base_name
            save_preprocessed_image(spot_path, cascade_out, crop_size=crop_size, output_size=image_size)

            stage_dataset_name = "severity_stage1" if level_name == "第一级" else "severity_stage2"
            severity_out = DATASETS_DIR / stage_dataset_name / split / severity / base_name
            save_preprocessed_image(spot_path, severity_out, crop_size=crop_size, output_size=image_size)

            rows.append(
                {
                    "raw_path": str(spot_path),
                    "cascade_level": level_name,
                    "image_index": index,
                    "split": split,
                    "severity_bin": severity,
                    "cascade_dataset_path": str(cascade_out),
                    "severity_dataset_path": str(severity_out),
                }
            )
            counts[f"{level_name}_{split}"] += 1
            counts[f"{level_name}_{severity}_{split}"] += 1

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = METADATA_DIR / "samples.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_path",
                "cascade_level",
                "image_index",
                "split",
                "severity_bin",
                "cascade_dataset_path",
                "severity_dataset_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_dir": str(source_dir),
        "crop_size": crop_size,
        "image_size": image_size,
        "total_samples": len(rows),
        "counts": dict(sorted(counts.items())),
        "metadata_csv": str(metadata_csv),
    }
    summary_json = METADATA_DIR / "dataset_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建模拟损伤光斑图分类数据集")
    parser.add_argument("--source", default=str(RAW_DATA_DIR), help="原始模拟图目录")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE, help="中心裁切边长")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="输出图像边长")
    parser.add_argument("--clean", action="store_true", help="重建数据集目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_datasets(
        source_dir=Path(args.source),
        crop_size=args.crop_size,
        image_size=args.image_size,
        clean=args.clean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
