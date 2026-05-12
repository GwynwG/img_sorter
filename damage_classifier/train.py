<<<<<<< HEAD
﻿from __future__ import annotations
=======
from __future__ import annotations
>>>>>>> 6ed2eb1deaef4a6128758f9b78052bf38af78ac2

import argparse
import json
import os
import shutil
from pathlib import Path

from .archive import archive_directory, archive_json
from .config import (
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    MODEL_SPECS,
    RUNS_DIR,
    YOLO_CONFIG_DIR,
    ensure_runtime_dirs,
)
from .dataset_builder import build_datasets

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import ultralytics.data.dataset as ultralytics_dataset
from ultralytics import YOLO


def _verify_images_sequential(self) -> list[tuple]:
    desc = f"{self.prefix}Scanning {self.root}..."
    path = Path(self.root).with_suffix(".cache")

    try:
        ultralytics_dataset.check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)
        cache = ultralytics_dataset.load_dataset_cache_file(path)
        assert cache["version"] == ultralytics_dataset.DATASET_CACHE_VERSION
        assert cache["hash"] == ultralytics_dataset.get_hash([x[0] for x in self.samples])
        nf, nc, n, samples = cache.pop("results")
        if ultralytics_dataset.LOCAL_RANK in {-1, 0}:
            desc_text = f"{desc} {nf} images, {nc} corrupt"
            ultralytics_dataset.TQDM(None, desc=desc_text, total=n, initial=n)
            if cache["msgs"]:
                ultralytics_dataset.LOGGER.info("\n".join(cache["msgs"]))
        return samples
    except (FileNotFoundError, AssertionError, AttributeError):
        nf, nc, msgs, samples, cache_dict = 0, 0, [], [], {}
        pbar = ultralytics_dataset.TQDM(self.samples, desc=desc, total=len(self.samples))
        for sample in pbar:
            verified, nf_f, nc_f, msg = ultralytics_dataset.verify_image((sample, self.prefix))
            if nf_f:
                samples.append(verified)
            if msg:
                msgs.append(msg)
            nf += nf_f
            nc += nc_f
            pbar.desc = f"{desc} {nf} images, {nc} corrupt"
        pbar.close()
        if msgs:
            ultralytics_dataset.LOGGER.info("\n".join(msgs))
        cache_dict["hash"] = ultralytics_dataset.get_hash([x[0] for x in self.samples])
        cache_dict["results"] = nf, nc, len(samples), samples
        cache_dict["msgs"] = msgs
        ultralytics_dataset.save_dataset_cache_file(self.prefix, path, cache_dict, ultralytics_dataset.DATASET_CACHE_VERSION)
        return samples


def _select_model_source(model_path: str | None) -> str:
    if not model_path:
        return DEFAULT_MODEL
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    if path.suffix.lower() == ".pt" and "cls" not in path.stem.lower():
        raise ValueError("Classification training expects a *-cls.pt checkpoint or a classify yaml model definition")
    return str(path)


def train_one(
    model_name: str,
    model_source: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
) -> dict[str, str]:
    ensure_runtime_dirs()
    os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)
    spec = MODEL_SPECS[model_name]
    dataset_dir = spec["dataset_dir"]
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}. Run dataset builder first.")

    ultralytics_dataset.ClassificationDataset.verify_images = _verify_images_sequential
    model = YOLO(model_source)
    model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(RUNS_DIR),
        name=model_name,
        exist_ok=True,
        verbose=True,
        pretrained=model_source if model_source.endswith(".pt") else False,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    artifact_dir = spec["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    exported_best = artifact_dir / "best.pt"
    exported_last = artifact_dir / "last.pt"
    shutil.copy2(best_path, exported_best)
    shutil.copy2(last_path, exported_last)

    summary = {
        "model_name": model_name,
        "display_name": spec["display_name"],
        "dataset_dir": str(dataset_dir),
        "run_dir": str(save_dir),
        "best_path": str(exported_best),
        "last_path": str(exported_last),
        "epochs": str(epochs),
        "imgsz": str(imgsz),
        "batch": str(batch),
        "device": device,
    }
    summary_path = artifact_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    archived_run_dir = archive_directory(save_dir, "runs", model_name)
    archived_model_dir = archive_directory(artifact_dir, "models", model_name)
    summary["archived_run_dir"] = str(archived_run_dir)
    summary["archived_model_dir"] = str(archived_model_dir)
    summary["archived_summary_path"] = str(archive_json(summary, "summaries", f"{model_name}_train_summary"))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练模拟损伤光斑图分类模型")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()) + ["all"],
        default=["all"],
        help="要训练的模型",
    )
    parser.add_argument("--model-source", default=None, help="分类模型 yaml 或 *-cls.pt 路径")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="训练设备，如 cpu 或 0")
    parser.add_argument("--workers", type=int, default=0, help="数据加载 worker 数")
    parser.add_argument("--rebuild-dataset", action="store_true", help="训练前重建数据集")
    parser.add_argument("--clean-dataset", action="store_true", help="配合 --rebuild-dataset 清理旧数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_dataset:
        build_datasets(clean=args.clean_dataset, image_size=args.imgsz)

    selected = list(MODEL_SPECS.keys()) if "all" in args.models else args.models
    model_source = _select_model_source(args.model_source)
    summaries = []
    for model_name in selected:
        print(f"[INFO] 训练 {model_name} ...")
        summaries.append(
            train_one(
                model_name=model_name,
                model_source=model_source,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
