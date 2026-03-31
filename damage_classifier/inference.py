from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_CROP_SIZE, DEFAULT_IMAGE_SIZE, YOLO_CONFIG_DIR, default_model_paths, severity_range_text
from .preprocessing import preprocess_image

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
from ultralytics import YOLO


@dataclass
class DamagePrediction:
    cascade_level: str
    cascade_confidence: float
    severity_bin: str
    severity_confidence: float
    severity_range_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DamagePredictionPipeline:
    def __init__(
        self,
        cascade_model_path: Path | str | None = None,
        stage1_model_path: Path | str | None = None,
        stage2_model_path: Path | str | None = None,
        crop_size: int = DEFAULT_CROP_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str = "cpu",
    ) -> None:
        os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)
        default_paths = default_model_paths()
        self.crop_size = crop_size
        self.image_size = image_size
        self.device = device
        self.cascade_model_path = Path(cascade_model_path or default_paths["cascade_level"])
        self.stage1_model_path = Path(stage1_model_path or default_paths["severity_stage1"])
        self.stage2_model_path = Path(stage2_model_path or default_paths["severity_stage2"])
        self._cascade_model: YOLO | None = None
        self._stage1_model: YOLO | None = None
        self._stage2_model: YOLO | None = None

    def ensure_models_ready(self) -> None:
        for path in (self.cascade_model_path, self.stage1_model_path, self.stage2_model_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到模型文件: {path}。请先运行训练脚本，或把训练好的 best.pt 放到默认模型目录下。"
                )

    @property
    def cascade_model(self) -> YOLO:
        if self._cascade_model is None:
            self.ensure_models_ready()
            self._cascade_model = YOLO(str(self.cascade_model_path))
        return self._cascade_model

    @property
    def stage1_model(self) -> YOLO:
        if self._stage1_model is None:
            self.ensure_models_ready()
            self._stage1_model = YOLO(str(self.stage1_model_path))
        return self._stage1_model

    @property
    def stage2_model(self) -> YOLO:
        if self._stage2_model is None:
            self.ensure_models_ready()
            self._stage2_model = YOLO(str(self.stage2_model_path))
        return self._stage2_model

    def _predict_label(self, model: YOLO, image_path: Path) -> tuple[str, float]:
        processed = preprocess_image(image_path, crop_size=self.crop_size, output_size=self.image_size)
        array = np.asarray(processed)
        result = model.predict(source=array, imgsz=self.image_size, device=self.device, verbose=False)[0]
        if result.probs is None:
            raise RuntimeError("Classification result did not include probabilities")
        label_index = int(result.probs.top1)
        confidence = float(result.probs.top1conf.item())
        label = str(result.names[label_index])
        return label, confidence

    def predict(self, image_path: Path | str) -> DamagePrediction:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"输入图片不存在: {path}")

        cascade_level, cascade_conf = self._predict_label(self.cascade_model, path)
        severity_model = self.stage1_model if cascade_level == "第一级" else self.stage2_model
        severity_bin, severity_conf = self._predict_label(severity_model, path)
        return DamagePrediction(
            cascade_level=cascade_level,
            cascade_confidence=round(cascade_conf, 4),
            severity_bin=severity_bin,
            severity_confidence=round(severity_conf, 4),
            severity_range_text=severity_range_text(severity_bin),
        )
