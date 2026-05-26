from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_IMAGE_SIZE, MODEL_SPECS, default_model_paths


@dataclass(frozen=True)
class PipelineNode:
    id: str
    label: str
    kind: str


@dataclass(frozen=True)
class PipelineEdge:
    source: str
    target: str
    label: str


@dataclass(frozen=True)
class PipelineGraph:
    nodes: tuple[PipelineNode, ...]
    edges: tuple[PipelineEdge, ...]


@dataclass(frozen=True)
class ModelStructureSummary:
    model_name: str
    display_name: str
    model_path: Path
    available: bool
    message: str
    class_labels: tuple[str, ...]
    image_size: int
    layer_count: int | None
    parameter_count: int | None


def build_pipeline_graph() -> PipelineGraph:
    nodes = (
        PipelineNode("input", "输入损伤光斑图", "input"),
        PipelineNode("cascade_level", "级联位置分类器", "model"),
        PipelineNode("severity_stage1", "第一级损伤程度分类器", "model"),
        PipelineNode("severity_stage2", "第二级损伤程度分类器", "model"),
        PipelineNode("severity_output", "S1 / S2 / S3 / S4", "output"),
    )
    edges = (
        PipelineEdge("input", "cascade_level", "判断第一级 / 第二级"),
        PipelineEdge("cascade_level", "severity_stage1", "若为第一级"),
        PipelineEdge("cascade_level", "severity_stage2", "若为第二级"),
        PipelineEdge("severity_stage1", "severity_output", "输出损伤阶段"),
        PipelineEdge("severity_stage2", "severity_output", "输出损伤阶段"),
    )
    return PipelineGraph(nodes=nodes, edges=edges)


def summarize_yolo_model(
    model_name: str,
    model_path: Path | str,
    *,
    display_name: str | None = None,
    class_labels: tuple[str, ...] = (),
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> ModelStructureSummary:
    resolved_path = Path(model_path)
    resolved_display_name = display_name or str(MODEL_SPECS.get(model_name, {}).get("display_name", model_name))
    if not resolved_path.exists():
        return ModelStructureSummary(
            model_name=model_name,
            display_name=resolved_display_name,
            model_path=resolved_path,
            available=False,
            message=f"未找到模型文件: {resolved_path}",
            class_labels=class_labels,
            image_size=image_size,
            layer_count=None,
            parameter_count=None,
        )

    try:
        from ultralytics import YOLO

        yolo = YOLO(str(resolved_path))
        torch_model = getattr(yolo, "model", None)
        layer_count = len(list(torch_model.modules())) if torch_model is not None else None
        parameter_count = (
            int(sum(parameter.numel() for parameter in torch_model.parameters()))
            if torch_model is not None
            else None
        )
        names = getattr(yolo, "names", None)
        labels = tuple(str(value) for value in names.values()) if isinstance(names, dict) else class_labels
        return ModelStructureSummary(
            model_name=model_name,
            display_name=resolved_display_name,
            model_path=resolved_path,
            available=True,
            message="模型结构读取成功",
            class_labels=labels or class_labels,
            image_size=image_size,
            layer_count=layer_count,
            parameter_count=parameter_count,
        )
    except Exception as exc:
        return ModelStructureSummary(
            model_name=model_name,
            display_name=resolved_display_name,
            model_path=resolved_path,
            available=False,
            message=f"模型结构读取失败: {exc}",
            class_labels=class_labels,
            image_size=image_size,
            layer_count=None,
            parameter_count=None,
        )


def summarize_default_models(
    *,
    model_paths: dict[str, Path] | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[ModelStructureSummary]:
    paths = model_paths or default_model_paths()
    summaries: list[ModelStructureSummary] = []
    for model_name, spec in MODEL_SPECS.items():
        summaries.append(
            summarize_yolo_model(
                model_name,
                paths[model_name],
                display_name=str(spec["display_name"]),
                class_labels=tuple(spec["classes"]),
                image_size=image_size,
            )
        )
    return summaries
