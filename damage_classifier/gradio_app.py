from __future__ import annotations

import argparse
import base64
import html
import html
import io
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import MODEL_SPECS, SEVERITY_BINS, default_model_paths
from .inference import DamagePredictionPipeline
from .model_visualization import build_pipeline_graph, summarize_yolo_model
from .training_artifacts import ModelArtifactSummary, summarize_all_artifacts
from .training_manager import TRAINING_PRESETS, TrainingManager, TrainingRequest


APP_CSS = """
:root {
    --jg-ink: #1f2933;
    --jg-muted: #5d6978;
    --jg-panel: rgba(255, 255, 255, 0.92);
    --jg-line: #d9e2ec;
    --jg-teal: #0f766e;
    --jg-coral: #c2410c;
    --jg-gold: #b7791f;
}

.gradio-container {
    background:
        radial-gradient(circle at 16% 8%, rgba(15, 118, 110, 0.16), transparent 28%),
        radial-gradient(circle at 85% 0%, rgba(194, 65, 12, 0.12), transparent 24%),
        linear-gradient(135deg, #f8faf7 0%, #eef4f3 48%, #fffaf2 100%);
    color: var(--jg-ink);
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}

#app-shell {
    max-width: 1220px;
    margin: 0 auto;
    padding: 28px 18px 36px;
}

.hero-band {
    border: 1px solid rgba(217, 226, 236, 0.88);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(240, 253, 250, 0.82));
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 18px 45px rgba(31, 41, 51, 0.08);
}

.hero-band h1 {
    margin: 0 0 8px;
    color: var(--jg-ink);
    font-size: 32px;
    line-height: 1.2;
    letter-spacing: 0;
}

.hero-band p {
    margin: 0;
    color: var(--jg-muted);
    font-size: 15px;
    line-height: 1.7;
}

.workflow-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 14px;
}

.workflow-strip span {
    border: 1px solid var(--jg-line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.74);
    padding: 10px 12px;
    color: #374151;
    font-size: 13px;
}

.workspace-row {
    align-items: stretch;
    gap: 18px;
}

.input-panel,
.result-panel {
    border: 1px solid rgba(217, 226, 236, 0.96);
    border-radius: 16px;
    background: var(--jg-panel);
    box-shadow: 0 16px 36px rgba(31, 41, 51, 0.08);
    padding: 18px;
}

.panel-heading h2 {
    margin: 0 0 6px;
    font-size: 19px;
    line-height: 1.3;
    letter-spacing: 0;
}

.panel-heading p {
    margin: 0 0 14px;
    color: var(--jg-muted);
    font-size: 13px;
    line-height: 1.65;
}

#image-input {
    border: 1px dashed rgba(15, 118, 110, 0.42);
    border-radius: 14px;
    overflow: hidden;
    background: rgba(240, 253, 250, 0.42);
}

#submit-button {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 48%, #06b6d4 100%) !important;
    border: 0 !important;
    color: #ffffff !important;
    min-height: 44px;
    border-radius: 10px;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.24);
    font-weight: 700;
}

#submit-button:hover {
    filter: brightness(1.04);
    box-shadow: 0 14px 26px rgba(59, 130, 246, 0.26);
}

#stage-visual img {
    border-radius: 14px;
    border: 1px solid var(--jg-line);
    box-shadow: 0 10px 25px rgba(31, 41, 51, 0.08);
}

.compact-output textarea,
.compact-output input {
    font-weight: 650;
}

@media (max-width: 780px) {
    #app-shell {
        padding: 18px 10px 28px;
    }

    .hero-band {
        padding: 18px;
        border-radius: 14px;
    }

    .hero-band h1 {
        font-size: 24px;
    }

    .workflow-strip {
        grid-template-columns: 1fr;
    }

    .input-panel,
    .result-panel {
        padding: 14px;
        border-radius: 14px;
    }
}

/* 批量处理表格样式 */
.batch-section {
    border: 1px solid rgba(217, 226, 236, 0.96);
    border-radius: 16px;
    background: var(--jg-panel);
    box-shadow: 0 16px 36px rgba(31, 41, 51, 0.08);
    padding: 18px;
    margin-top: 24px;
}

.batch-table {
    width: 100%;
    max-width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    margin-top: 16px;
}

.batch-table th {
    background: linear-gradient(135deg, #e7f7f4, #f0fdfa);
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    color: var(--jg-ink);
    border-bottom: 2px solid var(--jg-line);
}

.batch-table td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--jg-line);
    vertical-align: middle;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.batch-table tr:hover {
    background: rgba(240, 253, 250, 0.5);
}

.thumbnail-cell {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid var(--jg-line);
}

/* 级联位置标签 */
.cascade-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 13px;
}

.cascade-badge.level-1 {
    background: #dbeafe;
    color: #1e40af;
}

.cascade-badge.level-2 {
    background: #fce7f3;
    color: #9d174d;
}

/* 进度条样式 */
.progress-bar-container {
    width: 100%;
    height: 24px;
    background: #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
}

.progress-bar {
    height: 100%;
    border-radius: 12px;
    transition: width 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.progress-bar.s1 { background: linear-gradient(90deg, #10b981, #34d399); }
.progress-bar.s2 { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.progress-bar.s3 { background: linear-gradient(90deg, #f97316, #fb923c); }
.progress-bar.s4 { background: linear-gradient(90deg, #ef4444, #f87171); }

.workbench-panel {
    border: 1px solid rgba(217, 226, 236, 0.96);
    border-radius: 16px;
    background: var(--jg-panel);
    box-shadow: 0 16px 36px rgba(31, 41, 51, 0.08);
    padding: 18px;
}

.artifact-table {
    width: 100%;
    max-width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 13px;
}

.artifact-table-wrap {
    width: 100%;
    max-width: 100%;
}

.artifact-table th,
.artifact-table td {
    border-bottom: 1px solid var(--jg-line);
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.artifact-table th {
    color: var(--jg-ink);
    background: #f1f5f9;
    font-weight: 700;
}

.artifact-table th:nth-child(1) { width: 15%; }
.artifact-table th:nth-child(2) { width: 9%; }
.artifact-table th:nth-child(3) { width: 6%; }
.artifact-table th:nth-child(4) { width: 8%; }
.artifact-table th:nth-child(5) { width: 8%; }
.artifact-table th:nth-child(6) { width: 27%; }
.artifact-table th:nth-child(7) { width: 27%; }

.path-cell {
    display: block;
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-all;
    white-space: normal;
    line-height: 1.45;
}

.artifact-image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 18px;
    align-items: start;
    margin-top: 16px;
}

.artifact-image-card {
    border: 1px solid var(--jg-line);
    border-radius: 12px;
    background: #ffffff;
    padding: 12px;
    box-shadow: 0 8px 22px rgba(31, 41, 51, 0.06);
}

.artifact-image-card h3 {
    margin: 0 0 10px;
    font-size: 15px;
    line-height: 1.35;
}

.artifact-image-card img {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #edf2f7;
}

.artifact-image-card p {
    margin: 8px 0 0;
    color: var(--jg-muted);
    font-size: 12px;
    overflow-wrap: anywhere;
    word-break: break-all;
}

.status-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    background: #e0f2fe;
    color: #075985;
    font-weight: 700;
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: normal;
}

.structure-flow {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    align-items: stretch;
    margin: 12px 0 18px;
}

.structure-node {
    border: 1px solid var(--jg-line);
    border-radius: 12px;
    background: #ffffff;
    padding: 12px;
    min-height: 86px;
}

.structure-node strong {
    display: block;
    margin-bottom: 6px;
}

.model-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.model-card {
    border: 1px solid var(--jg-line);
    border-radius: 12px;
    background: #ffffff;
    padding: 14px;
}

.model-card h3 {
    margin: 0 0 8px;
    font-size: 16px;
}

.model-card p {
    margin: 4px 0;
    color: var(--jg-muted);
    font-size: 13px;
    overflow-wrap: anywhere;
}

@media (max-width: 900px) {
    .structure-flow,
    .model-card-grid {
        grid-template-columns: 1fr;
    }
}

.workbench-panel {
    border: 1px solid rgba(217, 226, 236, 0.96);
    border-radius: 16px;
    background: var(--jg-panel);
    box-shadow: 0 16px 36px rgba(31, 41, 51, 0.08);
    padding: 18px;
}

.artifact-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.artifact-table th,
.artifact-table td {
    border-bottom: 1px solid var(--jg-line);
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
}

.artifact-table th {
    color: var(--jg-ink);
    background: #f1f5f9;
    font-weight: 700;
}

.status-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    background: #e0f2fe;
    color: #075985;
    font-weight: 700;
}

.structure-flow {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    align-items: stretch;
    margin: 12px 0 18px;
}

.structure-node {
    border: 1px solid var(--jg-line);
    border-radius: 12px;
    background: #ffffff;
    padding: 12px;
    min-height: 86px;
}

.structure-node strong {
    display: block;
    margin-bottom: 6px;
}

.model-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.model-card {
    border: 1px solid var(--jg-line);
    border-radius: 12px;
    background: #ffffff;
    padding: 14px;
}

.model-card h3 {
    margin: 0 0 8px;
    font-size: 16px;
}

.model-card p {
    margin: 4px 0;
    color: var(--jg-muted);
    font-size: 13px;
    overflow-wrap: anywhere;
}

@media (max-width: 900px) {
    .structure-flow,
    .model-card-grid {
        grid-template-columns: 1fr;
    }
}
"""

STAGE_COLORS = {
    "S1": "#2f9e7e",
    "S2": "#d2a72c",
    "S3": "#d97706",
    "S4": "#be3a2b",
}


def _select_model_path(selected_path: str | None, default_path: Path) -> Path:
    return Path(selected_path) if selected_path else default_path


def _severity_sequence_position(severity_bin: str) -> dict[str, float | int | str]:
    if severity_bin not in SEVERITY_BINS:
        raise KeyError(f"Unknown severity bin: {severity_bin}")

    stage_names = tuple(SEVERITY_BINS.keys())
    members = SEVERITY_BINS[severity_bin]
    start = members.start
    end = members.stop - 1
    total = max(stage_range.stop - 1 for stage_range in SEVERITY_BINS.values())
    midpoint = (start + end) / 2
    midpoint_percent = round(((midpoint - 1) / (total - 1)) * 100, 2)
    return {
        "severity_bin": severity_bin,
        "stage_number": stage_names.index(severity_bin) + 1,
        "stage_count": len(stage_names),
        "start": start,
        "end": end,
        "total": total,
        "midpoint_percent": midpoint_percent,
    }


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    try:
        draw.text(xy, text, font=font, fill=fill)
    except UnicodeEncodeError:
        fallback = text.encode("ascii", "ignore").decode("ascii") or "damage stage"
        draw.text(xy, fallback, font=ImageFont.load_default(), fill=fill)


def _build_severity_visual(severity_bin: str, cascade_level: str, confidence: float) -> Image.Image:
    position = _severity_sequence_position(severity_bin)
    width, height = 960, 420
    image = Image.new("RGB", (width, height), "#f8faf7")
    draw = ImageDraw.Draw(image)

    title_font = _load_font(34, bold=True)
    label_font = _load_font(22, bold=True)
    body_font = _load_font(18)
    small_font = _load_font(15)

    draw.rounded_rectangle((26, 24, width - 26, height - 24), radius=28, fill="#ffffff", outline="#d9e2ec", width=2)
    draw.rounded_rectangle((26, 24, width - 26, 118), radius=28, fill="#e7f7f4")
    draw.rectangle((26, 88, width - 26, 118), fill="#e7f7f4")

    stage_color = STAGE_COLORS[severity_bin]
    _draw_text(draw, (54, 46), f"{cascade_level} / {severity_bin}", font=title_font, fill="#1f2933")
    _draw_text(
        draw,
        (54, 88),
        f"模型判断约位于 {position['start']:03d}-{position['end']:03d} 段，置信度 {confidence:.2%}",
        font=body_font,
        fill="#52606d",
    )

    timeline_left, timeline_top = 78, 176
    timeline_width, timeline_height = width - 156, 58
    gap = 10
    segment_count = len(SEVERITY_BINS)
    segment_width = (timeline_width - gap * (segment_count - 1)) / segment_count

    for index, (stage, members) in enumerate(SEVERITY_BINS.items()):
        x0 = int(timeline_left + index * (segment_width + gap))
        x1 = int(x0 + segment_width)
        y0 = timeline_top
        y1 = timeline_top + timeline_height
        active = stage == severity_bin
        fill = STAGE_COLORS[stage] if active else "#eef2f4"
        outline = STAGE_COLORS[stage] if active else "#d9e2ec"
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=fill, outline=outline, width=2)
        label_fill = "#ffffff" if active else "#52606d"
        _draw_text(draw, (x0 + 18, y0 + 14), stage, font=label_font, fill=label_fill)
        _draw_text(draw, (x0 + 74, y0 + 18), f"{members.start:03d}-{members.stop - 1:03d}", font=small_font, fill=label_fill)

    marker_x = int(timeline_left + (position["midpoint_percent"] / 100) * timeline_width)
    marker_top = timeline_top - 34
    draw.polygon(
        [(marker_x, marker_top), (marker_x - 14, marker_top + 26), (marker_x + 14, marker_top + 26)],
        fill=stage_color,
    )
    draw.line((marker_x, marker_top + 28, marker_x, timeline_top + timeline_height + 26), fill=stage_color, width=4)

    draw.rounded_rectangle((54, 290, width - 54, 358), radius=18, fill="#fff7ed", outline="#fed7aa", width=2)
    _draw_text(
        draw,
        (78, 306),
        "阶段解读",
        font=label_font,
        fill="#9a3412",
    )
    _draw_text(
        draw,
        (196, 310),
        f"当前结果落在整个模拟序列约 {position['midpoint_percent']:.1f}% 的位置，可作为损伤进展的快速读数。",
        font=body_font,
        fill="#5d6978",
    )

    _draw_text(draw, (78, 374), "001 起始", font=small_font, fill="#697586")
    _draw_text(draw, (width - 164, 374), f"{position['total']:03d} 末段", font=small_font, fill="#697586")
    return image


def _get_progress_percent(severity_bin: str) -> int:
    """获取损伤程度对应的进度百分比"""
    stage_map = {"S1": 25, "S2": 50, "S3": 75, "S4": 100}
    return stage_map.get(severity_bin, 0)


def _image_to_base64_thumbnail(image_path: str, size: tuple = (80, 80)) -> str:
    """将图片转换为 base64 编码的缩略图"""
    img = Image.open(image_path)
    img.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_batch_result_html(results: list[dict]) -> str:
    """构建批量处理结果的 HTML 表格"""
    if not results:
        return '<p style="color: var(--jg-muted); text-align: center;">暂无结果，请上传图片并点击"批量识别"</p>'

    rows = ""
    for result in results:
        # 缩略图
        thumb_b64 = _image_to_base64_thumbnail(result["image_path"])
        thumb_html = f'<img src="data:image/jpeg;base64,{thumb_b64}" class="thumbnail-cell" />'

        # 级联位置标签
        cascade_class = "level-1" if result["cascade_level"] == "第一级" else "level-2"
        cascade_html = f'<span class="cascade-badge {cascade_class}">{result["cascade_level"]}</span>'

        # 进度条
        severity_bin = result["severity_bin"]
        progress_pct = _get_progress_percent(severity_bin)
        progress_class = severity_bin.lower() if severity_bin in ("S1", "S2", "S3", "S4") else ""
        progress_html = f'''
        <div class="progress-bar-container">
            <div class="progress-bar {progress_class}" style="width: {progress_pct}%">
                {severity_bin} ({progress_pct}%)
            </div>
        </div>
        '''

        rows += f"""
        <tr>
            <td>{thumb_html}</td>
            <td>{cascade_html}</td>
            <td>{progress_html}</td>
        </tr>
        """

    return f"""
    <table class="batch-table">
        <thead>
            <tr>
                <th style="width: 100px;">缩略图</th>
                <th style="width: 120px;">基点位置</th>
                <th>损伤程度</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def _export_to_excel(results: list[dict]) -> str:
    """将批量处理结果导出为 Excel 文件"""
    if not results:
        raise gr.Error("没有可导出的结果")

    data = []
    for i, result in enumerate(results):
        data.append({
            "序号": i + 1,
            "图片路径": result["image_path"],
            "图片文件名": Path(result["image_path"]).name,
            "级联位置": result["cascade_level"],
            "级联位置置信度": result["cascade_confidence"],
            "损伤程度": result["severity_bin"],
            "损伤程度置信度": result["severity_confidence"],
            "生命周期进度": f"{_get_progress_percent(result['severity_bin'])}%",
        })

    df = pd.DataFrame(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"损伤识别结果_{timestamp}.xlsx"
    output_path = Path("output") / filename
    output_path.parent.mkdir(exist_ok=True)

    df.to_excel(output_path, index=False, engine="openpyxl")

    return str(output_path)


def build_demo(pipeline: DamagePredictionPipeline) -> gr.Blocks:
    default_paths = {
        "cascade_level": pipeline.cascade_model_path,
        "severity_stage1": pipeline.stage1_model_path,
        "severity_stage2": pipeline.stage2_model_path,
    }
    pipeline_cache: dict[tuple[str, str, str, str], DamagePredictionPipeline] = {}
    training_manager = TrainingManager()

    def get_pipeline(
        cascade_model_path: str | None,
        stage1_model_path: str | None,
        stage2_model_path: str | None,
    ) -> DamagePredictionPipeline:
        cascade_path = _select_model_path(cascade_model_path, default_paths["cascade_level"])
        stage1_path = _select_model_path(stage1_model_path, default_paths["severity_stage1"])
        stage2_path = _select_model_path(stage2_model_path, default_paths["severity_stage2"])
        cache_key = (
            str(cascade_path.resolve()),
            str(stage1_path.resolve()),
            str(stage2_path.resolve()),
            pipeline.device,
        )
        if cache_key not in pipeline_cache:
            pipeline_cache[cache_key] = DamagePredictionPipeline(
                cascade_model_path=cascade_path,
                stage1_model_path=stage1_path,
                stage2_model_path=stage2_path,
                crop_size=pipeline.crop_size,
                image_size=pipeline.image_size,
                device=pipeline.device,
            )
        return pipeline_cache[cache_key]

    def predict(
        image_path: str,
        cascade_model_path: str | None,
        stage1_model_path: str | None,
        stage2_model_path: str | None,
    ):
        if not image_path:
            raise gr.Error("请先上传一张损伤光斑图")
        active_pipeline = get_pipeline(cascade_model_path, stage1_model_path, stage2_model_path)
        prediction = active_pipeline.predict(Path(image_path))
        return (
            prediction.cascade_level,
            f"{prediction.cascade_confidence:.4f}",
            prediction.severity_bin,
            f"{prediction.severity_confidence:.4f}",
            prediction.severity_range_text,
            _build_severity_visual(
                prediction.severity_bin,
                prediction.cascade_level,
                prediction.severity_confidence,
            ),
        )

    with gr.Blocks(title="模拟损伤光斑图分级识别") as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                """
                <section class="hero-band">
                    <h1>模拟损伤光斑图分级识别</h1>
                    <p>上传一张损伤光斑图后，系统会给出级联位置、S1-S4 损伤阶段，并生成一张序列位置图辅助判断。</p>
                    <div class="workflow-strip">
                        <span>1. 上传损伤光斑图</span>
                        <span>2. 识别第一级 / 第二级</span>
                        <span>3. 查看 S1-S4 阶段位置</span>
                    </div>
                </section>
                """
            )
            gr.Markdown("")
            with gr.Accordion("模型选择（不选择则使用默认最新模型）", open=False):
                with gr.Row():
                    cascade_model = gr.File(label="级联位置模型 best.pt", file_types=[".pt"], type="filepath")
                    stage1_model = gr.File(label="第一级程度模型 best.pt", file_types=[".pt"], type="filepath")
                    stage2_model = gr.File(label="第二级程度模型 best.pt", file_types=[".pt"], type="filepath")
            with gr.Row(elem_classes=["workspace-row"]):
                with gr.Column(scale=5, elem_classes=["input-panel"]):
                    gr.HTML(
                        """
                        <div class="panel-heading">
                            <h2>图像输入</h2>
                            <p>建议上传单张损伤光斑图。画面越接近训练数据中的光斑中心区域，识别结果越容易稳定。</p>
                        </div>
                        """
                    )
                    image_input = gr.Image(type="filepath", label="上传损伤光斑图", elem_id="image-input", height=390)
                    submit = gr.Button("开始识别", variant="primary", elem_id="submit-button")
                with gr.Column(scale=7, elem_classes=["result-panel"]):
                    gr.HTML(
                        """
                        <div class="panel-heading">
                            <h2>识别结果</h2>
                            <p>右侧阶段图把 S1-S4 放回 001-121 的模拟序列中，方便快速判断损伤进展位置。</p>
                        </div>
                        """
                    )
                    stage_visual = gr.Image(label="损伤阶段位置图", elem_id="stage-visual", type="pil", height=330)
                    with gr.Row():
                        cascade_level = gr.Textbox(label="级联位置", elem_classes=["compact-output"])
                        cascade_conf = gr.Textbox(label="级联位置置信度", elem_classes=["compact-output"])
                    with gr.Row():
                        severity_bin = gr.Textbox(label="损伤程度分档", elem_classes=["compact-output"])
                        severity_conf = gr.Textbox(label="损伤程度置信度", elem_classes=["compact-output"])
                    severity_text = gr.Textbox(label="程度说明", lines=2, elem_classes=["compact-output"])
            submit.click(
                fn=predict,
                inputs=[image_input, cascade_model, stage1_model, stage2_model],
                outputs=[cascade_level, cascade_conf, severity_bin, severity_conf, severity_text, stage_visual],
            )

                with gr.Tab("批量识别"):
                    with gr.Column(elem_classes=["batch-section"]):
                        gr.HTML(
                            """
                            <div class="panel-heading">
                                <h2>批量图片处理</h2>
                                <p>一次上传多张损伤光斑图，系统将自动批量识别并以表格形式展示结果，支持导出为 Excel 文件。</p>
                            </div>
                            """
                        )
                        batch_files = gr.File(
                            label="上传多张图片",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        with gr.Row():
                            batch_submit = gr.Button("批量识别", variant="primary", elem_id="submit-button")
                            export_button = gr.Button("导出 Excel", variant="secondary")
                        download_file = gr.File(label="下载 Excel 文件", visible=False)
                        batch_results_html = gr.HTML(
                            value='<p style="color: var(--jg-muted); text-align: center;">暂无结果，请上传图片并点击"批量识别"</p>'
                        )
                    batch_submit.click(
                        fn=batch_predict,
                        inputs=[batch_files, cascade_model, stage1_model, stage2_model],
                        outputs=[batch_results_html],
                    )
                    export_button.click(
                        fn=export_batch_results,
                        inputs=[],
                        outputs=[download_file],
                    )

    return demo


def parse_args() -> argparse.Namespace:
    defaults = default_model_paths()
    parser = argparse.ArgumentParser(description="启动模拟损伤光斑图分级识别 Gradio 页面")
    parser.add_argument("--cascade-model", default=str(defaults["cascade_level"]), help="级联位置模型路径")
    parser.add_argument("--stage1-model", default=str(defaults["severity_stage1"]), help="第一级程度模型路径")
    parser.add_argument("--stage2-model", default=str(defaults["severity_stage2"]), help="第二级程度模型路径")
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu 或 0")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DamagePredictionPipeline(
        cascade_model_path=args.cascade_model,
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        device=args.device,
    )
    demo = build_demo(pipeline)
    demo.launch(server_name=args.host, server_port=args.port, css=APP_CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

