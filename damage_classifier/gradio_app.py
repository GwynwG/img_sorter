from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from .config import default_model_paths
from .inference import DamagePredictionPipeline


def build_demo(pipeline: DamagePredictionPipeline) -> gr.Blocks:
    def predict(image_path: str):
        if not image_path:
            raise gr.Error("请先上传一张损伤光斑图")
        prediction = pipeline.predict(Path(image_path))
        return (
            prediction.cascade_level,
            f"{prediction.cascade_confidence:.4f}",
            prediction.severity_bin,
            f"{prediction.severity_confidence:.4f}",
            prediction.severity_range_text,
        )

    with gr.Blocks(title="模拟损伤光斑图分级识别") as demo:
        gr.Markdown(
            """
            # 模拟损伤光斑图分级识别
            上传一张损伤光斑图，系统会先判断损伤位于第一级还是第二级，再给出 4 档粗略损伤程度。
            """
        )
        with gr.Row():
            image_input = gr.Image(type="filepath", label="上传损伤光斑图")
            with gr.Column():
                cascade_level = gr.Textbox(label="级联位置")
                cascade_conf = gr.Textbox(label="级联位置置信度")
                severity_bin = gr.Textbox(label="损伤程度分档")
                severity_conf = gr.Textbox(label="损伤程度置信度")
                severity_text = gr.Textbox(label="程度说明", lines=2)
        submit = gr.Button("开始识别", variant="primary")
        submit.click(
            fn=predict,
            inputs=[image_input],
            outputs=[cascade_level, cascade_conf, severity_bin, severity_conf, severity_text],
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
    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
