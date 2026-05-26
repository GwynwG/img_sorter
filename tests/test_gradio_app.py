from __future__ import annotations

import unittest

import gradio as gr
from PIL import Image, ImageChops

from damage_classifier.gradio_app import _build_severity_visual, _severity_sequence_position
from damage_classifier.inference import DamagePredictionPipeline


class GradioAppVisualizationTests(unittest.TestCase):
    def test_severity_sequence_position_uses_configured_range(self):
        position = _severity_sequence_position("S3")

        self.assertEqual(position["stage_number"], 3)
        self.assertEqual(position["start"], 61)
        self.assertEqual(position["end"], 90)
        self.assertEqual(position["total"], 121)
        self.assertAlmostEqual(position["midpoint_percent"], 62.08, places=2)

    def test_build_severity_visual_returns_non_blank_image(self):
        image = _build_severity_visual("S4", "第二级", 0.9234)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (960, 420))
        blank = Image.new("RGB", image.size, image.getpixel((0, 0)))
        self.assertIsNotNone(ImageChops.difference(image, blank).getbbox())

    def test_build_demo_returns_blocks_without_loading_weights(self):
        pipeline = DamagePredictionPipeline(
            cascade_model_path="missing-cascade.pt",
            stage1_model_path="missing-stage1.pt",
            stage2_model_path="missing-stage2.pt",
            device="cpu",
        )

        demo = __import__("damage_classifier.gradio_app", fromlist=["build_demo"]).build_demo(pipeline)

        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
