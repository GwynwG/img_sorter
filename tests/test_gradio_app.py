from __future__ import annotations

import unittest
from pathlib import Path

import gradio as gr
from PIL import Image, ImageChops

from damage_classifier.gradio_app import (
    APP_CSS,
    _build_artifact_images_html,
    _build_artifact_summary_html,
    _build_severity_visual,
    _severity_sequence_position,
)
from damage_classifier.inference import DamagePredictionPipeline
from damage_classifier.training_artifacts import ModelArtifactSummary


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

    def test_temporary_screenshot_build_hides_feature_tabs(self):
        self.assertIn("/* temporary screenshot build: hide feature tabs */", APP_CSS)
        self.assertIn('#app-shell [role="tablist"]', APP_CSS)
        self.assertIn("display: none !important;", APP_CSS)

    def test_artifact_table_wraps_long_paths(self):
        summary = ModelArtifactSummary(
            model_name="cascade_level",
            display_name="级联位置分类器",
            run_dir=Path("D:/project/img_sorter/damage_artifacts/runs/cascade_level"),
            model_dir=Path("D:/project/img_sorter/damage_artifacts/models/cascade_level"),
            args={},
            epoch_count=195,
            top1_accuracy=1.0,
            val_loss=0.1112,
            best_path=Path("D:/project/img_sorter/damage_artifacts/models/cascade_level/best.pt"),
            last_path=None,
            results_csv=Path("D:/project/img_sorter/damage_artifacts/runs/cascade_level/results.csv"),
            results_png=None,
            confusion_matrix=None,
            confusion_matrix_normalized=None,
            train_summary_path=None,
            status="available",
        )

        html = _build_artifact_summary_html([summary])

        self.assertIn('class="artifact-table"', html)
        self.assertIn('class="path-cell"', html)
        self.assertIn(str(Path("D:/project/img_sorter/damage_artifacts/models/cascade_level/best.pt")), html)

    def test_artifact_images_render_as_page_flow_html(self):
        summary = ModelArtifactSummary(
            model_name="cascade_level",
            display_name="级联位置分类器",
            run_dir=Path("run"),
            model_dir=Path("model"),
            args={},
            epoch_count=1,
            top1_accuracy=1.0,
            val_loss=0.1,
            best_path=None,
            last_path=None,
            results_csv=None,
            results_png=Path("results.png"),
            confusion_matrix=Path("confusion_matrix.png"),
            confusion_matrix_normalized=Path("confusion_matrix_normalized.png"),
            train_summary_path=None,
            status="available",
        )

        html = _build_artifact_images_html([summary])

        self.assertIn('class="artifact-image-grid"', html)
        self.assertIn('class="artifact-image-card"', html)
        self.assertNotIn("overflow-y", html)
        self.assertNotIn("height: 460", html)


if __name__ == "__main__":
    unittest.main()
