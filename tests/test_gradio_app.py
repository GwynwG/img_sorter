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
    _build_structure_html,
    _build_training_status_html,
    _severity_sequence_position,
)
from damage_classifier.inference import DamagePredictionPipeline
from damage_classifier.training_artifacts import ModelArtifactSummary
from damage_classifier.training_manager import TrainingSnapshot


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

    def test_restored_build_keeps_feature_tabs_visible(self):
        self.assertNotIn("/* temporary screenshot build: hide feature tabs */", APP_CSS)
        self.assertNotIn('#app-shell [role="tablist"]', APP_CSS)

    def test_delivery_css_hides_gradio_developer_footer_actions(self):
        self.assertIn("footer .show-api", APP_CSS)
        self.assertIn("footer .settings", APP_CSS)
        self.assertIn("display: none !important;", APP_CSS)

    def test_training_status_html_uses_lightweight_status_card(self):
        snapshot = TrainingSnapshot(
            state="idle",
            model_target=None,
            command=[],
            log_path=None,
            returncode=None,
            message="暂无训练任务",
            started_at=None,
            finished_at=None,
        )

        html = _build_training_status_html(snapshot)

        self.assertIn('class="training-status-card"', html)
        self.assertNotIn('class="workbench-panel"', html)

    def test_structure_html_does_not_render_nested_workbench_panel(self):
        html = _build_structure_html(
            {
                "cascade_level": Path("missing-cascade.pt"),
                "severity_stage1": Path("missing-stage1.pt"),
                "severity_stage2": Path("missing-stage2.pt"),
            }
        )

        self.assertIn('class="structure-summary"', html)
        self.assertNotIn('class="workbench-panel"', html)

    def test_artifact_summary_uses_spacious_cards_for_long_paths(self):
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

        self.assertIn('class="artifact-summary-list"', html)
        self.assertIn('class="artifact-summary-card"', html)
        self.assertIn('class="artifact-summary-metrics"', html)
        self.assertIn('class="artifact-path-grid"', html)
        self.assertIn('class="status-pill"', html)
        self.assertIn('class="path-cell"', html)
        self.assertNotIn("<table", html)
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

        self.assertIn('class="artifact-model-stack"', html)
        self.assertIn('class="artifact-model-block"', html)
        self.assertIn('class="artifact-curve-row"', html)
        self.assertIn('class="artifact-confusion-row"', html)
        self.assertIn('class="artifact-image-card"', html)
        self.assertLess(html.index('class="artifact-curve-row"'), html.index('class="artifact-confusion-row"'))
        self.assertNotIn('class="artifact-image-grid"', html)
        self.assertNotIn("overflow-y", html)
        self.assertNotIn("height: 460", html)


if __name__ == "__main__":
    unittest.main()
