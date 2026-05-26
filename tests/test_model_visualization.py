from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from damage_classifier.model_visualization import build_pipeline_graph, summarize_yolo_model


class ModelVisualizationTests(unittest.TestCase):
    def test_build_pipeline_graph_contains_two_stage_flow(self):
        graph = build_pipeline_graph()

        labels = {node.label for node in graph.nodes}
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn("输入损伤光斑图", labels)
        self.assertIn("级联位置分类器", labels)
        self.assertIn("第一级损伤程度分类器", labels)
        self.assertIn("第二级损伤程度分类器", labels)
        self.assertIn("S1 / S2 / S3 / S4", labels)
        self.assertIn(("input", "cascade_level"), edges)
        self.assertIn(("cascade_level", "severity_stage1"), edges)
        self.assertIn(("cascade_level", "severity_stage2"), edges)
        self.assertIn(("severity_stage1", "severity_output"), edges)
        self.assertIn(("severity_stage2", "severity_output"), edges)

    def test_summarize_yolo_model_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.pt"

            summary = summarize_yolo_model(
                "cascade_level",
                missing_path,
                display_name="级联位置分类器",
                class_labels=("第一级", "第二级"),
                image_size=320,
            )

            self.assertFalse(summary.available)
            self.assertEqual(summary.model_name, "cascade_level")
            self.assertEqual(summary.display_name, "级联位置分类器")
            self.assertEqual(summary.model_path, missing_path)
            self.assertEqual(summary.class_labels, ("第一级", "第二级"))
            self.assertEqual(summary.image_size, 320)
            self.assertIn("未找到模型文件", summary.message)
            self.assertIsNone(summary.layer_count)
            self.assertIsNone(summary.parameter_count)


if __name__ == "__main__":
    unittest.main()
