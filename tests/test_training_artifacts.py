from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from damage_classifier.training_artifacts import summarize_model_artifacts


class TrainingArtifactTests(unittest.TestCase):
    def test_summarize_model_artifacts_reads_metrics_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "cascade_level"
            model_dir = root / "models" / "cascade_level"
            run_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)

            (run_dir / "results.csv").write_text(
                "\n".join(
                    [
                        "epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss,lr/pg0",
                        "1,1.0,0.8,0.50,1.0,0.72,0.001",
                        "2,2.0,0.4,0.85,1.0,0.31,0.0005",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "args.yaml").write_text("epochs: 2\nbatch: 16\ndevice: '0'\n", encoding="utf-8")
            (run_dir / "results.png").write_bytes(b"png")
            (run_dir / "confusion_matrix.png").write_bytes(b"png")
            (run_dir / "confusion_matrix_normalized.png").write_bytes(b"png")
            (model_dir / "best.pt").write_bytes(b"best")
            (model_dir / "last.pt").write_bytes(b"last")
            (model_dir / "train_summary.json").write_text('{"run_dir": "example"}', encoding="utf-8")

            summary = summarize_model_artifacts(
                "cascade_level",
                display_name="级联位置分类器",
                run_dir=run_dir,
                model_dir=model_dir,
            )

            self.assertEqual(summary.model_name, "cascade_level")
            self.assertEqual(summary.display_name, "级联位置分类器")
            self.assertEqual(summary.status, "available")
            self.assertEqual(summary.epoch_count, 2)
            self.assertEqual(summary.top1_accuracy, 0.85)
            self.assertEqual(summary.val_loss, 0.31)
            self.assertEqual(summary.args["epochs"], "2")
            self.assertEqual(summary.args["device"], "0")
            self.assertEqual(summary.best_path, model_dir / "best.pt")
            self.assertEqual(summary.results_csv, run_dir / "results.csv")
            self.assertEqual(summary.results_png, run_dir / "results.png")
            self.assertEqual(summary.confusion_matrix, run_dir / "confusion_matrix.png")
            self.assertEqual(summary.confusion_matrix_normalized, run_dir / "confusion_matrix_normalized.png")
            self.assertEqual(summary.train_summary_path, model_dir / "train_summary.json")

    def test_summarize_model_artifacts_handles_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = summarize_model_artifacts(
                "severity_stage1",
                display_name="第一级损伤程度分类器",
                run_dir=root / "runs" / "severity_stage1",
                model_dir=root / "models" / "severity_stage1",
            )

            self.assertEqual(summary.status, "missing")
            self.assertEqual(summary.epoch_count, 0)
            self.assertIsNone(summary.top1_accuracy)
            self.assertIsNone(summary.val_loss)
            self.assertIsNone(summary.best_path)
            self.assertIsNone(summary.results_csv)
            self.assertEqual(summary.args, {})


if __name__ == "__main__":
    unittest.main()
