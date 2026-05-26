from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from damage_classifier.training_manager import TrainingManager, TrainingRequest, build_training_command


class FakeRunningProcess:
    returncode = None

    def poll(self):
        return None


class TrainingManagerTests(unittest.TestCase):
    def test_build_training_command_for_all_models(self):
        request = TrainingRequest(
            model_target="all",
            epochs=12,
            imgsz=320,
            batch=8,
            device="cpu",
            workers=0,
            model_source="D:/models/yolo11n-cls.pt",
        )

        command = build_training_command(
            request,
            python_executable="python",
            script_path=Path("D:/project/img_sorter/train_damage_models.py"),
        )

        self.assertEqual(command[0], "python")
        self.assertEqual(command[1], "D:\\project\\img_sorter\\train_damage_models.py")
        self.assertIn("--models", command)
        self.assertIn("all", command)
        self.assertIn("--epochs", command)
        self.assertIn("12", command)
        self.assertIn("--imgsz", command)
        self.assertIn("320", command)
        self.assertIn("--batch", command)
        self.assertIn("8", command)
        self.assertIn("--device", command)
        self.assertIn("cpu", command)
        self.assertIn("--workers", command)
        self.assertIn("0", command)
        self.assertIn("--model-source", command)
        self.assertIn("D:/models/yolo11n-cls.pt", command)

    def test_build_training_command_accepts_chinese_target_label(self):
        request = TrainingRequest(
            model_target="第一级损伤程度分类器",
            epochs=5,
            imgsz=224,
            batch=4,
            device="0",
            workers=2,
        )

        command = build_training_command(request, python_executable="python", script_path=Path("train_damage_models.py"))

        model_index = command.index("--models") + 1
        self.assertEqual(command[model_index], "severity_stage1")

    def test_training_manager_rejects_concurrent_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingManager(log_dir=Path(tmp), validate_datasets=False)
            request = TrainingRequest(
                model_target="cascade_level",
                epochs=1,
                imgsz=320,
                batch=1,
                device="cpu",
                workers=0,
            )

            with patch("damage_classifier.training_manager.subprocess.Popen", return_value=FakeRunningProcess()):
                first = manager.start_training(request)
                self.assertEqual(first.state, "running")

                with self.assertRaisesRegex(RuntimeError, "已有训练任务正在运行"):
                    manager.start_training(request)


if __name__ == "__main__":
    unittest.main()
