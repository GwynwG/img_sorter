<<<<<<< HEAD
﻿from __future__ import annotations
=======
from __future__ import annotations
>>>>>>> 6ed2eb1deaef4a6128758f9b78052bf38af78ac2

import shutil
import unittest
from pathlib import Path

from PIL import Image

from damage_classifier.config import PROJECT_ROOT, severity_bin_for_index, severity_range_text, split_for_index
from damage_classifier.preprocessing import parse_image_index, preprocess_image


class DamageClassifierTests(unittest.TestCase):
    def test_severity_bin_mapping(self):
        self.assertEqual(severity_bin_for_index(1), "S1")
        self.assertEqual(severity_bin_for_index(30), "S1")
        self.assertEqual(severity_bin_for_index(31), "S2")
        self.assertEqual(severity_bin_for_index(61), "S3")
        self.assertEqual(severity_bin_for_index(121), "S4")

    def test_split_mapping(self):
        self.assertEqual(split_for_index(1), "train")
        self.assertEqual(split_for_index(25), "val")
        self.assertEqual(split_for_index(55), "val")
        self.assertEqual(split_for_index(114), "train")
        self.assertEqual(split_for_index(121), "val")

    def test_severity_range_text(self):
        self.assertIn("001-030", severity_range_text("S1"))
        self.assertIn("091-121", severity_range_text("S4"))

    def test_parse_image_index(self):
        self.assertEqual(parse_image_index(Path("img_001.jpg")), 1)
        self.assertEqual(parse_image_index(Path("img_121.jpg")), 121)

    def test_preprocess_image_output_shape(self):
        tmpdir = PROJECT_ROOT / "tests" / "_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / "img_001.jpg"
        try:
            Image.new("L", (2000, 2000), color=128).save(path)
            processed = preprocess_image(path, crop_size=1200, output_size=320)
            self.assertEqual(processed.mode, "RGB")
            self.assertEqual(processed.size, (320, 320))
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
