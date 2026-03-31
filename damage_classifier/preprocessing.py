from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


IMAGE_INDEX_RE = re.compile(r"img_(\d+)$", re.IGNORECASE)


def parse_image_index(image_path: Path) -> int:
    match = IMAGE_INDEX_RE.fullmatch(image_path.stem)
    if not match:
        raise ValueError(f"Unsupported image filename: {image_path.name}")
    return int(match.group(1))


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    size = min(crop_size, width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image_path: Path, crop_size: int, output_size: int) -> Image.Image:
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
        cropped = center_crop(grayscale, crop_size)
        resized = cropped.resize((output_size, output_size), Image.Resampling.BICUBIC)
        return resized.convert("RGB")


def save_preprocessed_image(image_path: Path, output_path: Path, crop_size: int, output_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = preprocess_image(image_path, crop_size=crop_size, output_size=output_size)
    processed.save(output_path, format="JPEG", quality=95)
