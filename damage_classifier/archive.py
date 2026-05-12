from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ARTIFACTS_DIR

ARCHIVES_DIR = ARTIFACTS_DIR / "archives"


def _date_suffix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _next_archive_path(parent: Path, base_name: str, extension: str = "") -> Path:
    date_suffix = _date_suffix()
    candidate = parent / f"{base_name}_{date_suffix}{extension}"
    if not candidate.exists():
        return candidate

    version = 2
    while True:
        candidate = parent / f"{base_name}_{date_suffix}_v{version}{extension}"
        if not candidate.exists():
            return candidate
        version += 1


def archive_directory(source_dir: Path, archive_group: str, archive_name: str) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Archive source directory not found: {source_dir}")
    target_parent = ARCHIVES_DIR / archive_group
    target_parent.mkdir(parents=True, exist_ok=True)
    target_dir = _next_archive_path(target_parent, archive_name)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def archive_json(data: dict[str, Any], archive_group: str, archive_name: str) -> Path:
    target_parent = ARCHIVES_DIR / archive_group
    target_parent.mkdir(parents=True, exist_ok=True)
    target_file = _next_archive_path(target_parent, archive_name, extension=".json")
    target_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_file
