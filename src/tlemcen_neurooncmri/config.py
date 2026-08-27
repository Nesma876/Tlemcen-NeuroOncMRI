"""Configuration loading for command-line wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    metadata_file: Path
    image_dir: Path
    output_dir: Path
    seed: int = 42
    max_images_per_patient: int = 30


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path = "configs/default.json") -> ProjectConfig:
    config_path = Path(path)
    base = config_path.parent if config_path.is_absolute() else Path.cwd()
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return ProjectConfig(
        metadata_file=_resolve(base, raw["metadata_file"]),
        image_dir=_resolve(base, raw["image_dir"]),
        output_dir=_resolve(base, raw.get("output_dir", "results/generated")),
        seed=int(raw.get("seed", 42)),
        max_images_per_patient=int(raw.get("max_images_per_patient", 30)),
    )
