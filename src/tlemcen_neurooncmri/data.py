"""Dataset structure checks for the external Tlemcen-NeuroOncMRI data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "ID_Patient",
    "Origine_Tumeur",
    "Sexe",
    "Localisation",
    "ANNOTATION",
    "Rapport",
}


@dataclass(frozen=True)
class DatasetReport:
    metadata_file: Path
    image_dir: Path
    n_rows: int
    missing_columns: tuple[str, ...]
    cp_image_dir: bool
    cs_image_dir: bool

    @property
    def ok(self) -> bool:
        return not self.missing_columns and self.cp_image_dir and self.cs_image_dir


def read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Feuil1")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path.suffix}")


def validate_dataset(metadata_file: str | Path, image_dir: str | Path) -> DatasetReport:
    metadata_path = Path(metadata_file)
    image_path = Path(image_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_path}")

    df = read_metadata(metadata_path)
    missing = tuple(sorted(REQUIRED_METADATA_COLUMNS.difference(df.columns)))
    return DatasetReport(
        metadata_file=metadata_path,
        image_dir=image_path,
        n_rows=len(df),
        missing_columns=missing,
        cp_image_dir=(image_path / "CP").is_dir(),
        cs_image_dir=(image_path / "CS").is_dir(),
    )
