"""Validate the expected external dataset layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlemcen_neurooncmri.config import load_config
from tlemcen_neurooncmri.data import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json", help="JSON config path.")
    parser.add_argument("--metadata-file", help="Override metadata .xlsx/.csv file.")
    parser.add_argument("--image-dir", help="Override image directory containing CP/ and CS/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    metadata_file = Path(args.metadata_file) if args.metadata_file else cfg.metadata_file
    image_dir = Path(args.image_dir) if args.image_dir else cfg.image_dir

    try:
        report = validate_dataset(metadata_file, image_dir)
    except Exception as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        print("Download the dataset from https://doi.org/10.17632/9ns6748zkc.1 and see DATA.md.", file=sys.stderr)
        return 2

    print(f"Metadata rows: {report.n_rows}")
    print(f"CP image directory present: {report.cp_image_dir}")
    print(f"CS image directory present: {report.cs_image_dir}")
    if report.missing_columns:
        print(f"Missing metadata columns: {', '.join(report.missing_columns)}", file=sys.stderr)
        return 1
    print("Dataset layout looks usable.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
