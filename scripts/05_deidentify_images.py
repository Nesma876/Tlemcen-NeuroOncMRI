"""Mask likely overlay text in image copies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.image_deidentification import process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Input PNG image root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for cleaned image copies.")
    parser.add_argument("--log-csv", type=Path, required=True, help="CSV cleaning log path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.input_root, args.output_root, args.log_csv)


if __name__ == "__main__":
    main()
