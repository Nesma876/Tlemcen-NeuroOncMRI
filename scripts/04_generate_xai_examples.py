"""Generate historical word and image occlusion example outputs."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlemcen_neurooncmri.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json", help="JSON config path.")
    parser.add_argument("--metadata-file", help="Override metadata Excel file.")
    parser.add_argument("--image-dir", help="Override image directory containing CP/ and CS/.")
    parser.add_argument("--output-dir", help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    os.environ["TNO_METADATA_FILE"] = str(Path(args.metadata_file) if args.metadata_file else cfg.metadata_file)
    os.environ["TNO_IMAGE_DIR"] = str(Path(args.image_dir) if args.image_dir else cfg.image_dir)
    os.environ["TNO_OUTPUT_DIR"] = str(Path(args.output_dir) if args.output_dir else cfg.output_dir)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "xai" / "word_and_image_occlusion.py"), run_name="__main__")


if __name__ == "__main__":
    main()
