from pathlib import Path

from tlemcen_neurooncmri.config import load_config
from tlemcen_neurooncmri.data import validate_dataset


def test_load_default_config():
    cfg = load_config("configs/default.json")
    assert cfg.seed == 42
    assert cfg.max_images_per_patient == 30
    assert cfg.metadata_file.name == "ClasseurPFE1 (1).xlsx"


def test_validate_synthetic_example():
    report = validate_dataset(
        Path("data/example/metadata_example.csv"),
        Path("data/example/images"),
    )
    assert report.ok
    assert report.n_rows == 2
    assert report.missing_columns == ()
