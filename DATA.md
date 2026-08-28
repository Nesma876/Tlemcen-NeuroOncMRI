# Data Access And Layout

## Source

Dataset: Tlemcen-NeuroOncMRI.

Official dataset DOI: https://doi.org/10.17632/9ns6748zkc.1.

Associated manuscript: *An Open North-African MRI-Report Dataset and Leakage-Audited Benchmark for Differentiating Primary Brain Tumors from Cerebral Metastases*.

Article DOI: pending.

## Redistribution

The full dataset is intentionally absent from this repository. Do not commit raw images, the full metadata workbook, extracted archives, checkpoints, or generated patient-level outputs containing sensitive information.

## Expected Local Layout

The default config expects:

```text
data/raw/
├── ClasseurPFE1 (1).xlsx
└── images_v2_final/
    └── data p_s/
        ├── CP/
        └── CS/
```

You can override this layout with:

```bash
python scripts/01_validate_dataset.py --metadata-file /path/to/ClasseurPFE1.xlsx --image-dir "/path/to/data p_s"
```

## Required Metadata Columns

The code paths inspected in this repository require the following columns:

```text
ID_Patient
Origine_Tumeur
Sexe
Localisation
ANNOTATION
Rapport
```

Some analyses also reference:

```text
Age or Âge
Métastase_Origine
Type_Général
```

## Image Conventions

The image scripts expect class folders named `CP` and `CS`. Patient image lookup uses filenames beginning with patient identifiers such as `P01 (` and extensions `.jpg` or `.png`.

## Preprocessing

`preprocessing/image_deidentification.py` masks likely text overlays in copies of PNG images. It never needs to overwrite the source dataset. The script writes cleaned copies and a CSV log for manual review of skipped low-confidence images.

## Derived Data

Generated outputs should be written under `results/generated/` and are ignored by Git. Historical reference values are documented in `README.md` and `results/reference/README.md`; reference result files should only be added if they are already approved for redistribution.

## Example Data

`data/example/metadata_example.csv` is synthetic and exists only for smoke tests. It is not scientific data and must not be used to report results.
