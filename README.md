# Tlemcen-NeuroOncMRI companion code

Companion repository for the manuscript *An Open North-African MRI-Report Dataset and Leakage-Audited Benchmark for Differentiating Primary Brain Tumors from Cerebral Metastases*.

This repository contains the code used to document and reproduce the leakage-audited benchmark analyses associated with the externally published Tlemcen-NeuroOncMRI dataset.

## Overview

This code connects the published dataset, the original executed Kaggle notebook, and extracted standalone scripts. It is intended for scientific traceability: readers can identify which entry point reproduces each documented analysis, which protocol was used, and which analyses are historical or exploratory.

Version: 1.0.0.

## Associated Dataset

Dataset name: Tlemcen-NeuroOncMRI.

Official dataset DOI: https://doi.org/10.17632/9ns6748zkc.1.

The dataset is not distributed through this repository. Download it from the official source and follow the terms on the dataset page. The dataset license is documented in the existing project license note as CC BY 4.0.

## Associated Publication

Manuscript: *An Open North-African MRI-Report Dataset and Leakage-Audited Benchmark for Differentiating Primary Brain Tumors from Cerebral Metastases*.

Article DOI: TODO_ARTICLE_DOI.

## Data Access

Download the dataset from the official Mendeley Data record: https://doi.org/10.17632/9ns6748zkc.1.

Do not commit the downloaded dataset, medical images, full metadata workbook, checkpoints, or generated patient-level outputs to this repository.

## Expected Dataset Structure

Expected local placement for the default configuration:

```text
data/raw/
├── ClasseurPFE1 (1).xlsx
└── images_v2_final/
    └── data p_s/
        ├── CP/
        └── CS/
```

See `DATA.md` for required columns, file conventions, and what is intentionally absent from Git.

## Repository Scope

The repository preserves the original Kaggle notebook and extracted scripts used for the manuscript. It also adds lightweight command-line wrappers, configuration, dataset validation, metadata, and tests so an external researcher can understand the path:

```text
published dataset -> validation -> preprocessing -> text/image analysis -> fusion/evaluation -> XAI outputs
```

No scientific results are recomputed during installation or testing, and no dataset files are included.

## Repository Structure

```text
.
├── configs/                 Default portable configuration
├── data/                    Dataset instructions and synthetic smoke-test example
├── docs/                    Audit notes for public release preparation
├── evaluation/              DeLong and XAI helper functions from the original work
├── figures/                 Placeholder for generated manuscript figures
├── fusion/                  Historical fusion scripts and strategies
├── gat/                     Exploratory patient-similarity GAT implementation
├── models/text/             Corrected masking text benchmark script
├── notebooks/               Original full executed notebook, outputs stripped for Git
├── preprocessing/           Image de-identification helper
├── results/                 Reference-output documentation, generated outputs ignored
├── scripts/                 Numbered command-line workflow wrappers
├── src/tlemcen_neurooncmri/ Lightweight config, data validation, and seed utilities
└── tests/                   Smoke tests that run without the full dataset
```

## Installation

Python 3.10 or newer is recommended.

Windows PowerShell:

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Validate the dataset:

```bash
python scripts/01_validate_dataset.py --metadata-file "data/raw/ClasseurPFE1 (1).xlsx" --image-dir "data/raw/images_v2_final/data p_s"
```

Run the corrected masking text benchmark:

```bash
python scripts/02_run_text_benchmark.py --config configs/default.json
```

Run the historical fusion reference analysis:

```bash
python scripts/03_run_fusion_reference.py --config configs/default.json
```

Generate the historical XAI examples:

```bash
python scripts/04_generate_xai_examples.py --config configs/default.json
```

Run image de-identification on copies of images:

```bash
python scripts/05_deidentify_images.py --input-root data/raw/images --output-root data/processed/images_cleaned --log-csv results/generated/cleaning_log.csv
```

## Reproducing The Experiments

Only outputs directly identifiable from the repository are listed here. Scripts that require the complete external dataset will fail early with an explanatory message if the dataset is absent.

| Analysis / paper output | Script | Output |
| --- | --- | --- |
| Corrected masking TF-IDF/CamemBERT benchmark | `scripts/02_run_text_benchmark.py` | `results/generated/multiseed_corrected_masking.csv` |
| Historical CamemBERT + BiomedCLIP guided fusion and DeLong test | `scripts/03_run_fusion_reference.py` | `results/generated/final_probabilities_for_plots.csv` |
| Five fusion strategies table | `fusion/all_5_fusion_strategies.py` | `fusion_5strategies_complet.csv` when run in original notebook context |
| Patient-similarity GAT exploratory analysis | `gat/patient_similarity_gat.py` | Console summary; image OOF CSV required for combined branch |
| Word/image occlusion examples | `scripts/04_generate_xai_examples.py` | `results/generated/word_occlusion_example.csv`, `results/generated/image_occlusion_example.csv` |
| Image overlay de-identification | `scripts/05_deidentify_images.py` | Cleaned image copies and CSV log |

The original notebook `notebooks/full_pipeline_original.ipynb` remains the provenance source for the extracted scripts.

## Paper-To-Code Map

| Paper analysis | Entry point | Protocol | Status |
| --- | --- | --- | --- |
| Text benchmark: TF-IDF safe features | `scripts/02_run_text_benchmark.py` | corrected masking | REPRODUCIBLE |
| Text leakage ablation: TF-IDF plus risk feature | `scripts/02_run_text_benchmark.py` | corrected masking | REPRODUCIBLE |
| Text benchmark: CamemBERT | `scripts/02_run_text_benchmark.py` | corrected masking | REPRODUCIBLE |
| BiomedCLIP image branch | `scripts/03_run_fusion_reference.py` | historical fusion run | REPRODUCIBLE/HISTORICAL |
| CamemBERT plus BiomedCLIP guided fusion | `scripts/03_run_fusion_reference.py` | historical masking | REPRODUCIBLE/HISTORICAL |
| DeLong guided fusion vs BiomedCLIP | `scripts/03_run_fusion_reference.py` | historical masking | REPRODUCIBLE/HISTORICAL |
| Five fusion strategies | `fusion/all_5_fusion_strategies.py` | historical notebook context | PARTIAL |
| Patient-similarity GAT | `gat/patient_similarity_gat.py` | exploratory transductive graph analysis | EXPLORATORY/PARTIAL |
| Text occlusion XAI | `scripts/04_generate_xai_examples.py` | historical interpretability run | REPRODUCIBLE/HISTORICAL |
| Image occlusion XAI | `scripts/04_generate_xai_examples.py` | historical interpretability run | REPRODUCIBLE/HISTORICAL |
| Image overlay de-identification | `scripts/05_deidentify_images.py` | preprocessing | REPRODUCIBLE |

## Historical Vs Corrected Masking

The final text benchmark uses corrected masking in `models/text/tfidf_camembert_corrected_multiseed.py`. This correction masks plural and related class-revealing terms and is documented in the script as the definitive text benchmark protocol.

Some fusion experiments were executed before that correction, using the historical masking in `fusion/camembert_biomedclip_fusion_delong.py` and notebook-derived fusion cells. These values are preserved for traceability of the original experimental record. They should not be interpreted as fusion results recalculated under the corrected masking protocol.

The two protocols are therefore intentionally both present:

- corrected masking: final text benchmark and leakage ablation;
- historical/pre-correction masking: historical fusion reference, five-strategy fusion context, and associated DeLong/XAI outputs.

## Exploratory Analyses

The GAT code is explicitly exploratory. It implements a transductive patient-similarity graph and documents that the tested patient's features are visible when graph similarities are built, while the tested label is masked during training.

`fusion/all_5_fusion_strategies.py` is also historical/notebook-dependent. It uses variables and intermediate embeddings created in earlier notebook cells. Because the repository does not contain approved serialized intermediate outputs for all those variables, it is classified as PARTIAL rather than a standalone primary entry point.

## Reported Reference Results

These values were already documented in the original repository and are not generated during tests:

| Model or analysis | AUC / result | Source |
| --- | --- | --- |
| TF-IDF + hand-crafted, corrected masking | 0.802 +/- 0.029, 6 seeds | Notebook cell 49 |
| TF-IDF + leaking feature ablation | 0.967 +/- 0.011, 6 seeds | Notebook cell 49 |
| CamemBERT, corrected masking | 0.775 | Notebook cell 49 |
| CamemBERT, pre-correction fusion run | 0.763 | Notebook cell 46 |
| BiomedCLIP slice-level | 0.802 +/- 0.024, 6 seeds | Notebook cells 28-31 |
| Agreement-guided fusion | 0.914 reference run | Notebook cells 37, 39, 46 |
| DeLong guided fusion vs BiomedCLIP | delta=+0.107, p=0.099 | Notebook cells 43, 46 |

## Reproducibility Notes

The wrappers use portable paths through CLI arguments, `configs/default.json`, or environment variables. Seeds are set in the historical scripts and exposed in the lightweight utilities. GPU and transformer/model-hub behavior may remain partly non-deterministic depending on CUDA, PyTorch, and external model versions.

The fusion script intentionally preserves the pre-correction text masking used in the historical fusion run. The corrected masking benchmark is separate and should not be silently substituted into the fusion results.

## Citation

Please cite all applicable research objects:

1. Dataset: Tlemcen-NeuroOncMRI, DOI https://doi.org/10.17632/9ns6748zkc.1.
2. Article: *An Open North-African MRI-Report Dataset and Leakage-Audited Benchmark for Differentiating Primary Brain Tumors from Cerebral Metastases*, DOI TODO_ARTICLE_DOI.
3. Software/code: this repository, URL TODO_REPOSITORY_URL, code DOI TODO_CODE_DOI after Zenodo archival.

## License

Code license: MIT, as stated in `LICENSE`.

Dataset license: separate from the code; see the official dataset page at https://doi.org/10.17632/9ns6748zkc.1.

## Acknowledgments

The repository preserves the authorship and license information present in the original code release materials. Dataset access and reuse remain governed by the official dataset record.
