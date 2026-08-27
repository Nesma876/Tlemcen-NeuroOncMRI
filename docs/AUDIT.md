# Repository Audit Notes

Audit date: 2026-08-27.

## Initial Contents

The opened workspace contained `github_repo_final.zip`, which extracted to a Python research repository with:

- `notebooks/full_pipeline_original.ipynb`
- `models/text/tfidf_camembert_corrected_multiseed.py`
- `fusion/`
- `gat/`
- `xai/`
- `evaluation/`
- `preprocessing/`
- `README.md`
- `LICENSE`
- `requirements.txt`

No Git history was present in the opened directory.

## Data And Artifacts

No full dataset files, checkpoints, large model files, or generated result CSVs were present after extraction. The notebook contained execution outputs and model download progress widgets; outputs were stripped for public repository hygiene.

## Hard-Coded Paths Found

- Kaggle metadata path in text, fusion, and XAI scripts.
- Kaggle image path in fusion and XAI scripts.
- Kaggle working output paths for generated CSV files.
- `/home/claude/...` paths in image de-identification main block.
- Local workbook path references in exploratory `__main__` blocks for `fusion_base_strategies.py` and `gat/patient_similarity_gat.py`.

The main reproducible wrappers now provide portable path configuration. Some exploratory historical blocks remain documented as requiring author review before being treated as primary CLI entry points.

## Secrets

No API keys, access tokens, credentials, or passwords were found in the inspected files. A notebook output warning mentioned `HF_TOKEN`, but no token value was present.

## Scientific Caveats

The repository itself documents that fusion results used pre-correction masking, while the final text benchmark used corrected masking. This was preserved and surfaced in public documentation.

## Historical Vs Corrected Masking

Corrected masking is used by:

- `models/text/tfidf_camembert_corrected_multiseed.py`
- `scripts/02_run_text_benchmark.py`

Historical/pre-correction masking is used by:

- `fusion/camembert_biomedclip_fusion_delong.py`
- `scripts/03_run_fusion_reference.py`
- `xai/word_and_image_occlusion.py`
- `scripts/04_generate_xai_examples.py`
- notebook cells preceding the final corrected text benchmark

`fusion/all_5_fusion_strategies.py` belongs to the historical notebook context. It depends on variables created in earlier notebook cells (`proba_camembert`, `proba_foundation`, embeddings, labels, and helper functions). Making it standalone would require supplying serialized intermediate outputs that are not present in the repository. It is therefore documented as PARTIAL rather than rewritten.

## Paper-To-Code Classification

| Analysis | Entry point | Protocol | Classification |
| --- | --- | --- | --- |
| Corrected text benchmark | `scripts/02_run_text_benchmark.py` | corrected masking | standalone reproducible |
| BiomedCLIP branch in fusion | `scripts/03_run_fusion_reference.py` | historical fusion run | standalone historical |
| Guided fusion and DeLong | `scripts/03_run_fusion_reference.py` | historical masking | standalone historical |
| Five fusion strategies | `fusion/all_5_fusion_strategies.py` | historical notebook context | partial |
| GAT | `gat/patient_similarity_gat.py` | exploratory transductive graph | exploratory partial |
| Text/image occlusion | `scripts/04_generate_xai_examples.py` | historical interpretability run | standalone historical |
| Image de-identification | `scripts/05_deidentify_images.py` | preprocessing | standalone reproducible |

## Privacy Review

The committed tree should not include the full dataset, medical images, full patient metadata, checkpoints, model weights, generated patient-level result CSVs, or notebook outputs. The only data under `data/example/` is synthetic smoke-test metadata and empty class folders.

Absolute `/kaggle/` paths remain in historical notebook/script sources as provenance defaults. No `C:\`, `D:\`, `/Users/`, or `/home/` personal source paths remain in executable defaults after the CLI cleanup, except this audit note describing what was found before cleanup.
