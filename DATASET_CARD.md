# Dataset Card: Prompt-Injection Safety Labeling and Judge Validation

## Scope

This repository contains a multi-stage dataset pipeline for prompt-injection safety:

1. Human annotation on a 3,000-example evaluation set
2. LLM-as-judge labeling and cross-checking
3. Scaling to a larger judged corpus
4. Preparing balanced data for classifier fine-tuning

## Primary dataset artifacts

### Human labeling artifacts

- `annotation_master_3000.csv` (3000 rows, 6 columns)
- `annotator_set_1_A.csv`, `annotator_set_1_B.csv`
- `annotator_set_2_A.csv`, `annotator_set_2_B.csv`
- `annotator_set_3_A.csv`, `annotator_set_3_B.csv`
- `annotator_set_1_A_annotated.csv`, `annotator_set_1_B_annotated.csv`
- `annotator_set_2_A_annotated.csv`, `annotator_set_2_B_annotated.csv`
- `annotator_set_3_A_annotated.csv`, `annotator_set_3_B_annotated.csv`
- `annotations_combined.csv` (3000 rows, 9 columns)
- `annotation_disagreements.csv` (235 rows, 9 columns)

### LLM judge and derived labels

- `dataset_with_llm_labels.csv` (3000 rows)
  - `llm_label`: 0.0=2804, 1.0=165 (some missing labels)
- `dataset_with_behavior_labels.csv` (3000 rows)
  - `behavior_label`: 0=1738, 1=1262

### Scaled and training-ready datasets

- `final_training_dataset_3000_clean.csv` (3000 rows)
  - `final_label`: 0=2968, 1=32
- `final_large_dataset.csv` (9298 rows)
  - `final_label`: 0.0=8448, 1.0=850
- `balanced_dataset.csv` (2550 rows)
  - `final_label`: 0.0=1700, 1.0=850

## Label schema used across files

Observed conventions in this repo:

- Binary numeric: `0` / `1`
- Text binary: `benign` / `malicious`
- Judge-style text: `SAFE` / `UNSAFE`
- Evaluation-style text: `PASS` / `FAIL`
- CoT probe style: `SAFE` / `INJECTION`

For cross-file metric calculation, normalized mapping used in this repo documentation:

- `safe`, `benign`, `pass`, `0` -> **0**
- `unsafe`, `malicious`, `injection`, `fail`, `1` -> **1**

## How the dataset was built in this repository

### Step 1: prompt source

- `wildjailbreak_prompts.csv` is used as a large prompt pool.

### Step 2: human annotation flow

- `split_for_annotators.py` produces A/B annotator packets.
- Annotators produce `*_annotated.csv`.
- `combine_annotations.py` merges and produces:
  - `annotations_combined.csv`
  - `annotation_disagreements.csv`

### Step 3: llm-judge augmentation

- `generate_llm_labels.py` creates `dataset_with_llm_labels.csv`.
- Additional judge studies generate:
  - `llm_judge_prompt_plus_cot_results.csv`
  - `sigir_*.csv`
  - `self_correction_results.csv`
  - `spectral_data.csv`
  - `evaluation_matrix_ollama_results.csv`

### Step 4: scaling + curation

- `fill_final_dataset.py` and related scripts produce:
  - `final_training_dataset_3000_clean.csv`
  - `final_large_dataset.csv`
  - `balanced_dataset.csv`

## Known quality caveats

- Some judge output fields include non-standard free-form text, especially in `sigir_xtram_prompt_only.csv`.
- `dataset_with_llm_labels.csv` includes missing `llm_label` entries.
- Label conventions differ across files and require normalization before cross-file metrics.
- `final_model_comparison.csv` is present but empty in this snapshot; this affects end-stage reporting completeness.

## Intended use

- Benchmarking robustness of prompt-injection detectors and LLM judges
- Comparing prompt-only vs prompt+CoT behavior
- Training/fine-tuning a safety classifier (DeBERTa pipeline scripts in this repo)

## Not intended use

- Safety-critical deployment without further validation
- Interpreting any single judge run as absolute truth
- Cross-file comparisons without explicit label normalization
