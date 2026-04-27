# Reproduce the Pipeline

This document reproduces the exact workflow represented by the current repository snapshot.

## Environment assumptions

- Python 3.10+
- Local Ollama server for LLM-judge scripts (`http://localhost:11434/api/generate`)
- HF Transformers stack for DeBERTa/PromptGuard scripts

## Stage A: Human labeling evidence

### Inputs

- `annotation_master_3000.csv`
- `annotator_set_*_A.csv`, `annotator_set_*_B.csv`
- `annotator_set_*_A_annotated.csv`, `annotator_set_*_B_annotated.csv`

### Run

```bash
python3 split_for_annotators.py
python3 combine_annotations.py
```

### Expected outputs

- `annotations_combined.csv` (3000 rows)
- `annotation_disagreements.csv` (235 rows)

## Stage B: Prompt-only failure and CoT sensitivity

### Run

```bash
python3 run_prompt_plus_cot_eval.py
python3 run_llm_judge_prompt_vs_prompt_cot.py
python3 run_prompt_vs_prompt_plus_cot_FAST.py
python3 run_cot_evaluation.py
```

### Expected outputs

- `prompt_plus_cot_results.csv` (306 rows)
- `llm_judge_prompt_vs_prompt_cot.csv` (700 rows)
- `prompt_vs_prompt_plus_cot_FAST_results.csv` (206 rows)
- `cot_evaluation_results.csv` (306 rows)

### Snapshot metrics from current repo

- Prompt-only accuracy (`true_label` vs `pred_prompt_only`): **0.5777**
- Prompt+CoT accuracy (`true_label` vs `pred_prompt_plus_cot`): **0.5971**
- Weak vs strong CoT verdict divergence visible in `cot_evaluation_results.csv`

## Stage C: LLM-as-judge validation

### Run

```bash
python3 generate_llm_labels.py
python3 run_self_correction_eval.py
python3 run_sigir_llm_judge.py
python3 run_spectral_interrogation.py
python3 run_committee_judging.py
python3 run_evaluation_ollama.py
python3 run_execution_matrix.py
```

### Expected outputs

- `dataset_with_llm_labels.csv` (3000 rows)
- `self_correction_results.csv` (306 rows)
- `sigir_deepset_prompt_only.csv` (116 rows)
- `sigir_deepset_prompt_plus_cot.csv` (116 rows)
- `sigir_xtram_prompt_only.csv` (600 rows)
- `sigir_xtram_prompt_plus_cot.csv` (600 rows)
- `spectral_data.csv` (336 rows)
- `committee_judged_results_balanced_v2.csv` (28 rows)
- `evaluation_matrix_ollama_results.csv` (460 rows)
- `execution_matrix_final.csv` (1000 rows)

### Snapshot metrics from current repo

- `llm_judge_prompt_plus_cot_results.csv`: normalized accuracy **0.8994** (`n=8352`)
- `sigir_deepset_prompt_only.csv`: **0.8103**
- `sigir_deepset_prompt_plus_cot.csv`: **0.6810**
- `sigir_xtram_prompt_only.csv`: **0.8930** (`n=598` parseable rows)
- `sigir_xtram_prompt_plus_cot.csv`: **0.7917**
- `self_correction_results.csv`: `PASS=71`, `FAIL=235`

## Stage D: Dataset scaling

### Run

```bash
python3 fill_final_dataset.py
```

### Expected outputs

- `final_training_dataset_3000_clean.csv` (3000 rows)
- `final_large_dataset.csv` (9298 rows)
- `balanced_dataset.csv` (2550 rows)
- `dataset_with_behavior_labels.csv` (3000 rows)

### Snapshot label distributions

- `final_training_dataset_3000_clean.csv`: `0=2968`, `1=32`
- `final_large_dataset.csv`: `0=8448`, `1=850`
- `balanced_dataset.csv`: `0=1700`, `1=850`
- `dataset_with_behavior_labels.csv`: `behavior_label 0=1738, 1=1262`

## Stage E: Classifier training and final comparison

### Run

```bash
python3 final_train_deberta.py
python3 eval_all_models.py
```

### Expected outputs

- DeBERTa classifier training/eval logs from `final_train_deberta.py`
- `final_model_comparison.csv` from `eval_all_models.py`

### Current snapshot status

- `final_model_comparison.csv` is present but empty (`0` rows).  
  The final comparison step is scripted but not yet captured in this checked-in artifact set.

## Integrity check script (optional)

If you want to verify row counts quickly:

```bash
python3 - <<'PY'
import csv
files = [
  "annotation_master_3000.csv","annotations_combined.csv","annotation_disagreements.csv",
  "prompt_vs_prompt_plus_cot_FAST_results.csv","llm_judge_prompt_plus_cot_results.csv",
  "final_training_dataset_3000_clean.csv","final_large_dataset.csv","balanced_dataset.csv",
  "final_model_comparison.csv"
]
for f in files:
    with open(f, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        r = csv.reader(fh)
        header = next(r, [])
        n = sum(1 for _ in r)
    print(f"{f}: rows={n}, cols={len(header)}")
PY
```
