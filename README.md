
# LLM Judge and Classifier Robustness for Prompt Injection

This repository documents an end-to-end safety evaluation pipeline:

**prompt-only failure -> human labeling -> llm-as-judge validation -> large-scale dataset generation -> classifier fine-tuning -> final classifier comparison**

The implementation and artifacts are built around DeBERTa-style classifier training, PromptGuard-style behavior checks, and multiple LLM judge protocols.

## Quick Demo

## Working Demo

This project includes an end-to-end working demo for prompt injection detection.

Pipeline:

User Prompt  
→ Victim LLM (Gemma / Qwen / Llama via Ollama)  
→ Generated Response  
→ Fine-tuned DeBERTa Classifier  
→ Fine-tuned PromptGuard Classifier  
→ Final Safety Verdict

### Run Demo

Activate environment:

```bash
conda activate jahnvi
move to project root : cd /workspace/amit/Jahanvi
run: python demo.py

## What was actually run in this repo

### 1) Prompt-only failure and CoT stress tests

- `prompt_vs_prompt_plus_cot_FAST_results.csv` (206 prompts)
  - Prompt-only accuracy: **0.5777**
  - Prompt+CoT accuracy: **0.5971**
- `cot_evaluation_results.csv` (306 prompts)
  - Weak CoT verdicts: **SAFE=254, INJECTION=52**
  - Strong CoT verdicts: **SAFE=31, INJECTION=275**
- `prompt_plus_cot_results.csv` (306 rows)
- `llm_judge_prompt_vs_prompt_cot.csv` (700 rows)

These files are the direct evidence for prompt-only instability and CoT sensitivity in judge outcomes.

### 2) Human labeling proof-of-work

- Master labeling pool: `annotation_master_3000.csv` (**3000 rows**)
- Annotator packets: `annotator_set_*_A.csv`, `annotator_set_*_B.csv`
- Annotated packets: `annotator_set_*_A_annotated.csv`, `annotator_set_*_B_annotated.csv`
- Combined labels: `annotations_combined.csv` (**3000 rows**)
- Disagreements: `annotation_disagreements.csv` (**235 rows**)

This is the audit trail used to justify manual annotation and disagreement handling.

### 3) LLM-as-judge validation

- `llm_judge_prompt_plus_cot_results.csv` (**8352 rows**)
  - True labels: **0=5796, 1=2556**
  - Normalized accuracy (`true_label` vs `pred_label`): **0.8994**
- `self_correction_results.csv` (306 rows)
  - New verdict distribution: **PASS=71, FAIL=235**
- SIGIR judge slices:
  - `sigir_deepset_prompt_only.csv` (116 rows), normalized accuracy: **0.8103**
  - `sigir_deepset_prompt_plus_cot.csv` (116 rows), normalized accuracy: **0.6810**
  - `sigir_xtram_prompt_only.csv` (600 rows, 598 clean verdict rows), normalized accuracy: **0.8930**
  - `sigir_xtram_prompt_plus_cot.csv` (600 rows), normalized accuracy: **0.7917**
- `spectral_data.csv` (336 rows; balanced `true_label`: 168 malicious / 168 benign)
- `committee_judged_results_balanced_v2.csv` (28 rows)

### 4) Large-scale dataset generation and scaling

- `final_training_dataset_3000_clean.csv` (**3000 rows**, labels: `0=2968`, `1=32`)
- `dataset_with_llm_labels.csv` (**3000 rows**, `llm_label`: `0=2804`, `1=165` with some missing)
- `dataset_with_behavior_labels.csv` (**3000 rows**, `behavior_label`: `0=1738`, `1=1262`)
- `final_large_dataset.csv` (**9298 rows**, labels: `0=8448`, `1=850`)
- `balanced_dataset.csv` (**2550 rows**, labels: `0=1700`, `1=850`)

### 5) Classifier fine-tuning and comparison

- Training script: `final_train_deberta.py` (uses `balanced_dataset.csv`)
- Comparison script: `eval_all_models.py`
- Output target: `final_model_comparison.csv`

Current repository status: `final_model_comparison.csv` exists but is **empty (0 rows)**, so the final comparison table is not yet materialized in this snapshot.

## Key scripts

- Human labeling: `split_for_annotators.py`, `combine_annotations.py`
- LLM judge labels: `generate_llm_labels.py`
- Dataset curation: `fill_final_dataset.py`
- Finetuning: `final_train_deberta.py`
- Multi-model evaluation: `eval_all_models.py`
- Prompt/CoT judge experiments:
  - `run_prompt_plus_cot_eval.py`
  - `run_llm_judge_prompt_vs_prompt_cot.py`
  - `run_prompt_vs_prompt_plus_cot_FAST.py`
  - `run_cot_evaluation.py`
  - `run_self_correction_eval.py`
  - `run_sigir_llm_judge.py`
  - `run_spectral_interrogation.py`
  - `run_evaluation_ollama.py`
  - `run_execution_matrix.py`

## Artifact files

- CoT artifacts:
  - `artifacts/cot/advbench_prompt_cot.json`
  - `artifacts/cot/deepset_prompt-injections_cot.json`
  - `artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json`

## Important integrity note

Some CSV verdict fields contain noisy free-form text in addition to `SAFE/UNSAFE`. Reported normalized accuracies in this README are computed after mapping standard values (`safe/benign/pass -> 0`, `unsafe/malicious/injection/fail -> 1`) and skipping unparsable entries.
=======
