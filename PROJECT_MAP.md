# Project Map (File-Exact)

This map links your exact files to the thesis narrative:

**prompt-only failure -> human labeling -> llm-as-judge validation -> large-scale dataset generation -> classifier fine-tuning -> final classifier comparison**

## 1) Prompt-only failure

### Runner scripts

- `run_prompt_plus_cot_eval.py`
- `run_llm_judge_prompt_vs_prompt_cot.py`
- `run_prompt_vs_prompt_plus_cot_FAST.py`
- `run_cot_evaluation.py`

### Result files

- `prompt_plus_cot_results.csv` (306 rows)
- `llm_judge_prompt_vs_prompt_cot.csv` (700 rows)
- `prompt_vs_prompt_plus_cot_FAST_results.csv` (206 rows)
- `cot_evaluation_results.csv` (306 rows)

### Key observed metrics

- Prompt-only accuracy: **0.5777**
- Prompt+CoT accuracy: **0.5971**
- CoT verdict shift is strong (`cot_evaluation_results.csv`: weak CoT mostly SAFE vs strong CoT mostly INJECTION)

## 2) Human labeling

### Label construction scripts

- `split_for_annotators.py`
- `combine_annotations.py`

### Evidence files

- `annotation_master_3000.csv`
- `annotator_set_1_A.csv`, `annotator_set_1_B.csv`
- `annotator_set_2_A.csv`, `annotator_set_2_B.csv`
- `annotator_set_3_A.csv`, `annotator_set_3_B.csv`
- `annotator_set_1_A_annotated.csv`, `annotator_set_1_B_annotated.csv`
- `annotator_set_2_A_annotated.csv`, `annotator_set_2_B_annotated.csv`
- `annotator_set_3_A_annotated.csv`, `annotator_set_3_B_annotated.csv`
- `annotations_combined.csv` (3000 rows)
- `annotation_disagreements.csv` (235 rows)

## 3) LLM-as-judge validation

### Runner scripts

- `generate_llm_labels.py`
- `run_self_correction_eval.py`
- `run_sigir_llm_judge.py`
- `run_spectral_interrogation.py`
- `run_committee_judging.py`
- `run_evaluation_ollama.py`
- `run_execution_matrix.py`

### Result files

- `llm_judge_prompt_plus_cot_results.csv` (8352 rows)
- `self_correction_results.csv` (306 rows)
- `sigir_deepset_prompt_only.csv` (116 rows)
- `sigir_deepset_prompt_plus_cot.csv` (116 rows)
- `sigir_xtram_prompt_only.csv` (600 rows)
- `sigir_xtram_prompt_plus_cot.csv` (600 rows)
- `spectral_data.csv` (336 rows)
- `committee_judged_results_balanced_v2.csv` (28 rows)
- `evaluation_matrix_ollama_results.csv` (460 rows)
- `execution_matrix_final.csv` (1000 rows)

### Key observed metrics

- `llm_judge_prompt_plus_cot_results.csv`: normalized accuracy **0.8994**
- `sigir_deepset_prompt_only.csv`: **0.8103**
- `sigir_deepset_prompt_plus_cot.csv`: **0.6810**
- `sigir_xtram_prompt_only.csv`: **0.8930** (on 598 parseable rows)
- `sigir_xtram_prompt_plus_cot.csv`: **0.7917**
- `self_correction_results.csv`: `PASS=71`, `FAIL=235`

## 4) Large-scale dataset generation

### Scripts and source

- `fill_final_dataset.py`
- prompt pool source: `wildjailbreak_prompts.csv`

### Output datasets

- `final_dataset_3000_ollama.csv` (3054 rows)
- `final_training_dataset_3000_clean.csv` (3000 rows)
- `dataset_with_llm_labels.csv` (3000 rows)
- `dataset_with_behavior_labels.csv` (3000 rows)
- `final_large_dataset.csv` (9298 rows)
- `balanced_dataset.csv` (2550 rows)

### Label distributions

- `final_training_dataset_3000_clean.csv`: `0=2968`, `1=32`
- `final_large_dataset.csv`: `0=8448`, `1=850`
- `balanced_dataset.csv`: `0=1700`, `1=850`

## 5) Classifier fine-tuning

### Scripts

- `final_train_deberta.py`

### Training dataset

- `balanced_dataset.csv`

## 6) Final classifier comparison

### Scripts

- `eval_all_models.py`

### Output target

- `final_model_comparison.csv`

### Current state

- `final_model_comparison.csv` is currently empty (0 rows), so this final stage is scripted but not yet exported as a populated artifact in this snapshot.

## Supporting artifact traces

- `artifacts/cot/advbench_prompt_cot.json`
- `artifacts/cot/deepset_prompt-injections_cot.json`
- `artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json`

These files preserve CoT artifact evidence used by the judge experiments.
