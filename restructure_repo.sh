#!/usr/bin/env bash
set -euo pipefail

# Deterministic, professor-safe repository restructuring script.
# - Safe to run from repo root.
# - Uses mkdir -p for directories.
# - Moves only when source exists.
# - Never overwrites existing destination files.
# - Preserves annotation evidence files.
# - Archives old/extra files into _archive_local/ instead of deleting.
# - Does NOT touch model weights/checkpoints.

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(pwd)"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "[ERROR] .git not found in current directory."
  echo "Run this script from the repository root."
  exit 1
fi

echo "[INFO] Running ${SCRIPT_NAME} in: ${REPO_ROOT}"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
safe_mkdir() {
  local dir="$1"
  mkdir -p "$dir"
}

safe_move() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "$src" ]]; then
    echo "[SKIP] Missing source: $src"
    return 0
  fi

  if [[ -e "$dst" ]]; then
    echo "[SKIP] Destination exists (no overwrite): $dst"
    return 0
  fi

  local dst_dir
  dst_dir="$(dirname "$dst")"
  mkdir -p "$dst_dir"

  echo "[MOVE] $src -> $dst"
  mv "$src" "$dst"
}

archive_file() {
  local src="$1"
  local dst="${REPO_ROOT}/_archive_local/$src"
  safe_move "$src" "$dst"
}

# Move from root if present; otherwise restore from _archive_local if present.
safe_move_or_restore() {
  local rel="$1"
  local dst="$2"
  if [[ -e "$rel" ]]; then
    safe_move "$rel" "$dst"
    return 0
  fi
  if [[ -e "_archive_local/$rel" ]]; then
    safe_move "_archive_local/$rel" "$dst"
    return 0
  fi
  echo "[SKIP] Missing in root and archive: $rel"
}

# -------------------------------------------------------------------
# Create required folder structure
# -------------------------------------------------------------------
safe_mkdir "${REPO_ROOT}/data/human_labeling"
safe_mkdir "${REPO_ROOT}/data/prompts"
safe_mkdir "${REPO_ROOT}/data/scaled_datasets"

safe_mkdir "${REPO_ROOT}/src/human_labeling"
safe_mkdir "${REPO_ROOT}/src/llm_judge_validation"
safe_mkdir "${REPO_ROOT}/src/dataset_generation"
safe_mkdir "${REPO_ROOT}/src/finetuning"
safe_mkdir "${REPO_ROOT}/src/comparison"

safe_mkdir "${REPO_ROOT}/experiments/prompt_only_vs_cot"
safe_mkdir "${REPO_ROOT}/experiments/A1_static_classifier"
safe_mkdir "${REPO_ROOT}/experiments/promptguard_baseline"
safe_mkdir "${REPO_ROOT}/experiments/A2_cot_static"
safe_mkdir "${REPO_ROOT}/experiments/A3_llm_judge"
safe_mkdir "${REPO_ROOT}/experiments/A4_pipeline"
safe_mkdir "${REPO_ROOT}/experiments/self_correction"
safe_mkdir "${REPO_ROOT}/experiments/committee_judge"
safe_mkdir "${REPO_ROOT}/experiments/committee_analysis"
safe_mkdir "${REPO_ROOT}/experiments/sigir_eval"
safe_mkdir "${REPO_ROOT}/experiments/spectral_eval"
safe_mkdir "${REPO_ROOT}/experiments/llm_judge_matrix"
safe_mkdir "${REPO_ROOT}/experiments/agreement_analysis"

safe_mkdir "${REPO_ROOT}/results/prompt_only_failure"
safe_mkdir "${REPO_ROOT}/results/llm_judge_validation"
safe_mkdir "${REPO_ROOT}/results/A4_pipeline"
safe_mkdir "${REPO_ROOT}/results/committee"
safe_mkdir "${REPO_ROOT}/results/final_comparison"
safe_mkdir "${REPO_ROOT}/results/artifacts/cot"

safe_mkdir "${REPO_ROOT}/report/figures"
safe_mkdir "${REPO_ROOT}/report/tables"
safe_mkdir "${REPO_ROOT}/report/appendix"
safe_mkdir "${REPO_ROOT}/notebooks"

safe_mkdir "${REPO_ROOT}/_archive_local"

# -------------------------------------------------------------------
# Move files according to the exact mapping
# -------------------------------------------------------------------

# data/human_labeling (annotation evidence preserved)
safe_move "annotation_master_3000.csv" "data/human_labeling/annotation_master_3000.csv"
safe_move "annotator_set_1_A.csv" "data/human_labeling/annotator_set_1_A.csv"
safe_move "annotator_set_1_B.csv" "data/human_labeling/annotator_set_1_B.csv"
safe_move "annotator_set_2_A.csv" "data/human_labeling/annotator_set_2_A.csv"
safe_move "annotator_set_2_B.csv" "data/human_labeling/annotator_set_2_B.csv"
safe_move "annotator_set_3_A.csv" "data/human_labeling/annotator_set_3_A.csv"
safe_move "annotator_set_3_B.csv" "data/human_labeling/annotator_set_3_B.csv"
safe_move "annotator_set_1_A_annotated.csv" "data/human_labeling/annotator_set_1_A_annotated.csv"
safe_move "annotator_set_1_B_annotated.csv" "data/human_labeling/annotator_set_1_B_annotated.csv"
safe_move "annotator_set_2_A_annotated.csv" "data/human_labeling/annotator_set_2_A_annotated.csv"
safe_move "annotator_set_2_B_annotated.csv" "data/human_labeling/annotator_set_2_B_annotated.csv"
safe_move "annotator_set_3_A_annotated.csv" "data/human_labeling/annotator_set_3_A_annotated.csv"
safe_move "annotator_set_3_B_annotated.csv" "data/human_labeling/annotator_set_3_B_annotated.csv"
safe_move "annotations_combined.csv" "data/human_labeling/annotations_combined.csv"
safe_move "annotation_disagreements.csv" "data/human_labeling/annotation_disagreements.csv"

# data/prompts
safe_move "wildjailbreak_prompts.csv" "data/prompts/wildjailbreak_prompts.csv"

# data/scaled_datasets
safe_move "final_dataset_3000_ollama.csv" "data/scaled_datasets/final_dataset_3000_ollama.csv"
safe_move "final_training_dataset_3000_clean.csv" "data/scaled_datasets/final_training_dataset_3000_clean.csv"
safe_move "dataset_with_llm_labels.csv" "data/scaled_datasets/dataset_with_llm_labels.csv"
safe_move "dataset_with_behavior_labels.csv" "data/scaled_datasets/dataset_with_behavior_labels.csv"
safe_move "balanced_dataset.csv" "data/scaled_datasets/balanced_dataset.csv"
safe_move "final_large_dataset.csv" "data/scaled_datasets/final_large_dataset.csv"

# src/human_labeling
safe_move "split_for_annotators.py" "src/human_labeling/split_for_annotators.py"
safe_move "combine_annotations.py" "src/human_labeling/combine_annotations.py"

# src/llm_judge_validation
safe_move "generate_llm_labels.py" "src/llm_judge_validation/generate_llm_labels.py"
safe_move "run_evaluation_ollama.py" "src/llm_judge_validation/run_evaluation_ollama.py"

# src/dataset_generation
safe_move "fill_final_dataset.py" "src/dataset_generation/fill_final_dataset.py"

# src/finetuning
safe_move "final_train_deberta.py" "src/finetuning/final_train_deberta.py"

# src/comparison
safe_move "eval_all_models.py" "src/comparison/eval_all_models.py"

# experiments
safe_move "run_prompt_plus_cot_eval.py" "experiments/prompt_only_vs_cot/run_prompt_plus_cot_eval.py"
safe_move "run_llm_judge_prompt_vs_prompt_cot.py" "experiments/prompt_only_vs_cot/run_llm_judge_prompt_vs_prompt_cot.py"
safe_move "run_prompt_vs_prompt_plus_cot_FAST.py" "experiments/prompt_only_vs_cot/run_prompt_vs_prompt_plus_cot_FAST.py"
safe_move "run_cot_evaluation.py" "experiments/prompt_only_vs_cot/run_cot_evaluation.py"
safe_move_or_restore "run_baseline_deberta.py" "experiments/A1_static_classifier/run_baseline_deberta.py"
safe_move_or_restore "finetune_promptguard.py" "experiments/promptguard_baseline/finetune_promptguard.py"
safe_move_or_restore "finetune_promptguard_9k.py" "experiments/promptguard_baseline/finetune_promptguard_9k.py"
safe_move_or_restore "approach_2_cot.py" "experiments/A2_cot_static/approach_2_cot.py"
safe_move_or_restore "run_a2_self_correction_deberta.py" "experiments/A2_cot_static/run_a2_self_correction_deberta.py"
safe_move_or_restore "run_finetune_prompt_cot.py" "experiments/A2_cot_static/run_finetune_prompt_cot.py"
safe_move_or_restore "run_a4_pipeline.py" "experiments/A4_pipeline/run_a4_pipeline.py"
safe_move_or_restore "a4_small_cot_gemma2b.py" "experiments/A4_pipeline/a4_small_cot_gemma2b.py"
safe_move_or_restore "a4_large_judge_gemma27b.py" "experiments/A4_pipeline/a4_large_judge_gemma27b.py"
safe_move "run_self_correction_eval.py" "experiments/self_correction/run_self_correction_eval.py"
safe_move "run_committee_judging.py" "experiments/committee_judge/run_committee_judging.py"
safe_move_or_restore "run_committee_generation.py" "experiments/committee_judge/run_committee_generation.py"
safe_move_or_restore "run_committee_analysis.py" "experiments/committee_analysis/run_committee_analysis.py"
safe_move_or_restore "analyze_committee_results.py" "experiments/committee_analysis/analyze_committee_results.py"
safe_move "run_sigir_llm_judge.py" "experiments/sigir_eval/run_sigir_llm_judge.py"
safe_move "run_spectral_interrogation.py" "experiments/spectral_eval/run_spectral_interrogation.py"
safe_move_or_restore "analyze_spectral_data.py" "experiments/spectral_eval/analyze_spectral_data.py"
safe_move "run_execution_matrix.py" "experiments/llm_judge_matrix/run_execution_matrix.py"
safe_move_or_restore "agreement_llm_vs_human.py" "experiments/agreement_analysis/agreement_llm_vs_human.py"
safe_move_or_restore "agreement_check.py" "experiments/agreement_analysis/agreement_check.py"
safe_move_or_restore "analyze_results.py" "experiments/agreement_analysis/analyze_results.py"

# results/prompt_only_failure
safe_move "prompt_plus_cot_results.csv" "results/prompt_only_failure/prompt_plus_cot_results.csv"
safe_move "llm_judge_prompt_plus_cot_results.csv" "results/prompt_only_failure/llm_judge_prompt_plus_cot_results.csv"
safe_move "llm_judge_prompt_vs_prompt_cot.csv" "results/prompt_only_failure/llm_judge_prompt_vs_prompt_cot.csv"
safe_move "prompt_vs_prompt_plus_cot_FAST_results.csv" "results/prompt_only_failure/prompt_vs_prompt_plus_cot_FAST_results.csv"
safe_move "cot_evaluation_results.csv" "results/prompt_only_failure/cot_evaluation_results.csv"

# results/llm_judge_validation
safe_move "self_correction_results.csv" "results/llm_judge_validation/self_correction_results.csv"
safe_move "evaluation_matrix_ollama_results.csv" "results/llm_judge_validation/evaluation_matrix_ollama_results.csv"
safe_move "execution_matrix_final.csv" "results/llm_judge_validation/execution_matrix_final.csv"
safe_move "committee_generations_balanced_v3.csv" "results/llm_judge_validation/committee_generations_balanced_v3.csv"
safe_move "committee_judged_results_balanced_v2.csv" "results/llm_judge_validation/committee_judged_results_balanced_v2.csv"
safe_move "sigir_deepset_prompt_only.csv" "results/llm_judge_validation/sigir_deepset_prompt_only.csv"
safe_move "sigir_deepset_prompt_plus_cot.csv" "results/llm_judge_validation/sigir_deepset_prompt_plus_cot.csv"
safe_move "sigir_xtram_prompt_only.csv" "results/llm_judge_validation/sigir_xtram_prompt_only.csv"
safe_move "sigir_xtram_prompt_plus_cot.csv" "results/llm_judge_validation/sigir_xtram_prompt_plus_cot.csv"
safe_move "spectral_data.csv" "results/llm_judge_validation/spectral_data.csv"
safe_move "execution_matrix_1000.csv" "results/llm_judge_validation/execution_matrix_1000.csv"
safe_move "execution_matrix_cleaned.csv" "results/llm_judge_validation/execution_matrix_cleaned.csv"
safe_move "committee_generations.csv" "results/committee/committee_generations.csv"
safe_move "committee_generations_balanced_v2.csv" "results/committee/committee_generations_balanced_v2.csv"
safe_move "committee_judged_results.csv" "results/committee/committee_judged_results.csv"
safe_move "a4_pipeline_results.csv" "results/A4_pipeline/a4_pipeline_results.csv"

# results/final_comparison
safe_move "final_model_comparison.csv" "results/final_comparison/final_model_comparison.csv"
safe_move "generated_dataset.csv" "data/scaled_datasets/generated_dataset.csv"
safe_move "final_training_dataset.csv" "data/scaled_datasets/final_training_dataset.csv"

# artifacts/cot -> results/artifacts/cot
safe_move "artifacts/cot/advbench_prompt_cot.json" "results/artifacts/cot/advbench_prompt_cot.json"
safe_move "artifacts/cot/deepset_prompt-injections_cot.json" "results/artifacts/cot/deepset_prompt-injections_cot.json"
safe_move "artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json" "results/artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json"

# -------------------------------------------------------------------
# Archive old/extra files (no deletion)
# -------------------------------------------------------------------

# Keep critical root files untouched:
# - .git
# - .gitignore
# - README.md
# - restructure_repo.sh
# - docs generated for reproducibility and review

EXTRA_FILES=(
  "a2_self_correction_deberta.log"
  "a4_output.log"
  "advbench_cot.log"
  "cot_flip.log"
  "cot_judge.log"
  "cot_output.log"
  "deepset_remaining.log"
  "execution_matrix.log"
  "finetune_cot.log"
  "generation_output.log"
  "judging_output.log"
  "llm_judge_prompt_plus_cot.log"
  "llmdge_prompt_plus_cot.log"
  "ollama_output.log"
  "output.log"
  "prompt_vs_cot_FAST_output.log"
  "refill_lang.log"
  "self_correction_output.log"
  "sigir_judge.log"
  "spectral_gen_output_v2.log"
  "generated_dataset_backup.csv"
  "clean_committee_generations.csv"
  ".strict_dup_hashes.json"
  ".annotation_strict_audit.json"
  ".annotator_pair_semantic_compare.json"
  ".annotator_content_key_compare.json"
  ".metrics_snapshot.json"
)

for f in "${EXTRA_FILES[@]}"; do
  # Never archive protected root files.
  if [[ "$f" == ".git" || "$f" == ".gitignore" || "$f" == "README.md" ]]; then
    continue
  fi
  archive_file "$f"
done

# Archive extra scripts not in final KEEP mapping (without deleting)
EXTRA_SCRIPTS=(
  "analyze_correction_results.py"
  "approach_3_gemma.py"
  "approach_3_gemma_upd.py"
  "approach_3_llm.py"
  "check_data_integrity.py"
  "check_final_dataset.py"
  "clean_deepset_english.py"
  "clean_deepset_langdetect.py"
  "download_classifier.py"
  "download_datasets.py"
  "download_model.py"
  "eval_qualifire.py"
  "exp2_behavior_model.py"
  "flatten_for_annotation.py"
  "full_pipeline.py"
  "generate_advbench_cot.py"
  "generate_behavior_labels.py"
  "generate_cot_ollama.py"
  "inspect_advbench.py"
  "inspect_arena.py"
  "omt_dataset_analysis.py"
  "omt_final_dataset_eda.py"
  "omt_finetuneattempt1.py"
  "plot.py"
  "refill_deepset_langdetect.py"
  "run_deepset_remaining.py"
  "run_prompt+cot_v2.py"
  "sanitycheck.py"
  "sanitycheck2.py"
  "sanitycheck3.py"
  "sanitychek4.py"
  "somefix.py"
  "test.py"
  "test_dataset.py"
  "train_deberta_specific.py"
  "withoutcot.py"
)

for f in "${EXTRA_SCRIPTS[@]}"; do
  # Do not touch potential model weights/checkpoints (explicit safeguard).
  case "$f" in
    *.pt|*.bin|*.safetensors|*.ckpt)
      echo "[SKIP] Weight/checkpoint protected: $f"
      continue
      ;;
  esac
  archive_file "$f"
done

# Optionally archive stale artifacts directory if now empty except cot moved.
if [[ -d "artifacts" ]]; then
  # Do not remove; just archive if it still contains files other than moved cot JSON.
  if [[ -n "$(ls -A artifacts 2>/dev/null || true)" ]]; then
    archive_file "artifacts"
  fi
fi

echo "[DONE] Restructure pass complete."
echo "[INFO] Review changes with: git status"
echo "[INFO] Archived extras are in: _archive_local/"
