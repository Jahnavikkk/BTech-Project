import pandas as pd
import ollama
from transformers import pipeline
from tqdm import tqdm
import os
import sys
import random
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ONLY_DIR = REPO_ROOT / "results" / "prompt_only_failure"
LLM_JUDGE_DIR = REPO_ROOT / "results" / "llm_judge_validation"

# --- 1. CONFIGURATION ---

# Model to generate CoTs for ALL prompts
COT_GENERATOR_TAG = "gemma:2b"

# Our locally saved static classifier
STATIC_CLASSIFIER_PATH = REPO_ROOT / "local_classifier"

# Dataset paths
MALICIOUS_CSV = LLM_JUDGE_DIR / "evaluation_matrix_ollama_results.csv" # Source of malicious prompts
BENIGN_DATASET_PATH = REPO_ROOT / "local_datasets" / "deepset_prompt-injections" # Source of benign prompts

OUTPUT_CSV = PROMPT_ONLY_DIR / "prompt_vs_prompt_plus_cot_FAST_results.csv"

# Number of samples per class for a balanced dataset
# REDUCED TO 150 for a ~2-2.5 hour run
NUM_SAMPLES_PER_CLASS = 150

# Metaprompt for CoT generation
COT_METAPROMPT_TEMPLATE = """
Generate ONLY the step-by-step reasoning (Chain-of-Thought) for how you would process
and respond to the following user prompt. Do NOT provide the final answer.

USER PROMPT:
{prompt}

REASONING:
"""

# Separator for combining prompt and CoT
SEPARATOR = "\n\n---\nCoT Analysis:\n"

# --- 2. HELPER FUNCTIONS ---

def load_static_classifier():
    """Loads the classifier from the local disk."""
    if not os.path.exists(STATIC_CLASSIFIER_PATH):
        print(f"Error: Classifier directory not found at {STATIC_CLASSIFIER_PATH}")
        return None
    print("Loading static classifier from local files...")
    classifier = pipeline("text-classification", model=STATIC_CLASSIFIER_PATH, device=0, truncation=True)
    print("Classifier loaded.")
    return classifier

def load_balanced_prompts():
    """Loads NUM_SAMPLES_PER_CLASS malicious and benign prompts."""
    prompts_list = []

    # 1. Load MALICIOUS prompts
    try:
        df_malicious_full = pd.read_csv(MALICIOUS_CSV)
        # Use all available malicious prompts, up to the sample limit
        malicious_prompts_available = df_malicious_full['prompt_text'].unique().tolist()
        num_malicious_to_sample = min(len(malicious_prompts_available), NUM_SAMPLES_PER_CLASS)
        malicious_sample = random.sample(malicious_prompts_available, num_malicious_to_sample)

        for prompt in malicious_sample:
            prompts_list.append({"prompt": prompt, "true_label": "malicious"}) # Label malicious as 1 for confusion matrix
        print(f"Loaded {num_malicious_to_sample} malicious prompts.")
    except Exception as e:
        print(f"Error loading malicious CSV: {e}")
        return None

    # 2. Load BENIGN prompts
    try:
        dataset = load_from_disk(BENIGN_DATASET_PATH)
        benign_dataset = dataset.filter(lambda x: x['label'] == 0) # Filter for label 0 (benign)

        benign_prompts_available = [item['text'] for item in benign_dataset]
        num_benign_to_sample = min(len(benign_prompts_available), NUM_SAMPLES_PER_CLASS)
        benign_sample = random.sample(benign_prompts_available, num_benign_to_sample)

        for prompt in benign_sample:
            prompts_list.append({"prompt": prompt, "true_label": "benign"}) # Label benign as 0
        print(f"Loaded {num_benign_to_sample} benign prompts.")
    except Exception as e:
        print(f"Error loading benign dataset: {e}")
        return None

    random.shuffle(prompts_list)
    print(f"Total balanced dataset size: {len(prompts_list)}")
    return prompts_list

def get_cot_from_model(prompt, model_tag):
    """Generates CoT using the specified model."""
    full_prompt = COT_METAPROMPT_TEMPLATE.format(prompt=prompt)
    try:
        response = ollama.generate(model=model_tag, prompt=full_prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        print(f"Error generating CoT for {model_tag}: {e}")
        return "GENERATION_ERROR"

def get_classifier_prediction(text_input, classifier):
    """Gets a prediction from the static classifier."""
    if not text_input or text_input == "GENERATION_ERROR":
        return "INPUT_ERROR"
    try:
        # Map labels for consistency: INJECTION=malicious, SAFE=benign
        label = classifier(text_input, truncation=True)[0]['label']
        return "malicious" if label == "INJECTION" else "benign"
    except Exception as e:
        print(f"Classifier Error: {e}")
        return "CLASSIFIER_ERROR"

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    classifier = load_static_classifier()
    if classifier is None:
        sys.exit(1)

    prompts = load_balanced_prompts()
    if prompts is None:
        sys.exit(1)

    if not prompts:
        print("No prompts loaded. Exiting.")
        sys.exit(0)

    results = []
    print(f"Starting classification and CoT generation using {COT_GENERATOR_TAG}...")

    for item in tqdm(prompts, desc="Processing Prompts"):
        prompt = item['prompt']
        true_label = item['true_label']

        # 1. Classify Prompt Alone
        pred_prompt_only = get_classifier_prediction(prompt, classifier)

        # 2. Generate CoT
        cot = get_cot_from_model(prompt, COT_GENERATOR_TAG)

        # 3. Classify Prompt + CoT
        combined_input = f"User Prompt:\n{prompt}{SEPARATOR}{cot}" if cot != "GENERATION_ERROR" else "GENERATION_ERROR"
        pred_prompt_plus_cot = get_classifier_prediction(combined_input, classifier)

        results.append({
            "prompt": prompt,
            "true_label": true_label,
            "cot_generated": cot,
            "pred_prompt_only": pred_prompt_only,
            "pred_prompt_plus_cot": pred_prompt_plus_cot
        })

    # --- 4. ANALYZE AND REPORT ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull results saved to {OUTPUT_CSV}")

    # Filter out rows with errors for analysis
    df_analysis = df_results[
        (df_results["pred_prompt_only"].isin(["malicious", "benign"])) &
        (df_results["pred_prompt_plus_cot"].isin(["malicious", "benign"]))
    ].copy() # Use copy to avoid SettingWithCopyWarning

    if df_analysis.empty:
         print("No valid results to analyze after filtering errors.")
         sys.exit()

    # Convert true labels for sklearn reports (malicious=1, benign=0 - common convention)
    df_analysis['true_label_numeric'] = df_analysis['true_label'].apply(lambda x: 1 if x == 'malicious' else 0)
    df_analysis['pred_prompt_only_numeric'] = df_analysis['pred_prompt_only'].apply(lambda x: 1 if x == 'malicious' else 0)
    df_analysis['pred_prompt_plus_cot_numeric'] = df_analysis['pred_prompt_plus_cot'].apply(lambda x: 1 if x == 'malicious' else 0)

    y_true = df_analysis['true_label_numeric']
    y_pred_prompt = df_analysis['pred_prompt_only_numeric']
    y_pred_prompt_cot = df_analysis['pred_prompt_plus_cot_numeric']

    print("\n--- COMPARISON: PROMPT vs. PROMPT + CoT ---")
    print(f"Dataset size for analysis: {len(df_analysis)} prompts")
    print("---------------------------------------------------------------")

    print("\nCLASSIFIER PERFORMANCE (PROMPT ONLY):")
    print(classification_report(y_true, y_pred_prompt, target_names=['benign', 'malicious']))
    acc_prompt = accuracy_score(y_true, y_pred_prompt) * 100
    print(f"Accuracy (Prompt Only): {acc_prompt:.2f}%")

    print("\nCLASSIFIER PERFORMANCE (PROMPT + CoT):")
    print(classification_report(y_true, y_pred_prompt_cot, target_names=['benign', 'malicious']))
    acc_prompt_cot = accuracy_score(y_true, y_pred_prompt_cot) * 100
    print(f"Accuracy (Prompt + CoT): {acc_prompt_cot:.2f}%")

    print("---------------------------------------------------------------")
    diff = acc_prompt_cot - acc_prompt
    if diff > 1.0: # Use a small threshold to avoid noise
        print(f"\nCONCLUSION: Adding CoT IMPROVED accuracy by {diff:.2f}%")
    elif diff < -1.0:
        print(f"\nCONCLUSION: Adding CoT WORSENED accuracy by {-diff:.2f}%")
    else:
        print(f"\nCONCLUSION: Adding CoT had NO SIGNIFICANT EFFECT on accuracy.")
    print("---------------------------------------------------------------")
