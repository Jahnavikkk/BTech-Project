# evaluate_with_prompt_guard.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os, json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "meta-llama/Prompt-Guard-86M"

def load_prompt_guard():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def map_prediction(predicted_class_id):
    # PromptGuard: 0=benign, 1=injection, 2=jailbreak
    return 1 if predicted_class_id in [1, 2] else 0

def evaluate_with_cot_only(dataset, tokenizer, model, cot_file):
    y_true, y_pred = [], []

    with open(cot_file, "r") as f:
        cot_lookup = json.load(f)

    for example in tqdm(dataset, desc=f"Evaluating CoT-only"):
        true_label = example["label"]
        cot = cot_lookup.get(str(example["id"]), "")

        inputs = tokenizer(
            cot,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predicted_label = map_prediction(predicted_class_id)

        y_true.append(true_label)
        y_pred.append(predicted_label)

    return y_true, y_pred

def report_results(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"\n--- Results (A2: CoT-only) on {dataset_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    datasets_to_test = [
        ("deepset/prompt-injections", "test",
         "artifacts/cot/deepset_prompt-injections_cot.json"),
        ("xTRam1/safe-guard-prompt-injection", "train[:600]",
         "artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json")
    ]

    tokenizer, model = load_prompt_guard()

    for dataset_name, split, cot_file in datasets_to_test:
        print(f"\nLoading dataset: {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, split=split)

        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", list(range(len(dataset))))

        if os.path.exists(cot_file):
            y_true, y_pred = evaluate_with_cot_only(dataset, tokenizer, model, cot_file)
            report_results(y_true, y_pred, dataset_name)
        else:
            print(f"⚠️ Skipping {dataset_name}, no CoT file found at {cot_file}")
