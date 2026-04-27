import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, recall_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================

CLASSIFIER_MODEL = "microsoft/deberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = [
    ("deepset/prompt-injections", "test",
     "artifacts/cot/deepset_prompt-injections_cot.json"),
    ("xTRam1/safe-guard-prompt-injection", "train[:600]",
     "artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json")
]

# =========================
# LOAD CLASSIFIER
# =========================

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_MODEL, num_labels=2
).to(DEVICE)
model.eval()

# Label convention:
# 0 = SAFE, 1 = INJECTION

def classify(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return pred

# =========================
# MAIN
# =========================

for dataset_name, split, cot_file in DATASETS:
    print(f"\n=== A2 Self-Correction with DeBERTa on {dataset_name} ===")

    dataset = load_dataset(dataset_name, split=split)

    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))

    with open(cot_file) as f:
        cot_map = json.load(f)

    y_true, y_pred = [], []

    for ex in tqdm(dataset):
        ex_id = str(ex["id"])
        if ex_id not in cot_map:
            continue

        cot_text = cot_map[ex_id]
        true_label = ex["label"]  # 0 benign, 1 malicious

        pred_label = classify(cot_text)

        y_true.append(true_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Self-Correction Rate (Recall on malicious): {recall:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
