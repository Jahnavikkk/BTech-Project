import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

DATASETS = {
    "DeepSet": {
        "name": "deepset/prompt-injections",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
    },
    "xTRAM": {
        "name": "xTRam1/safe-guard-prompt-injection",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
    },
}

DEVICE = 0  # GPU

# ------------------------
# LOAD CLASSIFIER
# ------------------------
print("Loading DeBERTa prompt injection classifier...")
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=DEVICE,
    truncation=True,
    batch_size=16,
)

# ------------------------
# EVALUATION FUNCTION
# ------------------------
def evaluate_dataset(ds_name, cfg):
    print(f"\n=== Evaluating on {ds_name} ===")

    dataset = load_dataset(cfg["name"], split=cfg["split"])

    y_true = []
    y_pred = []

    for ex in tqdm(dataset, desc=f"Running {ds_name}"):
        text = ex[cfg["text_col"]]
        label = ex[cfg["label_col"]]  # 0 = benign, 1 = malicious

        pred = classifier(text)[0]
        pred_label = 1 if pred["label"].upper() == "INJECTION" else 0

        y_true.append(label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1-score : {f1:.4f}")

    return acc, p, r, f1

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    results = {}

    for name, cfg in DATASETS.items():
        results[name] = evaluate_dataset(name, cfg)

    print("\n=== Summary ===")
    for name, (acc, p, r, f1) in results.items():
        print(f"{name}: Acc={acc:.3f}, P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
