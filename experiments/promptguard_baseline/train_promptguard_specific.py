# train_promptguard_specific.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "/workspace/amit/Jahanvi/promptguard"
# usage:
# python train_promptguard_specific.py gemma
# python train_promptguard_specific.py llama
# python train_promptguard_specific.py qwen

MODEL_MAP = {
    "gemma": "gemma2:2b",
    "llama": "llama3.1:8b",
    "qwen": "qwen2.5:14b"
}

REPO_ROOT = Path("/workspace/amit/Jahanvi")

DATA_PATH = REPO_ROOT / "final_large_dataset.csv"

SAVE_ROOT = REPO_ROOT / "results" / "promptguard_specific"

TEXT_COL_CANDIDATES = ["prompt", "response"]
LABEL_COL = "final_label"
VICTIM_COL_CANDIDATES = [
    "victim_model",
    "model_name",
    "model",
    "target_model"
]

# =========================================================
# INPUT ARG
# =========================================================

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python train_promptguard_specific.py [gemma|llama|qwen]"
    )

target_key = sys.argv[1].lower()

if target_key not in MODEL_MAP:
    raise ValueError("Use one of: gemma, llama, qwen")

target_model_name = MODEL_MAP[target_key]

print(f"\nTraining PromptGuard for victim model: {target_model_name}\n")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print(f"Original dataset size: {len(df)}")

# find victim model column
victim_col = None
for c in VICTIM_COL_CANDIDATES:
    if c in df.columns:
        victim_col = c
        break

if victim_col is None:
    raise ValueError(
        f"No victim model column found. Checked: {VICTIM_COL_CANDIDATES}"
    )

print(f"Using victim model column: {victim_col}")

# filter specific model
df = df[df[victim_col].astype(str) == target_model_name].copy()

print(f"Filtered dataset size: {len(df)}")

if len(df) == 0:
    raise ValueError(
        f"No rows found for model: {target_model_name}"
    )

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())

# keep needed cols
df = df.dropna(subset=TEXT_COL_CANDIDATES + [LABEL_COL])

df["text"] = (
    "Prompt: " + df["prompt"].astype(str)
    + "\nResponse: " + df["response"].astype(str)
)

df["label"] = df[LABEL_COL].astype(int)

# =========================================================
# SPLIT
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTrain size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

train_ds = Dataset.from_pandas(
    train_df[["text", "label"]].reset_index(drop=True)
)

test_ds = Dataset.from_pandas(
    test_df[["text", "label"]].reset_index(drop=True)
)

# =========================================================
# TOKENIZER
# =========================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# =========================================================
# MODEL
# =========================================================

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    report = classification_report(
        labels,
        preds,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "harmful_f1": report["1"]["f1-score"]
    }

# =========================================================
# TRAINING
# =========================================================

save_dir = SAVE_ROOT / f"{target_key}_promptguard"

training_args = TrainingArguments(
    output_dir=str(save_dir),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",

    num_train_epochs=3,
    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    weight_decay=0.01,
    logging_steps=50,

    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\nStarting training...\n")

trainer.train()

# =========================================================
# FINAL EVAL
# =========================================================

print("\nRunning final evaluation...\n")

preds = trainer.predict(test_ds)

y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_df["label"].values

print("\n==============================")
print(f"PROMPTGUARD ({target_key.upper()}) RESULTS")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)
