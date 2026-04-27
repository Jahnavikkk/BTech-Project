import json
import random
from datasets import load_from_disk, load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 512
SEED = 42

# ------------------------
# LOAD CoT FILES
# ------------------------

with open("artifacts/cot/deepset_prompt-injections_cot.json") as f:
    DEEPSET_COT = json.load(f)

with open("artifacts/cot/xTRam1_safe-guard-prompt-injection_cot.json") as f:
    XTRAM_COT = json.load(f)

# ⚠️ This file will exist after AdvBench CoT finishes
with open("artifacts/cot/advbench_prompt_cot.json") as f:
    ADVBENCH_COT = json.load(f)

# ------------------------
# BUILD DATASET
# ------------------------

rows = []

def format_input(prompt, cot):
    return f"PROMPT:\n{prompt}\n\nMODEL REASONING:\n{cot}"

# ---- DeepSet ----
deepset = load_from_disk("local_datasets/deepset_prompt-injections")
for i, ex in enumerate(deepset):
    rows.append({
        "text": format_input(ex["text"], DEEPSET_COT.get(str(i), "")),
        "label": int(ex["label"])
    })

# ---- xTRam ----
xtram = load_from_disk("local_datasets/xTRam1_safe-guard-prompt-injection")
for i, ex in enumerate(xtram):
    rows.append({
        "text": format_input(ex["text"], XTRAM_COT.get(str(i), "")),
        "label": int(ex["label"])
    })

# ---- AdvBench (malicious only) ----
advbench = load_dataset(
    "arrow",
    data_files={"train": "BTP_FINAL_DATA/data-00000-of-00001.arrow"}
)["train"]

for i, ex in enumerate(advbench):
    rows.append({
        "text": format_input(ex["prompt"], ADVBENCH_COT.get(str(i), "")),
        "label": 1
    })

print(f"Total training examples: {len(rows)}")

# ------------------------
# TRAIN / TEST SPLIT
# ------------------------

train_rows, test_rows = train_test_split(
    rows,
    test_size=0.2,
    stratify=[r["label"] for r in rows],
    random_state=SEED
)

train_ds = Dataset.from_list(train_rows)
test_ds = Dataset.from_list(test_rows)

# ------------------------
# TOKENIZATION
# ------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------
# MODEL
# ------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ------------------------
# METRICS
# ------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }

# ------------------------
# TRAINING
# ------------------------

training_args = TrainingArguments(
    output_dir="finetune_prompt_cot",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
