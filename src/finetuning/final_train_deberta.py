import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALED_DIR = REPO_ROOT / "data" / "scaled_datasets"

# LOAD DATA
df = pd.read_csv(SCALED_DIR / "balanced_dataset.csv")

# FORMAT
df["text"] = df.apply(lambda x: f"Prompt: {x['prompt']}\nResponse: {x['response']}", axis=1)

# SPLIT
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df[["text", "final_label"]])
test_ds = Dataset.from_pandas(test_df[["text", "final_label"]])

# TOKENIZER
model_name = "/workspace/amit/Jahanvi/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(x):
    return tokenizer(x["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("final_label", "labels")
test_ds = test_ds.rename_column("final_label", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")

# MODEL
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# TRAINING
args = TrainingArguments(
    output_dir=str(REPO_ROOT / "results" / "final_comparison" / "final_model"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

# PREDICT
preds = trainer.predict(test_ds)

if isinstance(preds.predictions, tuple):
    logits = preds.predictions[0]
else:
    logits = preds.predictions

probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
y_pred = (probs > 0.5).astype(int)
y_true = preds.label_ids

print("\n=== FINAL MODEL RESULTS ===")
print(classification_report(y_true, y_pred, digits=4))
