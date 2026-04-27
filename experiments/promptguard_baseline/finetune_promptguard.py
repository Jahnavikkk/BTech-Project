import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("balanced_dataset.csv")

print("Dataset size:", len(df))
print(df["final_label"].value_counts())

df["text"] = df.apply(
    lambda x: f"Prompt: {x['prompt']}\nResponse: {x['response']}",
    axis=1
)

# -------------------------
# SPLIT
# -------------------------

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(
    train_df[["text", "final_label"]]
)

test_ds = Dataset.from_pandas(
    test_df[["text", "final_label"]]
)

# -------------------------
# MODEL
# -------------------------

model_path = "/workspace/amit/Jahanvi/promptguard"

tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(x):
    return tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column(
    "final_label",
    "labels"
)

test_ds = test_ds.rename_column(
    "final_label",
    "labels"
)

train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2
)

# -------------------------
# TRAINING
# -------------------------

args = TrainingArguments(
    output_dir="./promptguard_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()

# -------------------------
# EVALUATION
# -------------------------

preds = trainer.predict(test_ds)

if isinstance(preds.predictions, tuple):
    logits = preds.predictions[0]
else:
    logits = preds.predictions

probs = torch.softmax(
    torch.tensor(logits),
    dim=1
)[:, 1].numpy()

y_pred = (probs > 0.5).astype(int)
y_true = preds.label_ids

print("\n=== PROMPTGUARD FINETUNED RESULTS ===")
print(classification_report(
    y_true,
    y_pred,
    digits=4
))
