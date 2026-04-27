import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_distances

# =========================
# LOAD DATA (LLM LABELS)
# =========================
df = pd.read_csv("dataset_with_behavior_labels.csv")

# sanity
print("\nBehavior distribution:")
print(df["behavior_label"].value_counts())

# =========================
# SPLIT (PROMPT LEVEL)
# =========================
unique_prompts = df["prompt"].unique()
np.random.seed(42)

train_prompts = np.random.choice(unique_prompts, int(0.8 * len(unique_prompts)), replace=False)
test_prompts = list(set(unique_prompts) - set(train_prompts))

train_df = df[df["prompt"].isin(train_prompts)].copy()
test_df = df[df["prompt"].isin(test_prompts)].copy()

# =========================
# FORMAT INPUT
# =========================
def format_input(row):
    return f"Prompt: {row['prompt']}\nResponse: {row['response']}"

train_df["text"] = train_df.apply(format_input, axis=1)
test_df["text"] = test_df.apply(format_input, axis=1)

# =========================
# DATASETS
# =========================
train_ds = Dataset.from_pandas(train_df[["text", "behavior_label"]])
test_ds = Dataset.from_pandas(test_df[["text", "behavior_label"]])

model_name = "/workspace/amit/Jahanvi/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("behavior_label", "labels")
test_ds = test_ds.rename_column("behavior_label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    "./exp2_results/final_model",
    output_hidden_states=True
)

# =========================
# TRAIN
# =========================
training_args = TrainingArguments(
    output_dir="./exp2_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

#trainer.train()
trainer.save_model("./exp2_results/final_model")
# =========================
# BEHAVIOR EVAL
# =========================
import torch
torch.cuda.empty_cache()
trainer.model.to("cpu")
trainer.model.eval()

from torch.utils.data import DataLoader

print("Running inference (CPU, streaming)...")

model.to("cpu")
model.eval()

loader = DataLoader(test_ds, batch_size=8)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.extend(probs.numpy())
        all_labels.extend(batch["labels"].numpy())

probs = np.array(all_probs)
y_true = np.array(all_labels)
y_pred = (probs > 0.5).astype(int)


print("\n=== BEHAVIOR CLASSIFICATION ===")
print(classification_report(y_true, y_pred, digits=4))

# =========================
# 🔥 EMBEDDING EXTRACTION
# =========================
def get_embeddings(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=False
            )

            # CLS token embedding
            cls_embed = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_embed.cpu().numpy())

    return np.vstack(all_embeddings)
model.to("cpu")
train_embeddings = get_embeddings(train_ds)
test_embeddings = get_embeddings(test_ds)

# =========================
# 🔥 CENTROIDS (CRITICAL)
# =========================
train_labels = train_df["behavior_label"].values

refusal_centroid = train_embeddings[train_labels == 0].mean(axis=0)
compliance_centroid = train_embeddings[train_labels == 1].mean(axis=0)

# =========================
# 🔥 RISK SCORE
# =========================
def compute_risk(embeddings):
    dist_to_refusal = cosine_distances(embeddings, refusal_centroid.reshape(1, -1)).flatten()
    dist_to_compliance = cosine_distances(embeddings, compliance_centroid.reshape(1, -1)).flatten()

    risk = dist_to_compliance - dist_to_refusal
    return risk

risk_scores = compute_risk(test_embeddings)

# normalize to probability
risk_probs = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())

test_df["risk_score"] = risk_scores
test_df["risk_prob"] = risk_probs
test_df["compliance_prob"] = probs  # from classifier

# =========================
# 🔥 CORRELATION WITH HARM (MAIN RESULT)
# =========================
print("\n=== RISK vs HARM ===")

print("\nTop risky samples:")
print(
    test_df.sort_values("risk_prob", ascending=False)[
        ["response", "final_label", "risk_prob"]
    ].head(10)
)

print("\nLow risk samples:")
print(
    test_df.sort_values("risk_prob", ascending=True)[
        ["response", "final_label", "risk_prob"]
    ].head(10)
)

# =========================
# SAVE
# =========================
test_df.to_csv("exp2_risk_analysis.csv", index=False)

print("\nSaved: exp2_risk_analysis.csv")
