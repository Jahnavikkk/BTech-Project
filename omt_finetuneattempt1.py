import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset

# =========================
# CONFIG
# =========================
MODEL_LIST = [
    "/workspace/amit/Jahanvi/deberta-v3-base",
    "/workspace/amit/Jahanvi/roberta-large",
    "/workspace/amit/Jahanvi/promptguard"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_training_dataset_3000_clean.csv")

# =========================
# FORMAT INPUT
# =========================
def format_input(row):
    return f"Prompt: {row['prompt']}\nModel: {row['model_name']}\nResponse: {row['response']}"

df['text'] = df.apply(format_input, axis=1)

# =========================
# SPLIT BY PROMPT (CRITICAL)
# =========================
unique_prompts = df['prompt_id'].unique()

train_ids = set(unique_prompts[:800])
test_ids = set(unique_prompts[800:])

train_df = df[df['prompt_id'].isin(train_ids)]
test_df = df[df['prompt_id'].isin(test_ids)]
# =========================
# OVERSAMPLING (CRITICAL)
# =========================
harmful_df = train_df[train_df['final_label'] == 1]
safe_df = train_df[train_df['final_label'] == 0]

print("Before balancing:")
print(train_df['final_label'].value_counts())

# Upsample harmful class
harmful_upsampled = harmful_df.sample(
    n=len(safe_df)//5,   # 1:5 ratio (safe:harmful)
    replace=True,
    random_state=42
)

train_df = pd.concat([safe_df, harmful_upsampled]).sample(frac=1, random_state=42)

print("After balancing:")
print(train_df['final_label'].value_counts())
# =========================
# DATASETS
# =========================
def prepare_dataset(dataframe):
    dataset = Dataset.from_pandas(dataframe[['text', 'final_label']])
    dataset = dataset.rename_column("final_label", "labels")
    return dataset

train_dataset = prepare_dataset(train_df)
test_dataset = prepare_dataset(test_df)

# =========================
# FOCAL LOSS
# =========================
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# =========================
# TRAINER WITH FOCAL LOSS
# =========================
class FocalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================
# TOKENIZATION
# =========================
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(
            example['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    return dataset.map(tokenize_fn, batched=True)

# =========================
# TRAIN LOOP
# =========================
results = []

for model_name in MODEL_LIST:
    print(f"\n\n========== TRAINING {model_name} ==========\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = tokenize_dataset(train_dataset, tokenizer)
    test_ds = tokenize_dataset(test_dataset, tokenizer)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.split('/')[-1]}",
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_steps=50,
        report_to="none"
    )

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    # =========================
    # EVALUATION
    # =========================
# =========================
# EVALUATION WITH THRESHOLD
# =========================
preds = trainer.predict(test_ds)

logits = preds.predictions
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

for THRESHOLD in [0.1, 0.2, 0.3, 0.4]:

	y_pred = (probs > THRESHOLD).astype(int)
	y_true = preds.label_ids

from sklearn.metrics import classification_report

print(f"\nThreshold used: {THRESHOLD}")
print(classification_report(y_true, y_pred, digits=4))  
print("\nSample probabilities for class 1:")
print(probs[:20])
# =========================
# FINAL RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("final_model_comparison.csv", index=False)

print("\n\n===== FINAL COMPARISON =====")
print(results_df)
