import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_training_dataset_3000_clean.csv")

# =========================
# 2. PREPROCESSING
# =========================
df['prompt_label'] = (df['dataset_label'].str.lower() == 'malicious').astype(int)
df['mismatch'] = df['prompt_label'] != df['final_label']

# Text features for later
df['prompt_length'] = df['prompt'].astype(str).apply(len)
df['response_length'] = df['response'].astype(str).apply(len)

# =========================
# 3. STEP 1 — MISMATCH ANALYSIS
# =========================
print("\n=== STEP 1: MISMATCH ===")

mismatch_rate = df['mismatch'].mean()
print(f"Mismatch Rate: {mismatch_rate:.4f}")

# Confusion Matrix
cm = confusion_matrix(df['prompt_label'], df['final_label'])
cm_df = pd.DataFrame(
    cm,
    index=["Prompt: Benign", "Prompt: Malicious"],
    columns=["Execution: Benign", "Execution: Malicious"]
)
print("\nConfusion Matrix:\n", cm_df)

# Save
cm_df.to_csv("confusion_matrix.csv")

# Plot
plt.figure()
sns.heatmap(cm_df, annot=True, fmt="d")
plt.title("Prompt vs Execution Labels")
plt.savefig("confusion_matrix.png")
plt.close()

# =========================
# 4. STEP 2 — MODEL-WISE COMPLIANCE
# =========================
print("\n=== STEP 2: MODEL COMPLIANCE ===")

model_compliance = df.groupby("model_name")['final_label'].mean().sort_values(ascending=False)
print(model_compliance)

model_compliance.to_csv("model_compliance.csv")

plt.figure()
model_compliance.plot(kind='bar')
plt.title("Model-wise Harmful Response Rate")
plt.ylabel("Harmful Rate")
plt.xticks(rotation=45)
plt.savefig("model_compliance.png")
plt.close()

# =========================
# 5. STEP 3 — MODEL AGREEMENT
# =========================
print("\n=== STEP 3: MODEL AGREEMENT ===")

# Count unique labels per prompt
agreement = df.groupby("prompt_id")['final_label'].nunique()

agreement_df = agreement.value_counts().rename({
    1: "All Agree",
    2: "Disagree"
})

print(agreement_df)

agreement_df.to_csv("model_agreement_counts.csv")

# Plot
plt.figure()
agreement_df.plot(kind='bar')
plt.title("Model Agreement Distribution")
plt.ylabel("Number of Prompts")
plt.savefig("model_agreement.png")
plt.close()

# =========================
# 6. STEP 3.1 — DISAGREEMENT ANALYSIS
# =========================
print("\n=== STEP 3.1: DISAGREEMENT ANALYSIS ===")

# Identify disagreement prompts
disagree_prompts = agreement[agreement > 1].index
agree_prompts = agreement[agreement == 1].index

df['agreement_group'] = df['prompt_id'].apply(
    lambda x: "Disagree" if x in disagree_prompts else "Agree"
)

# --- Compare lengths ---
length_stats = df.groupby("agreement_group")[['prompt_length', 'response_length']].mean()
print("\nLength Comparison:\n", length_stats)

length_stats.to_csv("agreement_length_stats.csv")

# Plot
plt.figure()
sns.boxplot(data=df, x='agreement_group', y='prompt_length')
plt.title("Prompt Length vs Agreement")
plt.savefig("prompt_length_agreement.png")
plt.close()

# --- Dataset distribution in disagreement ---
dataset_disagreement = (
    df[df['agreement_group'] == "Disagree"]
    .groupby("dataset")
    .size()
    / df.groupby("dataset").size()
)

print("\nDataset-wise Disagreement Rate:\n", dataset_disagreement)

dataset_disagreement.to_csv("dataset_disagreement_rate.csv")

plt.figure()
dataset_disagreement.plot(kind='bar')
plt.title("Dataset-wise Disagreement Rate")
plt.ylabel("Rate")
plt.xticks(rotation=45)
plt.savefig("dataset_disagreement.png")
plt.close()

# =========================
# 7. STEP 4 — DATASET-WISE EXECUTION BEHAVIOR
# =========================
print("\n=== STEP 4: DATASET BEHAVIOR ===")

dataset_behavior = df.groupby("dataset")['final_label'].mean().sort_values(ascending=False)
print(dataset_behavior)

dataset_behavior.to_csv("dataset_behavior.csv")

plt.figure()
dataset_behavior.plot(kind='bar')
plt.title("Dataset-wise Harmful Execution Rate")
plt.ylabel("Harmful Rate")
plt.xticks(rotation=45)
plt.savefig("dataset_behavior.png")
plt.close()

# =========================
# 8. EXTRA — LABEL IMBALANCE
# =========================
print("\n=== EXTRA: LABEL DISTRIBUTION ===")

label_dist = df['final_label'].value_counts(normalize=True)
print(label_dist)

label_dist.to_csv("label_distribution.csv")

plt.figure()
label_dist.plot(kind='bar')
plt.title("Label Distribution")
plt.ylabel("Proportion")
plt.savefig("label_distribution.png")
plt.close()

# =========================
# 9. SAVE FINAL ENRICHED DATA
# =========================
df.to_csv("dataset_with_analysis_features.csv", index=False)

print("\n ALL ANALYSIS COMPLETED — FILES SAVED")
