import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("final_training_dataset_3000_clean.csv")

print("Dataset shape:", df.shape)
print()

# -----------------------------
# BASIC DISTRIBUTION
# -----------------------------
print("===== LABEL DISTRIBUTION =====")
label_counts = df["final_label"].value_counts()
print(label_counts)
print(label_counts / len(df))
print()

# Plot
plt.figure()
label_counts.plot(kind="bar")
plt.title("Final Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("label_distribution.png")
plt.close()

---

# -----------------------------
# PROMPT vs EXECUTION
# -----------------------------
if "prompt_label" in df.columns:
    print("===== PROMPT vs EXECUTION =====")
    cross = pd.crosstab(df["prompt_label"], df["final_label"])
    print(cross)
    print()

    # % execution given harmful prompt
    harmful_exec_rate = cross.loc[1, 1] / cross.loc[1].sum()
    print(f"Execution rate for harmful prompts: {harmful_exec_rate:.4f}")
    print()

---

# -----------------------------
# MODEL-WISE BEHAVIOR
# -----------------------------
print("===== MODEL-WISE EXECUTION RATE =====")

model_stats = df.groupby("model_name")["final_label"].agg(["mean", "count"])
model_stats = model_stats.rename(columns={"mean": "execution_rate"})
print(model_stats)
print()

# Plot
plt.figure()
model_stats["execution_rate"].plot(kind="bar")
plt.title("Execution Rate by Model")
plt.ylabel("Execution Rate")
plt.savefig("model_execution.png")
plt.close()

---

# -----------------------------
# MODEL AGREEMENT
# -----------------------------
print("===== MODEL AGREEMENT =====")

agreement = df.groupby("prompt_id")["final_label"].nunique().value_counts()
print(agreement)
print()

# Interpretation counts
all_same = (df.groupby("prompt_id")["final_label"].nunique() == 1).sum()
mixed = (df.groupby("prompt_id")["final_label"].nunique() > 1).sum()

print(f"All models agree: {all_same}")
print(f"Mixed behavior: {mixed}")
print()

---

# -----------------------------
# DATASET-WISE EXECUTION
# -----------------------------
print("===== DATASET-WISE EXECUTION RATE =====")

dataset_stats = df.groupby("dataset")["final_label"].mean().sort_values(ascending=False)
print(dataset_stats)
print()

# Plot
plt.figure()
dataset_stats.plot(kind="bar")
plt.title("Execution Rate by Dataset")
plt.ylabel("Execution Rate")
plt.savefig("dataset_execution.png")
plt.close()

---

# -----------------------------
# PROMPT-LEVEL RISK ANALYSIS
# -----------------------------
print("===== PROMPT RISK DISTRIBUTION =====")

prompt_sum = df.groupby("prompt_id")["final_label"].sum()

risk_counts = prompt_sum.value_counts().sort_index()
print(risk_counts)
print()

"""
0 → all models safe  
1 → one model fails  
2 → two models fail  
3 → all models fail  
"""

plt.figure()
risk_counts.plot(kind="bar")
plt.title("Prompt Risk Levels")
plt.xlabel("Number of models executing harmful intent")
plt.ylabel("Count")
plt.savefig("prompt_risk.png")
plt.close()

---

# -----------------------------
# INSIGHT SUMMARY (AUTO PRINT)
# -----------------------------
print("===== AUTO INSIGHTS =====")

total = len(df)
malicious = label_counts.get(1, 0)

print(f"- Only {malicious/total:.2%} responses are malicious → strong imbalance")

print(f"- Most prompts are safe across all models: {risk_counts.get(0,0)}")

if risk_counts.get(3,0) > 0:
    print(f"- {risk_counts.get(3)} prompts cause ALL models to fail → critical cases")

if mixed > 0:
    print(f"- {mixed} prompts show disagreement → model safety is inconsistent")

top_model = model_stats["execution_rate"].idxmax()
worst_model = model_stats["execution_rate"].idxmax()

print(f"- Model with highest execution rate: {worst_model}")

hard_dataset = dataset_stats.idxmax()
print(f"- Most adversarial dataset: {hard_dataset}")

print("\n✅ Analysis complete. Plots saved.")
