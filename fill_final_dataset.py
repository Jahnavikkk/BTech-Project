import pandas as pd

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv("final_dataset_3000_ollama.csv")

print("Initial rows:", len(df))

# -----------------------------
# STEP 1: normalize prompt
# -----------------------------
df["prompt_norm"] = df["prompt"].astype(str).str.strip()

# -----------------------------
# STEP 2: get unique prompts
# -----------------------------
unique_prompts = df["prompt_norm"].unique()

print("Total prompts before:", len(unique_prompts))

# -----------------------------
# STEP 3: pick first 1000 prompts
# -----------------------------
selected_prompts = unique_prompts[:1000]

# -----------------------------
# STEP 4: filter dataset
# -----------------------------
df = df[df["prompt_norm"].isin(selected_prompts)]

# -----------------------------
# STEP 5: rebuild prompt_id
# -----------------------------
prompt_map = {p: i for i, p in enumerate(selected_prompts)}
df["prompt_id"] = df["prompt_norm"].map(prompt_map)

# -----------------------------
# FINAL CHECK
# -----------------------------
print("\n===== FINAL CHECK =====")
print("Total rows:", len(df))
print("Unique prompts:", df["prompt_id"].nunique())

print("\nGroup sizes:")
print(df.groupby("prompt_id").size().value_counts())

# -----------------------------
# SAVE
# -----------------------------
df = df[["prompt_id", "prompt", "dataset", "model_name", "response", "final_label", "is_reconstructed"]]

df.to_csv("final_training_dataset_3000_clean.csv", index=False)

print("\n✅ Saved: final_training_dataset_3000_clean.csv")
