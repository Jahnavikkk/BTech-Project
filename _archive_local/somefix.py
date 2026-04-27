import pandas as pd

# -----------------------------
# LOAD FILES
# -----------------------------
df_final = pd.read_csv("final_training_dataset_3000_clean.csv")
df_combined = pd.read_csv("annotations_combined.csv")

print("Initial final dataset shape:", df_final.shape)
print("Combined dataset shape:", df_combined.shape)

# -----------------------------
# FIND PROMPT LABEL COLUMN
# -----------------------------
print("\nColumns in annotations_combined:")
print(df_combined.columns)

# 🔥 CHANGE THIS IF NEEDED
PROMPT_LABEL_COL = "dataset_label"   # or "prompt_label"

# -----------------------------
# NORMALIZE PROMPTS (CRITICAL)
# -----------------------------
df_final["prompt_clean"] = df_final["prompt"].astype(str).str.strip()
df_combined["prompt_clean"] = df_combined["prompt"].astype(str).str.strip()

# -----------------------------
# EXTRACT PROMPT-LEVEL LABELS
# -----------------------------
prompt_labels = (
    df_combined[["prompt_clean", PROMPT_LABEL_COL]]
    .drop_duplicates()
)

print("\nUnique prompts in labels:", len(prompt_labels))

# -----------------------------
# MERGE
# -----------------------------
df_merged = df_final.merge(
    prompt_labels,
    on="prompt_clean",
    how="left"
)

# -----------------------------
# SANITY CHECK
# -----------------------------
missing = df_merged[PROMPT_LABEL_COL].isnull().sum()

print("\nMissing prompt labels after merge:", missing)

if missing > 0:
    print("❌ WARNING: Some prompts did not match. Check cleaning.")
else:
    print("✅ All prompts matched successfully")

# -----------------------------
# CLEANUP
# -----------------------------
df_merged = df_merged.drop(columns=["prompt_clean","is_reconstructed"])

# -----------------------------
# FINAL CHECK
# -----------------------------
print("\n===== FINAL CHECK =====")
print("Shape:", df_merged.shape)
print("Columns:", df_merged.columns.tolist())

print("\nLabel distribution:")
print(df_merged[PROMPT_LABEL_COL].value_counts())

# -----------------------------
# OVERWRITE ORIGINAL FILE
# -----------------------------
df_merged.to_csv("final_training_dataset_3000_clean.csv", index=False)

print("\n✅ File overwritten with prompt labels added")
