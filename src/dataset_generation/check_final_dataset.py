import pandas as pd

df = pd.read_csv("final_training_dataset_3000_clean.csv")
df= df.drop('is_reconstructed',axis=1)
print("===== BASIC INFO =====")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()

# -----------------------------
# Fix prompt_id
# -----------------------------
df["prompt_clean"] = df["prompt"].str.strip()

prompt_to_id = {p: i for i, p in enumerate(df["prompt_clean"].unique())}
df["prompt_id_fixed"] = df["prompt_clean"].map(prompt_to_id)

# -----------------------------
# 🔥 THIS IS THE MOST IMPORTANT CHECK
# -----------------------------
print("===== GROUP SIZE USING CLEAN PROMPT =====")
group_sizes = df.groupby("prompt_id_fixed").size().value_counts()
print(group_sizes)
print()

# -----------------------------
# Row count check
# -----------------------------
print("===== ROW COUNT CHECK =====")
if len(df) != 3000:
    print("❌ ERROR: Expected 3000 rows, got", len(df))
else:
    print("✅ Row count correct (3000)")
print()

# -----------------------------
# Null check
# -----------------------------
print("===== NULL CHECK =====")
nulls = df.isnull().sum()
print(nulls)

if nulls.sum() > 0:
    print("❌ ERROR: Null values present")
else:
    print("✅ No nulls")
print()
