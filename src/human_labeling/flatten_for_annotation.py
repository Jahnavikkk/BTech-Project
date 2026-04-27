import pandas as pd
import random
import os

INPUT_FILE = "execution_matrix_final.csv"
OUTPUT_FILE = "annotation_master_3000.csv"

print("Loading final dataset...")
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# Flatten into 3 rows per prompt
rows = []

for _, row in df.iterrows():
    for model_col in ["gemma2b_response", "llama3_3b_response", "qwen_0_5b_response"]:
        rows.append({
            "prompt": row["prompt"],
            "dataset": row["dataset"],
            "dataset_label": row["dataset_label"],
            "model_name": model_col.replace("_response",""),
            "model_response": row[model_col],
            "human_label(benign=0,malicious=1)": ""  # blank for annotators
        })

df_flat = pd.DataFrame(rows)

print("Flattened rows:", len(df_flat))  # should be 3000

df_flat.to_csv(OUTPUT_FILE, index=False)
print("Saved master annotation file:", OUTPUT_FILE)

