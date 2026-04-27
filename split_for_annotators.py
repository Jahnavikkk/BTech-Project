import pandas as pd
import random
import os

INPUT_FILE = "annotation_master_3000.csv"

df = pd.read_csv(INPUT_FILE)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 3 chunks of 1000
chunks = [df[i:i+1000] for i in range(0, 3000, 1000)]

for i, chunk in enumerate(chunks):
    base_name = f"annotator_set_{i+1}"
    
    # Create duplicate copies
    chunk.to_csv(f"{base_name}_A.csv", index=False)
    chunk.to_csv(f"{base_name}_B.csv", index=False)

print("Created 6 annotation files (3 sets × 2 duplicates)")
