import pandas as pd
import re

INPUT_FILE = "execution_matrix_1000.csv"
OUTPUT_FILE = "execution_matrix_cleaned.csv"

def is_english(text):
    if not isinstance(text, str):
        return False
    ascii_chars = sum(c.isascii() for c in text)
    ratio = ascii_chars / max(len(text), 1)
    return ratio > 0.8

print("Loading CSV...")
df = pd.read_csv(INPUT_FILE)

print("Original size:", len(df))

# Identify non-English DeepSet prompts
mask_non_english = (
    (df["dataset"] == "deepset") &
    (~df["prompt"].apply(is_english))
)

removed_count = mask_non_english.sum()
print("Removing non-English DeepSet prompts:", removed_count)

df_cleaned = df[~mask_non_english].copy()

print("Cleaned size:", len(df_cleaned))

df_cleaned.to_csv(OUTPUT_FILE, index=False)
print("Saved cleaned file.")
