import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

INPUT_FILE = "execution_matrix_1000.csv"
OUTPUT_FILE = "execution_matrix_cleaned.csv"

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

print("Loading CSV...")
df = pd.read_csv(INPUT_FILE)

print("Original size:", len(df))

mask_remove = (
    (df["dataset"] == "deepset") &
    (~df["prompt"].apply(is_english))
)

removed = mask_remove.sum()
print("Removing non-English DeepSet prompts:", removed)

df_cleaned = df[~mask_remove].copy()

print("Cleaned size:", len(df_cleaned))

df_cleaned.to_csv(OUTPUT_FILE, index=False)
print("Saved cleaned file.")
