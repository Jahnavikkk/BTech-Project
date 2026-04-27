import pandas as pd
import random
import ollama
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import time

INPUT_FILE = "execution_matrix_cleaned.csv"
FINAL_FILE = "execution_matrix_final.csv"
TARGET_DEEPSET = 300

MODELS = {
    "gemma2b": "gemma:2b",
    "llama3_3b": "llama3.2:3b",
    "qwen_0_5b": "qwen2.5:0.5b"
}

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def generate_response(model_tag, prompt):
    try:
        response = ollama.generate(model=model_tag, prompt=prompt, stream=False)
        return response["response"].replace("\n"," ").strip()
    except:
        return "GENERATION_ERROR"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE)

current_deepset = df[df["dataset"]=="deepset"]
current_count = len(current_deepset)

needed = TARGET_DEEPSET - current_count
print("Current DeepSet:", current_count)
print("Need to refill:", needed)

used_prompts = set(df["prompt"].tolist())

print("Loading DeepSet train...")
train_ds = load_dataset("deepset/prompt-injections", split="train")

candidates = []
for item in train_ds:
    text = item["text"]
    if text not in used_prompts and is_english(text):
        candidates.append(item)

random.shuffle(candidates)
selected = candidates[:needed]

print("Selected:", len(selected))

new_rows = []

for item in tqdm(selected):
    prompt = item["text"]
    label = "malicious" if item["label"] == 1 else "benign"

    row = {
        "prompt": prompt,
        "dataset": "deepset",
        "dataset_label": label
    }

    for name, tag in MODELS.items():
        row[f"{name}_response"] = generate_response(tag, prompt)
        time.sleep(1)

    new_rows.append(row)

df_final = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df_final.to_csv(FINAL_FILE, index=False)

print("Final size:", len(df_final))
print("Saved to execution_matrix_final.csv")
