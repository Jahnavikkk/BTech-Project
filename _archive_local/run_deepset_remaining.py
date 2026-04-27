import pandas as pd
import random
import ollama
from datasets import load_dataset
from tqdm import tqdm
import time

OUTPUT_FILE = "execution_matrix_1000.csv"
REQUIRED_TOTAL = 300

MODELS = {
    "gemma2b": "gemma:2b",
    "llama3_3b": "llama3.2:3b",
    "qwen_0_5b": "qwen2.5:0.5b"
}

def load_existing_prompts():
    df = pd.read_csv(OUTPUT_FILE)
    used = df[df["dataset"]=="deepset"]["prompt"].unique()
    return set(used)

def generate_response(model_tag, prompt):
    try:
        response = ollama.generate(model=model_tag, prompt=prompt, stream=False)
        return response["response"].replace("\n"," ").strip()
    except Exception:
        return "GENERATION_ERROR"

if __name__ == "__main__":
    print("Loading existing file...")
    df_existing = pd.read_csv(OUTPUT_FILE)
    used_prompts = load_existing_prompts()
    already_count = len(used_prompts)

    needed = REQUIRED_TOTAL - already_count
    print(f"Already have {already_count}, need {needed} more.")

    print("Loading DeepSet train split...")
    train_ds = load_dataset("deepset/prompt-injections", split="train")

    candidates = []
    for item in train_ds:
        if item["text"] not in used_prompts:
            candidates.append(item)

    random.shuffle(candidates)
    selected = candidates[:needed]

    print(f"Selected {len(selected)} new prompts.")

    new_rows = []

    for item in tqdm(selected, desc="Running remaining DeepSet prompts"):
        prompt = item["text"]
        label = "malicious" if item["label"] == 1 else "benign"

        row = {
            "prompt": prompt,
            "dataset": "deepset",
            "dataset_label": label
        }

        for name, tag in MODELS.items():
            print(f"Running {name}...")
            row[f"{name}_response"] = generate_response(tag, prompt)
            time.sleep(1)

        new_rows.append(row)

    print("Appending to CSV...")
    pd.DataFrame(new_rows).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=False,
        index=False
    )

    print("Done.")

