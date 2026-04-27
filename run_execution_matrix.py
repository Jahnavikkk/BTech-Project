import os
import random
import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import ollama

# -------------------------
# CONFIG
# -------------------------

TOTAL_SAMPLES = 1000
RATIO = {
    "arena": 200,
    "advbench": 200,
    "deepset": 300,
    "xtram": 300
}

MODELS = {
    "gemma2b": "gemma:2b",
    "llama3_3b": "llama3.2:3b",
    "qwen_0_5b": "qwen2.5:0.5b"
}

OUTPUT_CSV = "execution_matrix_1000.csv"

# Adjust paths if needed
ARENA_PARQUET = "BTP_FINAL_DATA/chatbot_arena_conversations/chatbot_arena.parquet"
DEEPSET_PATH = "local_datasets/deepset_prompt-injections"
XTRAM_PATH = "local_datasets/xTRam1_safe-guard-prompt-injection"
ADVBENCH_ARROW = "BTP_FINAL_DATA/data-00000-of-00001.arrow"


# -------------------------
# LOAD DATA
# -------------------------

def load_arena(n):
    dataset = load_dataset("parquet", data_files={"train": ARENA_PARQUET})["train"]
    prompts = []

    for row in dataset:
        if (
            row["turn"] == 1
            and row["language"] == "English"
            and row["conversation_a"]
        ):
            text = row["conversation_a"][0]["content"]
            if text and len(text) > 10:
                prompts.append(text)

    sample = random.sample(prompts, min(n, len(prompts)))
    return [{
        "prompt": p,
        "dataset": "arena",
        "dataset_label": "benign"
    } for p in sample]


def load_advbench(n):
    dataset = load_dataset("arrow", data_files={"train": ADVBENCH_ARROW})["train"]
    prompts = [row["prompt"] for row in dataset]
    sample = random.sample(prompts, min(n, len(prompts)))
    return [{
        "prompt": p,
        "dataset": "advbench",
        "dataset_label": "malicious"
    } for p in sample]


def load_deepset(n):
    dataset = load_from_disk(DEEPSET_PATH)
    sample = random.sample(range(len(dataset)), min(n, len(dataset)))

    results = []
    for i in sample:
        row = dataset[i]
        label = "malicious" if row["label"] == 1 else "benign"
        results.append({
            "prompt": row["text"],
            "dataset": "deepset",
            "dataset_label": label
        })
    return results


def load_xtram(n):
    dataset = load_from_disk(XTRAM_PATH)
    sample = random.sample(range(len(dataset)), min(n, len(dataset)))

    results = []
    for i in sample:
        row = dataset[i]
        label = "malicious" if row["label"] == 1 else "benign"
        results.append({
            "prompt": row["text"],
            "dataset": "xtram",
            "dataset_label": label
        })
    return results


# -------------------------
# MODEL EXECUTION
# -------------------------

def run_model(prompt, model_tag):
    try:
        response = ollama.generate(
            model=model_tag,
            prompt=prompt,
            stream=False
        )
        return response["response"].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    print("Loading datasets...")

    data = []
    data.extend(load_arena(RATIO["arena"]))
    data.extend(load_advbench(RATIO["advbench"]))
    data.extend(load_deepset(RATIO["deepset"]))
    data.extend(load_xtram(RATIO["xtram"]))

    random.shuffle(data)

    print(f"Total prompts loaded: {len(data)}")

    rows = []

    for item in tqdm(data, desc="Running executions"):
        row = {
            "prompt": item["prompt"],
            "dataset": item["dataset"],
            "dataset_label": item["dataset_label"]
        }

        for model_name, model_tag in MODELS.items():
            print(f"Running {model_name}...")
            response = run_model(item["prompt"], model_tag)
            row[f"{model_name}_response"] = response

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nExecution complete. Saved to {OUTPUT_CSV}")
