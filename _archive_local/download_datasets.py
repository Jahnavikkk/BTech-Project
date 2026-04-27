import os
from datasets import load_dataset

# Configuration for the datasets we need
DATASETS_CONFIG = [
    {"name": "deepset/prompt-injections", "split": "test"},
    {"name": "xTRam1/safe-guard-prompt-injection", "split": "train"},
    {"name": "llm-attacks/AdvBench", "split": "train"}
]

# Directory to save the local copies
SAVE_DIR = "./local_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

print("--- Starting Dataset Download ---")

for config in DATASETS_CONFIG:
    try:
        dataset_name = config['name']
        local_path = os.path.join(SAVE_DIR, dataset_name.replace("/", "_")) # Create a valid folder name
        
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset(dataset_name, split=config['split'])
        
        print(f"Saving to disk at {local_path}...")
        dataset.save_to_disk(local_path)
        
        print(f"Successfully saved {dataset_name}.\n")
        
    except Exception as e:
        print(f"!!! FAILED to download {dataset_name}. Error: {e}\n")

print("--- Download process complete. ---")

