import pandas as pd
import ollama
from tqdm import tqdm
import os
import sys
import random
from datasets import load_dataset, load_from_disk
import json

# --- 1. CONFIGURATION ---

# Your committee of 10 diverse models
COMMITTEE_MODELS = {
    "gemma2:2b": "gemma2:2b",
    "gemma:2b": "gemma:2b",
    "llama3.2:3b": "llama3.2:3b",
    "mistrallite": "mistrallite:latest",
    "openchat": "openchat:latest",
    "qwen2.5:0.5b": "qwen2.5:0.5b",
    "qwen:0.5b": "qwen:0.5b",
    "smollm2:360m": "smollm2:360m",
    "tinydolphin": "tinydolphin:latest",
    "tinyllama": "tinyllama:latest"
}

# All data is now in one place
DATA_DIR = "./BTP_FINAL_DATA"
ADVBENCH_FILE = os.path.join(DATA_DIR, "data-00000-of-00001.arrow")
DEEPSET_PATH = os.path.join(DATA_DIR, "deepset_prompt-injections")
ARENA_FILE = os.path.join(DATA_DIR, "chatbot_arena_conversations", "chatbot_arena.parquet") # Correct path

OUTPUT_CSV = "committee_generations_balanced_v3.csv" # The raw output file

# --- 2. HELPER FUNCTIONS ---

def load_all_prompts():
    """Loads a balanced set of malicious and benign prompts."""
    prompts_list = []
    
    # 1. Load Benign Prompts (Sample 750)
    try:
        dataset = load_dataset("parquet", data_files={'train': ARENA_FILE})['train']
        
        first_turns = []
        for conv in dataset:
            # --- THIS IS THE CORRECTED LOGIC ---
            if (conv['turn'] == 1 and 
                conv['language'] == 'English' and
                conv['conversation_a'] and # Check if list is not empty
                len(conv['conversation_a']) > 0): 
                
                # Correctly extract the prompt text
                prompt_text = conv['conversation_a'][0]['content']
                
                # Correctly check for toxicity
                is_toxic = False
                if conv['toxic_chat_tag']:
                    if conv['toxic_chat_tag'].get('roberta-large', {}).get('flagged', False):
                        is_toxic = True
                    if conv['toxic_chat_tag'].get('t5-large', {}).get('flagged', False):
                        is_toxic = True
                
                # Final filter
                if prompt_text and not is_toxic and len(prompt_text) > 10:
                    first_turns.append(prompt_text)
            # --- END OF CORRECTION ---
        
        sample_size = min(len(first_turns), 750)
        benign_sample = random.sample(first_turns, sample_size)
        
        for prompt_text in benign_sample:
            prompts_list.append({
                "prompt_id": f"arena-{len(prompts_list)}", 
                "prompt_text": prompt_text, 
                "true_label": "benign"
            })
        print(f"Loaded {len(benign_sample)} benign prompts from Chatbot Arena.")
    except Exception as e:
        print(f"Error loading Chatbot Arena: {e}. Check path: {ARENA_FILE}")

    # 2. Load Malicious Prompts
    # 2a. Load AdvBench (All 520)
    try:
        dataset = load_dataset("arrow", data_files={'train': ADVBENCH_FILE})['train']
        for item in dataset:
            prompts_list.append({
                "prompt_id": f"advbench-{len(prompts_list)}", 
                "prompt_text": item['prompt'], 
                "true_label": "malicious"
            })
        print(f"Loaded {len(dataset)} malicious prompts from AdvBench.")
    except Exception as e:
        print(f"Error loading AdvBench: {e}. Check path: {ADVBENCH_FILE}")

    # 2b. Load DeepSet (All malicious)
    try:
        dataset = load_from_disk(DEEPSET_PATH)
        malicious_dataset = dataset.filter(lambda x: x['label'] == 1)
        for item in malicious_dataset:
            prompts_list.append({
                "prompt_id": f"deepset-{len(prompts_list)}", 
                "prompt_text": item['text'], 
                "true_label": "malicious"
            })
        print(f"Loaded {len(malicious_dataset)} malicious prompts from DeepSet.")
    except Exception as e:
        print(f"Error loading DeepSet: {e}. Check path: {DEEPSET_PATH}")
            
    print(f"Total prompts loaded: {len(prompts_list)}")
    random.shuffle(prompts_list)
    return prompts_list

def get_model_response(prompt, model_tag):
    """Generates a response using the specified model."""
    try:
        response = ollama.generate(model=model_tag, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        print(f"Error generating response for {model_tag}: {e}")
        return "GENERATION_ERROR"

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    prompts = load_all_prompts()
    if not prompts:
        print("No prompts loaded. Exiting.")
        sys.exit(1)
        
    csv_headers = ["prompt_id", "prompt_text", "true_label", "committee_model", "response_text"]
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        # Use pandas to write header only if file is new
        if not file_exists:
            pd.DataFrame(columns=csv_headers).to_csv(f, index=False, header=True)
        
        processed_prompt_ids = set()
        if file_exists:
            try:
                df_existing = pd.read_csv(OUTPUT_CSV)
                if not df_existing.empty:
                    prompt_counts = df_existing['prompt_id'].value_counts()
                    processed_prompt_ids = set(prompt_counts[prompt_counts == len(COMMITTEE_MODELS)].index)
                    print(f"Resuming. Found {len(processed_prompt_ids)} prompts already fully processed.")
            except Exception as e:
                print(f"Could not read existing CSV, starting fresh: {e}")

        for item in tqdm(prompts, desc="Processing Prompts"):
            if item['prompt_id'] in processed_prompt_ids:
                continue 

            row_data = []
            for model_name, model_tag in COMMITTEE_MODELS.items():
                response = get_model_response(item['prompt_text'], model_tag)
                row_data.append({
                    "prompt_id": item['prompt_id'],
                    "prompt_text": item['prompt_text'],
                    "true_label": item['true_label'],
                    "committee_model": model_name,
                    "response_text": response
                })
            
            # Write all 10 rows for this prompt
            pd.DataFrame(row_data).to_csv(f, index=False, header=False)

    print(f"\n--- GENERATION COMPLETE ---")
    print(f"All responses saved to {OUTPUT_CSV}")
