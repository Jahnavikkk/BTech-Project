from datasets import load_dataset
import os

DATA_DIR = "./BTP_FINAL_DATA"
ARENA_FILE = os.path.join(DATA_DIR, "chatbot_arena_conversations", "chatbot_arena.parquet")

try:
    print(f"Loading {ARENA_FILE}...")
    dataset = load_dataset("parquet", data_files={'train': ARENA_FILE})['train']
    
    print("\n--- DATASET LOADED ---")
    print("\n[+] Column Names:")
    print(dataset.column_names)
    
    print("\n[+] First Row Example:")
    print(dataset[0])
    
except Exception as e:
    print(f"\nError inspecting file: {e}")
