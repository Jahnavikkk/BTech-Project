from transformers import pipeline
import os

# The model we need to download
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
# The local directory where we will save it
SAVE_DIR = "./local_classifier"

print(f"--- Attempting to download {MODEL_NAME} ---")

try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Download the model by creating a pipeline and saving its components
    classifier_pipeline = pipeline("text-classification", model=MODEL_NAME)
    classifier_pipeline.save_pretrained(SAVE_DIR)
    
    print(f"\n[+] Success! Model saved to {SAVE_DIR}")

except Exception as e:
    print(f"\n[!] FAILED to download model. Error: {e}")
    print("Please ensure you have run the lynx login and set the HTTPS_PROXY variable.")
