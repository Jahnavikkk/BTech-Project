from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with your real Hugging Face token
import os
token = os.getenv("HF_TOKEN")

# Authenticate
login(token=HF_TOKEN)

# Model name
model_name = "meta-llama/Prompt-Guard-86M"

# Download tokenizer + model
print("⏳ Downloading model... this may take a few minutes")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HF_TOKEN)

print("✅ Model downloaded and ready to use!")




