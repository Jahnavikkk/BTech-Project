from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
ds = load_dataset("qualifire/prompt-injections-benchmark", split="test[:1000]")

# Load model
model_name = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Evaluate
texts = ds["text"]
labels = [0 if l == "benign" else 1 for l in ds["label"]]

scores = []
preds = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    preds.append(pred)

# Metrics
acc = accuracy_score(labels, preds)
print(f"Accuracy: {acc:.3f}")
print("Classification Report:")
print(classification_report(labels, preds, target_names=["benign", "jailbreak"]))
