from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/workspace/amit/Jahanvi/promptguard"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("loaded successfully")
