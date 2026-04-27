import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OLLAMA_URL = "http://localhost:11434/api/generate"

# victim model = Gemma
VICTIM_MODEL = "gemma2:2b"

# path to your FINETUNED DeBERTa model
DEBERTA_PATH = "./results/final_comparison/final_model"


# ----------------------------------
# GEMMA RESPONSE GENERATION
# ----------------------------------

def generate_response(prompt):
    payload = {
        "model": VICTIM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        out = res.json()

        if "response" in out:
            return out["response"].strip()

    except Exception as e:
        print(f"Generation error: {e}")

    return None


# ----------------------------------
# LOAD FINETUNED DEBERTA
# ----------------------------------

print("Loading fine-tuned DeBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_PATH)
model.eval()


# ----------------------------------
# DEBERTA PREDICTION
# ----------------------------------

def predict_harmfulness(prompt, response):
    text = f"Prompt: {prompt}\nResponse: {response}"

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, round(confidence, 4)


# ----------------------------------
# MAIN DEMO
# ----------------------------------

def main():
    print("=" * 60)
    print("BTP WORKING SYSTEM DEMO")
    print("=" * 60)
    print("Pipeline: Prompt → Gemma Response → DeBERTa Classifier")
    print("=" * 60)

    prompt = input("\nEnter prompt:\n> ").strip()

    if not prompt:
        print("Prompt cannot be empty.")
        return

    # Step 1: victim model generation
    print("\nGenerating response using Gemma...\n")

    response = generate_response(prompt)

    if response is None:
        print("Failed to generate response from Gemma.")
        return

    print("=" * 60)
    print("GEMMA RESPONSE")
    print("=" * 60)
    print(response)

    # Step 2: classifier prediction
    print("\nRunning fine-tuned DeBERTa classifier...\n")

    pred, conf = predict_harmfulness(prompt, response)

    print("=" * 60)
    print("CLASSIFIER OUTPUT")
    print("=" * 60)

    if pred == 1:
        print(f"Prediction: HARMFUL / UNSAFE")
        print(f"Confidence: {conf}")
    else:
        print(f"Prediction: SAFE / BENIGN")
        print(f"Confidence: {conf}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
