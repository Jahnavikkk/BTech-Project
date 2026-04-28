# demo.py
# Pipeline:
# User Prompt
# -> choose victim model (Gemma / Qwen / Llama)
# -> victim model generates response
# -> response passed to:
#       1. Fine-tuned DeBERTa (model agnostic)
#       2. Fine-tuned PromptGuard (model agnostic)
# -> final verdict shown

import requests
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# CONFIG
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_OPTIONS = {
    "1": "gemma2:2b",
    "2": "qwen2.5:14b",
    "3": "llama3.1:8b"
}

# CHANGE THESE TO YOUR ACTUAL SAVED PATHS
DEBERTA_PATH = "./results/final_comparison/final_model"
PROMPTGUARD_PATH = "./results/final_comparison/promptguard_final_model"

MAX_LEN = 512

# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading DeBERTa...")

deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
deberta_model = AutoModelForSequenceClassification.from_pretrained(
    DEBERTA_PATH
)
deberta_model.eval()

print("DeBERTa loaded.")

print("\nLoading PromptGuard...")

pg_tokenizer = AutoTokenizer.from_pretrained(PROMPTGUARD_PATH)
pg_model = AutoModelForSequenceClassification.from_pretrained(
    PROMPTGUARD_PATH
)
pg_model.eval()

print("PromptGuard loaded.\n")


# =========================================================
# VICTIM MODEL RESPONSE
# =========================================================

def generate_response(prompt, victim_model):
    payload = {
        "model": victim_model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        result = response.json()

        if "response" in result:
            return result["response"].strip()

    except Exception as e:
        print(f"Generation error: {e}")

    return None


# =========================================================
# CLASSIFIER PREDICTION
# =========================================================

def classify(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, round(confidence, 4)


def label_to_text(label):
    return "HARMFUL / UNSAFE" if label == 1 else "SAFE / BENIGN"


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("BTP WORKING SYSTEM DEMO")
    print("=" * 70)
    print("Pipeline:")
    print("Prompt -> Victim LLM -> Response -> DeBERTa + PromptGuard")
    print("=" * 70)

    print("\nSelect victim model:\n")
    print("1. Gemma")
    print("2. Qwen")
    print("3. Llama\n")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice not in MODEL_OPTIONS:
        print("Invalid choice.")
        return

    victim_model = MODEL_OPTIONS[choice]

    prompt = input("\nEnter user prompt:\n> ").strip()

    if not prompt:
        print("Prompt cannot be empty.")
        return

    print(f"\nGenerating response using {victim_model}...\n")

    response = generate_response(prompt, victim_model)

    if response is None:
        print("Failed to generate model response.")
        return

    print("=" * 70)
    print("VICTIM MODEL RESPONSE")
    print("=" * 70)
    print(response)

    combined_text = (
        f"Prompt: {prompt}\n"
        f"Response: {response}"
    )

    print("\nRunning DeBERTa classifier...")
    deberta_pred, deberta_conf = classify(
        combined_text,
        deberta_tokenizer,
        deberta_model
    )

    print("Running PromptGuard classifier...")
    pg_pred, pg_conf = classify(
        combined_text,
        pg_tokenizer,
        pg_model
    )

    print("\n" + "=" * 70)
    print("CLASSIFIER RESULTS")
    print("=" * 70)

    print(
        f"DeBERTa      : {label_to_text(deberta_pred)} "
        f"(confidence={deberta_conf})"
    )

    print(
        f"PromptGuard  : {label_to_text(pg_pred)} "
        f"(confidence={pg_conf})"
    )

    # Conservative safety logic:
    # if either says harmful -> final harmful
    final_pred = max(deberta_pred, pg_pred)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(label_to_text(final_pred))

    print("\nDemo complete.\n")


if __name__ == "__main__":
    main()
