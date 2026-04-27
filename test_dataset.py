# Import the necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm  # A library to show a progress bar
from sklearn.metrics import classification_report


def evaluate_prompt_guard_on_benchmark():
    """
    Evaluates the PromptGuard model on a public prompt injection benchmark dataset.
    """
    print("--- Starting Evaluation of meta-llama/Prompt-Guard-86M ---")

    # 1. MODEL AND PUBLIC DATASET
    model_id = "meta-llama/Prompt-Guard-86M"
    dataset_name = "deepset/prompt-injections"  # Public benchmark dataset

    print(f"Loading model: {model_id}")
    print(f"Loading public benchmark dataset: {dataset_name}")

    # 2. LOAD THE DATASET
    try:
        dataset = load_dataset(dataset_name, split="test")
        print(f"Dataset loaded successfully with {len(dataset)} examples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. LOAD THE TOKENIZER AND MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. SET UP THE GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")
    model.eval()

    # 5. RUN THE EVALUATION
    y_true = []
    y_pred = []

    # Model labels: 0 = benign, 1 = injection, 2 = jailbreak
    # Dataset labels: 0 = benign, 1 = malicious
    # Map model's {1,2} → 1 (malicious), 0 → 0 (benign)

    print(f"\nRunning predictions on {len(dataset)} test samples...")

    for example in tqdm(dataset, desc="Evaluating"):
        prompt = example["text"]
        true_label = example["label"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()

            predicted_label = 1 if predicted_class_id in [1, 2] else 0

        y_true.append(true_label)
        y_pred.append(predicted_label)

    # 6. CALCULATE AND PRINT RESULTS
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print("\n--- Evaluation Complete ---")
    print(f"Evaluated on dataset: {dataset_name}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malicious"]))
    print("---------------------------")


# Run the main function
if __name__ == "__main__":
    evaluate_prompt_guard_on_benchmark()
