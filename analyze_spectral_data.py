import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, util
import sys
import re
import torch 
# --- 1. CONFIGURATION ---
INPUT_CSV = "spectral_data.csv"
SUSPICIOUS_KEYWORDS = ['ignore', 'rules', 'policy', 'harmful', 'unsafe', 'jailbreak', 'forget']
# Use a fast, local sentence transformer model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

# --- 2. FEATURE EXTRACTION ---

# Load the embedding model globally once
print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
print(f"Model loaded on {device}.")

def extract_features(group):
    """Calculates the expanded feature vector for a group of k responses."""
    
    # Ensure all response texts are strings, handle potential errors
    responses = group['response_text'].fillna("").astype(str).tolist()
    labels = group['response_label'].fillna("UNKNOWN").astype(str).tolist()

    if not responses or len(responses) < 2: # Need at least 2 responses for variance/dispersion
        return pd.Series({
            'label_variance': 0.0,
            'length_variance': 0.0,
            'keyword_jitter': 0.0,
            'semantic_dispersion': 0.0 
        })

    # X1: Label Variance
    label_map = {'INJECTION': 1, 'SAFE': 0, 'UNKNOWN': 0.5, 'ERROR': 0.5}
    numeric_labels = [label_map.get(label, 0.5) for label in labels]
    label_variance = np.var(numeric_labels) if len(numeric_labels) > 1 else 0.0
    
    # X2: Length Variance
    response_lengths = [len(text) for text in responses]
    length_variance = np.var(response_lengths) if len(response_lengths) > 1 else 0.0
    
    # X3: Keyword Jitter
    keyword_counts = []
    for text in responses:
        count = 0
        for keyword in SUSPICIOUS_KEYWORDS:
            count += len(re.findall(keyword, text.lower()))
        keyword_counts.append(count)
    keyword_jitter = np.var(keyword_counts) if len(keyword_counts) > 1 else 0.0

    # X4: Semantic Dispersion (using cosine distance)
    try:
        if len(responses) >= 2:
            embeddings = embedding_model.encode(responses, convert_to_tensor=True, device=device)
            # Calculate pairwise cosine similarities
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            # Convert similarity to distance (1 - similarity)
            # Get the upper triangle of the distance matrix (excluding diagonal)
            distances = 1 - cosine_scores[np.triu_indices(len(embeddings), k=1)].cpu().numpy()
            semantic_dispersion = np.mean(distances) if distances.size > 0 else 0.0
        else:
            semantic_dispersion = 0.0
    except Exception as e:
        print(f"Warning: Error calculating semantic dispersion - {e}")
        semantic_dispersion = 0.0 # Assign a default value on error

    return pd.Series({
        'label_variance': label_variance,
        'length_variance': length_variance,
        'keyword_jitter': keyword_jitter,
        'semantic_dispersion': semantic_dispersion
    })

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {INPUT_CSV}. Processing {df['prompt'].nunique()} unique prompts.")
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Please run 'run_spectral_interrogation.py' first.")
        sys.exit(1)

    print("Extracting expanded features from raw data...")
    # Group by prompt and apply our feature extraction
    # Need to handle potential errors during apply if groups are problematic
    features_df = df.groupby('prompt', group_keys=False).apply(extract_features)
    
    # Add the true_label back in
    labels = df.drop_duplicates(subset=['prompt']).set_index('prompt')['true_label']
    features_df = features_df.join(labels)
    
    # Drop rows where feature extraction might have failed (e.g., if only 1 probe response)
    features_df.dropna(inplace=True) 

    if features_df.empty:
         print("Error: No valid features extracted. Exiting.")
         sys.exit(1)

    # --- 4. THE VIABILITY TEST (TRAIN & TEST SVM) ---
    
    feature_columns = ['label_variance', 'length_variance', 'keyword_jitter', 'semantic_dispersion']
    X = features_df[feature_columns]
    y = features_df['true_label']
    
    # Check if we have enough data and both classes
    if len(X) < 10 or len(y.unique()) < 2:
        print("Error: Not enough data or only one class present after feature extraction. Cannot train model.")
        sys.exit(1)

    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining simple SVM model on {len(feature_columns)} features...")
    # Train the simple ML model
    model = SVC(kernel='rbf', C=1.0, random_state=42) 
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the unseen 20%
    y_pred = model.predict(X_test_scaled)
    
    # --- 5. THE "PROOF OF VIABILITY" REPORT ---
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print("\n=======================================================")
    print("      SPECTRAL GUARD-ML VIABILITY TEST (Expanded Features)      ")
    print("=======================================================")
    print(f"Trained a simple SVM on {len(X_train)} prompt feature vectors ({len(feature_columns)} features).")
    print(f"Tested on {len(X_test)} unseen prompt feature vectors.")
    print("-------------------------------------------------------")
    print(f"   >>> VIABILITY TEST ACCURACY: {accuracy:.2f}% <<<   ")
    print("-------------------------------------------------------")
    
    if accuracy > 90:
        print("\nCONCLUSION: VIABLE. The signal is strong and separable.")
        print("A simple ML model can detect attacks using instability features.")
    elif accuracy > 75:
        print("\nCONCLUSION: PROMISING. The signal is present.")
    else:
        print("\nCONCLUSION: NOT VIABLE (with these features).")
        print("The instability features are not a separable signal.")
        
    print("\nFull Classification Report:\n")
    print(classification_report(y_test, y_pred))

