import pandas as pd

# Set pandas to show full text, not truncated text
pd.set_option('display.max_colwidth', None)

# --- 1. Load the Data ---
csv_file = "evaluation_matrix_ollama_results.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {csv_file}. Found {len(df)} total prompts.\n")
except FileNotFoundError:
    print(f"ERROR: Could not find the file {csv_file}")
    exit()

# --- 2. Define Your Models ---
# These are the column names from your script
weak_model_verdict = 'tinydolphin_verdict'
strong_model_verdict = 'gemma3-27b_verdict'

# --- 3. Find the "Split Decision" Prompts ---
# This is the core of your hypothesis:
# Prompts where the weak model FAILED but the strong one PASSED.
split_decision_df = df[
    (df[weak_model_verdict] == 'FAIL') & 
    (df[strong_model_verdict] == 'PASS')
]

# --- 4. Print the Results to Your Terminal ---
print("======================================================")
print(f"Found {len(split_decision_df)} prompts that prove the hypothesis!")
print("======================================================")

# Print the top 10 examples
for index, row in split_decision_df.head(10).iterrows():
    print(f"\nPROMPT (ID: {row['prompt_id']}):")
    print(f"   {row['prompt_text']}")
    print(f"   ---------------------------------")
    print(f"   WEAK MODEL ({weak_model_verdict}):   {row[weak_model_verdict]}")
    print(f"   STRONG MODEL ({strong_model_verdict}): {row[strong_model_verdict]}")
    print("------------------------------------------------------")
