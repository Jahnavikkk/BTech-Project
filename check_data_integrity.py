import pandas as pd
import sys

# --- CONFIGURATION ---

# The large, corrupted file we are checking
INPUT_CSV = "committee_generations_balanced_v3.csv" 

# The new, clean file we will create if we find good data
CLEAN_OUTPUT_CSV = "clean_committee_generations.csv"

# The number of models in our committee. A "clean" prompt must have exactly this many responses.
EXPECTED_RESPONSES_PER_PROMPT = 10

# --- MAIN EXECUTION ---

print(f"--- Starting Integrity Check on {INPUT_CSV} ---")

# Load the massive CSV file
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"Successfully loaded {len(df)} total rows from the file.")
except FileNotFoundError:
    print(f"ERROR: Cannot find the input file: {INPUT_CSV}")
    print("Please make sure the file is in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    sys.exit(1)

if 'prompt_id' not in df.columns:
    print("ERROR: The CSV file does not contain a 'prompt_id' column. Cannot proceed.")
    sys.exit(1)

# This is the core logic:
# 1. Count how many times each unique 'prompt_id' appears.
prompt_counts = df['prompt_id'].value_counts()

# 2. Find the 'prompt_id's that appear exactly 10 times. These are our "clean" prompts.
clean_prompt_ids = prompt_counts[prompt_counts == EXPECTED_RESPONSES_PER_PROMPT].index

# 3. Filter the original DataFrame to keep *only* the rows that belong to these clean prompts.
clean_df = df[df['prompt_id'].isin(clean_prompt_ids)]

# --- REPORTING ---

num_clean_prompts = len(clean_prompt_ids)
num_clean_rows = len(clean_df)

print("\n--- DATA INTEGRITY REPORT ---")
print(f"Total unique prompts found in file: {len(prompt_counts)}")
print(f"Found {num_clean_prompts} clean, fully processed prompts (with exactly {EXPECTED_RESPONSES_PER_PROMPT} model responses).")
print(f"Found {len(prompt_counts[prompt_counts != EXPECTED_RESPONSES_PER_PROMPT])} corrupted/partial prompts.")

# --- FINAL ACTION ---

if num_clean_prompts > 0:
    print("\n--- SUCCESS ---")
    print(f"Extracting {num_clean_rows} clean responses...")
    
    # Save the new, clean DataFrame
    clean_df.to_csv(CLEAN_OUTPUT_CSV, index=False)
    
    print(f"\nSuccessfully saved clean data to '{CLEAN_OUTPUT_CSV}'")
    print("You can now proceed to Phase 2 (Judging) using this new file.")
    print("Remember to update the 'INPUT_CSV' variable in 'run_committee_judging.py' to point to this new file.")
else:
    print("\n--- ACTION REQUIRED ---")
    print("No clean data was found. The file is unusable.")
    print("Please re-run the 'run_committee_generation_SAMPLED.py' script to generate fresh, clean data.")
