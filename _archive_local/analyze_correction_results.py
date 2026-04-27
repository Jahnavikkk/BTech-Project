import pandas as pd
import sys

# Set pandas to show full text, not truncated text
pd.set_option('display.max_colwidth', None)

# --- 1. Load the Data ---
csv_file = "self_correction_results.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {csv_file}. Found {len(df)} total prompts.\n")
except FileNotFoundError:
    print(f"ERROR: Could not find the file {csv_file}")
    sys.exit()

# --- 2. Calculate the Key Metric ---
# This is the % of prompts that are now 'PASS' (originally all 306 were 'FAIL')
self_correction_rate = (df['new_verdict'] == 'PASS').mean() * 100

# --- 3. Print the Main Report ---
print("======================================================")
print("      CHAPTER 3: SELF-CORRECTION RESULTS      ")
print("======================================================")
print(f"Total Prompts Tested: {len(df)}")
print(f"Self-Correction Rate: {self_correction_rate:.2f}%")
print("======================================================")
print(f"\n(This is the percentage of 'FAIL' prompts that the model \n successfully 'self-corrected' to a 'PASS' by using CoT.)\n")

# --- 4. Show Qualitative Examples ---

print("--- EXAMPLES OF SUCCESS (FAIL -> PASS) ---")
# Get up to 5 examples of successful self-correction
pass_examples = df[df['new_verdict'] == 'PASS'].head(5)
if pass_examples.empty:
    print("No examples of successful self-correction found.")
else:
    for index, row in pass_examples.iterrows():
        print(f"\nPROMPT: {row['prompt']}")
        print("---------------------------------")
        print("FULL RESPONSE (CoT + Answer):")
        print(row['full_response'])
        print("---------------------------------")
        print(f"NEW VERDICT: {row['new_verdict']} (SUCCESS)")
        print("======================================================\n")

print("\n\n--- EXAMPLES OF FAILURE (FAIL -> FAIL) ---")
# Get up to 5 examples where it still failed
fail_examples = df[df['new_verdict'] == 'FAIL'].head(5)
if fail_examples.empty:
    print("No examples of failed self-correction found.")
else:
    for index, row in fail_examples.iterrows():
        print(f"\nPROMPT: {row['prompt']}")
        print("---------------------------------")
        print("FULL RESPONSE (CoT + Answer):")
        print(row['full_response'])
        print("---------------------------------")
        print(f"NEW VERDICT: {row['new_verdict']} (FAILURE)")
        print("======================================================\n")
