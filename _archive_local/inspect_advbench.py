from datasets import load_dataset

# The file you uploaded
advbench_file = "data-00000-of-00001.arrow"

print(f"--- Inspecting {advbench_file} ---")

# Load the dataset from the file
dataset = load_dataset("arrow", data_files={'train': advbench_file})['train']

# Print the column names (the schema)
print("\n[+] Column Names (Features):")
print(dataset.features)

# Print the first row to see an example
print("\n[+] First Row Example:")
print(dataset[0])
