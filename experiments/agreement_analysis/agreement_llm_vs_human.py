import pandas as pd

df = pd.read_csv("dataset_with_llm_labels.csv")

df = df.dropna(subset=["llm_label", "final_label"])

agreement = (df["llm_label"] == df["final_label"]).mean()

print(f"\nLLM vs Human Agreement: {agreement:.4f}\n")
