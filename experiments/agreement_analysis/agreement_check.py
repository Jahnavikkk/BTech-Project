import pandas as pd

df = pd.read_csv("dataset_with_behavior_labels.csv")

df = df.dropna(subset=["behavior_label", "final_label"])

agreement = (df["behavior_label"] == df["final_label"]).mean()

print(f"\nLLM vs Human Agreement: {agreement:.4f}\n")
