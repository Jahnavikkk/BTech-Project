import pandas as pd

df = pd.read_csv("final_large_dataset.csv")

print(len(df))
print(df["final_label"].value_counts())
print(df["final_label"].value_counts(normalize=True))
df = pd.read_csv("final_large_dataset.csv")

harmful = df[df["final_label"] == 1]
benign = df[df["final_label"] == 0].sample(len(harmful)*2, random_state=42)

df_bal = pd.concat([harmful, benign])

df_bal.to_csv("balanced_dataset.csv", index=False)

print(df_bal["final_label"].value_counts(normalize=True))
