import pandas as pd

df = pd.read_csv("dataset_with_behavior_labels.csv")

print(df["behavior_label"].value_counts())
print(df.sample(10)[["response", "behavior_label"]])
