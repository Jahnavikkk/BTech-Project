import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("final_large_dataset.csv")

df["final_label"].value_counts().plot(kind="bar")
plt.title("Expanded Dataset Label Distribution")
plt.savefig("large_dataset_dist.png")
