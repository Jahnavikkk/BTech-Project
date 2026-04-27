import pandas as pd
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN_LABEL_DIR = REPO_ROOT / "data" / "human_labeling"

pairs = [
    ("annotator_set_1_A_annotated.csv", "annotator_set_1_B_annotated.csv"),
    ("annotator_set_2_A_annotated.csv", "annotator_set_2_B_annotated.csv"),  
    ("annotator_set_3_A_annotated.csv", "annotator_set_3_B_annotated.csv"),
]

all_data = []
all_disagreements = []

def read_file(file):
    try:
        return pd.read_csv(HUMAN_LABEL_DIR / file, encoding="latin1", engine="python", quotechar='"')
    except:
        return pd.read_csv(HUMAN_LABEL_DIR / file, encoding="latin1", engine="python", quotechar='"')

def get_label_col(df):
    for c in df.columns:
        if "human_label" in c.lower():
            return c
    raise ValueError("No label column found")

for fileA, fileB in pairs:

    print(f"\nProcessing {fileA} and {fileB}")

    dfA = read_file(fileA)
    dfB = read_file(fileB)

    # clean column names
    dfA.columns = dfA.columns.str.strip().str.replace('"', '')
    dfB.columns = dfB.columns.str.strip().str.replace('"', '')

    dfA = dfA.reset_index(drop=True)
    dfB = dfB.reset_index(drop=True)

    print("dfB columns:")
    for c in dfB.columns:
        print(repr(c))

    # detect label columns
    label_col_A = get_label_col(dfA)
    label_col_B = get_label_col(dfB)

    merged = dfA.copy()

    merged["label_A"] = dfA[label_col_A]
    merged["label_B"] = dfB[label_col_B]
    # --- Flip check for Set 3 ---
    if "set_3" in fileA:
       print("\n--- Flip check for Set 3 ---")

       print("A distribution:")
       print(merged["label_A"].value_counts())

       print("\nB distribution:")
       print(merged["label_B"].value_counts())

    flip_score = (merged["label_A"] == (1 - merged["label_B"])).mean()
    print("\nFlip score:", round(flip_score, 3))
    # agreement
    merged["agree"] = merged["label_A"] == merged["label_B"]
    # disagreements (including missing)
    disagreements = merged[
        (merged["label_A"] != merged["label_B"]) |
        (merged["label_A"].isna()) |
        (merged["label_B"].isna())
    ]

    # true disagreements
    true_disagreements = merged[
        (merged["label_A"] != merged["label_B"]) &
        (~merged["label_A"].isna()) &
        (~merged["label_B"].isna())
    ]

    # missing labels
    missing_labels = merged[
        (merged["label_A"].isna()) |
        (merged["label_B"].isna())
    ]

    print("Rows:", len(merged))
    print("True disagreements:", len(true_disagreements))
    print("Missing labels:", len(missing_labels))
    print("Total flagged rows:", len(disagreements))

    # kappa (only valid rows)
    kappa_df = merged.dropna(subset=["label_A", "label_B"])

    if len(kappa_df) > 0:
        kappa = cohen_kappa_score(
            kappa_df["label_A"],
            kappa_df["label_B"]
        )
        print("Cohen's Kappa:", round(kappa, 3))
    else:
        print("Cohen's Kappa: Not enough valid data")

    all_data.append(merged)
    all_disagreements.append(disagreements)

# combine all
full = pd.concat(all_data, ignore_index=True)
disagree = pd.concat(all_disagreements, ignore_index=True)

# save outputs
full.to_csv(HUMAN_LABEL_DIR / "annotations_combined.csv", index=False)
disagree.to_csv(HUMAN_LABEL_DIR / "annotation_disagreements.csv", index=False)

print("\nSaved:")
print("annotations_combined.csv")
print("annotation_disagreements.csv")

print("\nTotal rows:", len(full))
print("Total disagreements:", len(disagree))
