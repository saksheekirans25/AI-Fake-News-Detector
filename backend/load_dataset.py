import pandas as pd

# Load data
df = pd.read_csv("train.tsv", sep='\t', header=None)

# Rename columns properly
columns = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]
df.columns = columns

# Keep only label and statement
df = df[["label", "statement"]]

# Drop rows with missing values
df = df.dropna()

# Simplify labels into binary classes: fake vs real
def simplify_label(label):
    if label in ["false", "pants-fire", "barely-true", "half-true"]:
        return "fake"
    else:
        return "real"

df["label"] = df["label"].apply(simplify_label)

# Check the cleaned result
print(df["label"].value_counts())
print(df.head())
