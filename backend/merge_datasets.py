import pandas as pd
import json
import os

# === Paths ===
data_dir = "datasets"
liar_path = os.path.join(data_dir, "liar_dataset.tsv")
fever_paths = [
    os.path.join(data_dir, "train.jsonl"),
    os.path.join(data_dir, "dev.jsonl")
]
fakenewsnet_paths = [
    (os.path.join(data_dir, "politifact_fake.csv"), "fake"),
    (os.path.join(data_dir, "politifact_real.csv"), "real"),
    (os.path.join(data_dir, "gossipcop_fake.csv"), "fake"),
    (os.path.join(data_dir, "gossipcop_real.csv"), "real")
]

# === Load LIAR ===
print("Loading LIAR...")
liar_df = pd.read_csv(liar_path, sep='\t', header=None, names=[
    'id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
])
liar_map = {
    'true': 'real', 'mostly-true': 'real', 'half-true': 'real',
    'barely-true': 'fake', 'false': 'fake', 'pants-fire': 'fake'
}
liar_df['label'] = liar_df['label'].map(liar_map)
liar_df = liar_df[['statement', 'label']].dropna()

# === Load FEVER ===
print("Loading FEVER...")
fever_data = []
for file in fever_paths:
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['label'] == 'SUPPORTS':
                label = 'real'
            elif item['label'] == 'REFUTES':
                label = 'fake'
            else:
                continue  # skip NOT ENOUGH INFO
            fever_data.append({
                'statement': item['claim'],
                'label': label
            })
fever_df = pd.DataFrame(fever_data)

# === Load FakeNewsNet CSVs ===
print("Loading FakeNewsNet...")
fakenews_df = []
for path, label in fakenewsnet_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[['title']].dropna()
        df = df.rename(columns={'title': 'statement'})
        df['label'] = label
        fakenews_df.append(df)
    else:
        print(f"Missing: {path}")
fakenews_df = pd.concat(fakenews_df, ignore_index=True)

# === Combine all ===
print("Merging all datasets...")
combined_df = pd.concat([liar_df, fever_df, fakenews_df], ignore_index=True)
print(f"Total samples: {len(combined_df)}")

# === Save ===
combined_df.to_csv("combined_dataset.csv", index=False)
print("Saved as combined_dataset.csv")
