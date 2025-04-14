import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer, GPT2Model
import torch
import joblib
import os
from tqdm import tqdm

# Load the combined dataset
df = pd.read_csv("combined_dataset.csv")
texts = df["statement"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# TEMP (for speed testing): Uncomment to limit data
texts = texts[:2000]
labels = labels[:2000]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Pad token if needed
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Batching for efficiency
def get_embeddings_batch(texts, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

# Generate embeddings
X = get_embeddings_batch(texts, batch_size=16)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and encoder
joblib.dump(clf, "gpt2_classifier.joblib")
joblib.dump(le, "label_encoder.joblib")
print("Model and label encoder saved.")
