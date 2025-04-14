import torch
import joblib
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Set pad token if not defined
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load trained classifier and label encoder
clf = joblib.load("gpt2_classifier.joblib")
le = joblib.load("label_encoder.joblib")

# Embedding function (batched)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Predict function
def predict(text):
    embedding = get_embedding(text).reshape(1, -1)
    pred = clf.predict(embedding)[0]
    confidence = np.max(clf.predict_proba(embedding))
    label = le.inverse_transform([pred])[0]
    return {
        "label": label,
        "confidence": round(confidence * 100, 2)
    }
