import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load your cleaned data
df = pd.read_csv("train.tsv", sep='\t', header=None)
columns = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]
df.columns = columns
df = df[["label", "statement"]].dropna()

# Simplify labels
def simplify_label(label):
    if label in ["false", "pants-fire", "barely-true", "half-true"]:
        return "fake"
    else:
        return "real"

df["label"] = df["label"].apply(simplify_label)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["statement"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict and print results
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
