from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import requests

FACT_CHECK_API_KEY = "AIzaSyAOG9TEfkY_q3vYSlRL0_iURsxanmUQdRc"
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def fetch_fact_check_results(text):
    params = {
        "query": text,
        "key": FACT_CHECK_API_KEY
    }
    response = requests.get(FACT_CHECK_URL, params=params)
    data = response.json()
    if "claims" in data:
        claim = data["claims"][0]
        return {
            "text": claim["text"],
            "claimant": claim.get("claimant", "Unknown"),
            "rating": claim["claimReview"][0]["textualRating"],
            "url": claim["claimReview"][0]["url"]
        }
    return None


# Load the pre-trained BERT classifier
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

app = Flask(__name__)
CORS(app)

def interpret_label(label):
    return "fake" if label == "LABEL_1" else "real"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text)[0]
    interpreted = interpret_label(result["label"])
    fact_check = fetch_fact_check_results(text)

    return jsonify({
        "result": interpreted,
        "confidence": round(float(result["score"]), 2),
        "fact_check": fact_check
    })
if __name__ == "__main__":
    app.run(debug=True)

