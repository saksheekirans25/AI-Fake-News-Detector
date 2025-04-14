from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from gpt2_model import predict as gpt2_predict

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

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = gpt2_predict(text)
    fact_check = fetch_fact_check_results(text)

    return jsonify({
        "result": result["label"],
        "confidence": result["confidence"],
        "fact_check": fact_check
    })

if __name__ == "__main__":
    app.run(debug=True)
