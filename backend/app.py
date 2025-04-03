from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS


# Load trained model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Create Flask app
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform the input and make prediction
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector).max()

    return jsonify({
        "result": prediction,
        "confidence": round(float(probability), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
