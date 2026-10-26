
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Load the trained model and TF-IDF vectoriser
model = joblib.load("model/service_request_model.joblib")
vectoriser = joblib.load("model/tfidf_vectoriser.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/categories", methods=["GET"])
def categories():
    return jsonify({
        "categories": ["road", "safety", "waste", "water"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    # Check that the request contains valid JSON
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400

    # Validate the request text
    request_text = data.get("request_text")

    if not isinstance(request_text, str) or not request_text.strip():
        return jsonify({
            "error": "request_text must be a non-empty string"
        }), 400

    # Clean and transform the text
    cleaned_text = request_text.lower().strip()
    text_tfidf = vectoriser.transform([cleaned_text])

    # Predict the category and confidence
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)

    return jsonify({
        "predicted_category": str(prediction),
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
