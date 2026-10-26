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
    return jsonify(["road", "safety", "waste", "water"])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    request_text = data.get("request_text", "")

    if not request_text.strip():
        return jsonify({"error": "request_text is required"}), 400

    cleaned_text = request_text.lower().strip()

    text_tfidf = vectoriser.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)

    return jsonify({
        "category": prediction,
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
