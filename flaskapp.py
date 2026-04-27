from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Score Predictor API. POST hours to /predict, or GET /predict?hours=5"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        hours = data.get("hours")
    else:
        hours = request.args.get("hours")

    if hours is None:
        return jsonify({"error": "provide hours"}), 400

    try:
        hours = float(hours)
    except ValueError:
        return jsonify({"error": "hours must be numeric"}), 400

    prediction = model.predict(np.array([[hours]]))[0]
    return jsonify({"hours": hours, "predicted_score": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
