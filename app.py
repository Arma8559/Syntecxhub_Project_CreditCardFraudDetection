import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

RF_PATH = os.path.join("outputs", "models", "random_forest.pkl")
XGB_PATH = os.path.join("outputs", "models", "xgboost.pkl")

rf_model = None
xgb_model = None

def load_models():
    global rf_model, xgb_model
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
    if os.path.exists(XGB_PATH):
        xgb_model = joblib.load(XGB_PATH)


load_models()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        amount = float(data.get("amount", 0))
        time_val = float(data.get("time", 0))

        scaler = StandardScaler()
        amount_scaled = (amount - 88.35) / 250.12
        time_scaled = (time_val - 94813.86) / 47488.15

        v_features = [float(data.get(f"v{i}", 0)) for i in range(1, 29)]

        feature_order = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
                         "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                         "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
                         "V25", "V26", "V27", "V28", "Amount"]

        features = [time_scaled] + v_features + [amount_scaled]
        X = np.array(features).reshape(1, -1)

        results = {}

        if rf_model is not None:
            rf_pred = int(rf_model.predict(X)[0])
            rf_prob = float(rf_model.predict_proba(X)[0][1])
            results["random_forest"] = {
                "prediction": rf_pred,
                "fraud_probability": round(rf_prob * 100, 2),
                "verdict": "FRAUD" if rf_pred == 1 else "LEGITIMATE"
            }

        if xgb_model is not None:
            xgb_pred = int(xgb_model.predict(X)[0])
            xgb_prob = float(xgb_model.predict_proba(X)[0][1])
            results["xgboost"] = {
                "prediction": xgb_pred,
                "fraud_probability": round(xgb_prob * 100, 2),
                "verdict": "FRAUD" if xgb_pred == 1 else "LEGITIMATE"
            }

        if results:
            probs = [v["fraud_probability"] for v in results.values()]
            avg_prob = round(sum(probs) / len(probs), 2)
            final_verdict = "FRAUD" if avg_prob >= 50 else "LEGITIMATE"
        else:
            avg_prob = 0
            final_verdict = "MODEL NOT LOADED"

        return jsonify({
            "success": True,
            "final_verdict": final_verdict,
            "avg_fraud_probability": avg_prob,
            "models": results,
            "amount": amount,
            "time": time_val
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/model_status")
def model_status():
    return jsonify({
        "random_forest": rf_model is not None,
        "xgboost": xgb_model is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
