import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

RF_PATH = os.path.join("outputs", "models", "random_forest.pkl")
XGB_PATH = os.path.join("outputs", "models", "xgboost.pkl")

rf_model = None
xgb_model = None


def load_models():
    global rf_model, xgb_model
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
        print(f"[OK] Random Forest loaded from {RF_PATH}")
    else:
        print(f"[ERROR] Random Forest not found at {RF_PATH}")
    if os.path.exists(XGB_PATH):
        xgb_model = joblib.load(XGB_PATH)
        print(f"[OK] XGBoost loaded from {XGB_PATH}")
    else:
        print(f"[ERROR] XGBoost not found at {XGB_PATH}")


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

        amount_scaled = (amount - 88.35) / 250.12
        time_scaled = (time_val - 94813.86) / 47488.15

        v_features = [float(data.get(f"v{i}", 0)) for i in range(1, 29)]

        features = [time_scaled] + v_features + [amount_scaled]
        X = np.array(features).reshape(1, -1)

        print("\n--- Incoming Transaction ---")
        print(f"Amount: {amount} (scaled: {amount_scaled:.4f})")
        print(f"Time: {time_val} (scaled: {time_scaled:.4f})")
        print(f"V1={v_features[0]:.2f} V3={v_features[2]:.2f} V4={v_features[3]:.2f} V14={v_features[13]:.2f}")
        print(f"Feature vector shape: {X.shape}")

        results = {}

        if rf_model is not None:
            rf_prob = float(rf_model.predict_proba(X)[0][1])
            rf_pred = 1 if rf_prob >= 0.35 else 0
            print(f"Random Forest fraud probability: {rf_prob * 100:.2f}%")
            results["random_forest"] = {
                "prediction": rf_pred,
                "fraud_probability": round(rf_prob * 100, 2),
                "verdict": "FRAUD" if rf_pred == 1 else "LEGITIMATE"
            }

        if xgb_model is not None:
            xgb_prob = float(xgb_model.predict_proba(X)[0][1])
            xgb_pred = 1 if xgb_prob >= 0.35 else 0
            print(f"XGBoost fraud probability:      {xgb_prob * 100:.2f}%")
            results["xgboost"] = {
                "prediction": xgb_pred,
                "fraud_probability": round(xgb_prob * 100, 2),
                "verdict": "FRAUD" if xgb_pred == 1 else "LEGITIMATE"
            }

        if results:
            probs = [v["fraud_probability"] for v in results.values()]
            avg_prob = round(sum(probs) / len(probs), 2)
            final_verdict = "FRAUD" if avg_prob >= 35 else "LEGITIMATE"
        else:
            avg_prob = 0
            final_verdict = "MODEL NOT LOADED"

        print(f"Average fraud prob: {avg_prob}% → Final verdict: {final_verdict}")
        print("----------------------------\n")

        return jsonify({
            "success": True,
            "final_verdict": final_verdict,
            "avg_fraud_probability": avg_prob,
            "models": results,
            "amount": amount,
            "time": time_val
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/model_status")
def model_status():
    return jsonify({
        "random_forest": rf_model is not None,
        "xgboost": xgb_model is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
