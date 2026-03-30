import os
import sys
import time

from src.data_loader import load_data, get_features_and_target
from src.eda import generate_eda_report
from src.preprocessing import scale_features, split_data, apply_smote
from src.model import train_random_forest, train_xgboost, save_model
from src.evaluate import evaluate_model, compare_models
DATA_PATH = os.path.join("data", "creditcard.csv")
PLOTS_DIR = os.path.join("outputs", "plots")
MODELS_DIR = os.path.join("outputs", "models")
REPORTS_DIR = os.path.join("outputs", "reports")

def main():
    start_time = time.time()

    print("\n" + "=" * 60)
    print("   Credit Card Fraud Detection — ML Pipeline")
    print("   Syntecxhub Internship | Week 3 | Project 2")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at '{DATA_PATH}'")
        print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Place 'creditcard.csv' inside the 'data/' folder.")
        sys.exit(1)

    for directory in [PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    print("\n[1/6] Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")

    print("\n[2/6] Running Exploratory Data Analysis...")
    generate_eda_report(df, PLOTS_DIR)

    print("\n[3/6] Preprocessing data...")
    df = scale_features(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    print("\n[4/6] Applying SMOTE oversampling...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\n[5/6] Training models...")

    print("      Training Random Forest...")
    rf_model = train_random_forest(X_train_res, y_train_res)
    save_model(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))

    print("      Training XGBoost...")
    xgb_model = train_xgboost(X_train_res, y_train_res)
    save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))

    print("\n[6/6] Evaluating models...")
    results = []
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest", PLOTS_DIR))
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost", PLOTS_DIR))

    compare_models(results, PLOTS_DIR)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline completed in {elapsed:.1f} seconds")
    print(f"  Plots saved   → {PLOTS_DIR}/")
    print(f"  Models saved  → {MODELS_DIR}/")
    print(f"  Reports saved → {REPORTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
