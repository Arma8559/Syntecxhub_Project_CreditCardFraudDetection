import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")

def load_model(filepath: str):
    return joblib.load(filepath)
