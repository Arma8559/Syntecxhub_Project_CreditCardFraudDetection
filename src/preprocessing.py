import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])
    df["Time"] = scaler_time.fit_transform(df[["Time"]])
    return df

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Normal: {(y_res == 0).sum():,} | Fraud: {(y_res == 1).sum():,}")
    return X_res, y_res

def apply_undersampling(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"After Undersampling — Normal: {(y_res == 0).sum():,} | Fraud: {(y_res == 1).sum():,}")
    return X_res, y_res

def apply_smote_tomek(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    print(f"After SMOTETomek — Normal: {(y_res == 0).sum():,} | Fraud: {(y_res == 1).sum():,}")
    return X_res, y_res
