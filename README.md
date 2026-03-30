# FraudSense — Credit Card Fraud Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-Oversampling-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p align="center">
  <strong>Syntecxhub ML Internship </strong>
</p>

> An end-to-end machine learning system for detecting fraudulent credit card transactions. Includes a full ML training pipeline (EDA → SMOTE → Random Forest + XGBoost) and a dark-themed interactive Flask web application for real-time predictions.

---

## Live Demo

Run locally → open `http://127.0.0.1:5000` after following the setup steps below.

---

## Overview

Credit card fraud costs the global economy billions of dollars annually. This project tackles the challenge of detecting fraud from a **highly imbalanced dataset** (only 0.172% fraud) using proven ML techniques and presents results through an intuitive web interface.

**Key challenge:** Standard accuracy is misleading — a model predicting "Normal" for everything achieves 99.83% accuracy but catches zero frauds. This project uses **Precision, Recall, ROC-AUC, and F1** as meaningful metrics, with **SMOTE** to handle class imbalance.

---

## Project Architecture

```
Two-layer system:

Layer 1 — ML Pipeline (run once)
  creditcard.csv → EDA → Scale → SMOTE → Train RF + XGBoost → Save .pkl

Layer 2 — Web App (run anytime after Layer 1)
  Browser (index.html) → POST /predict → Flask (app.py) → .pkl model → JSON verdict
```

### Full Project Structure

```
Syntecxhub_Project_CreditCardFraudDetection/
│
├── data/
│   └── creditcard.csv              ← Download from Kaggle (not in Git)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              ← CSV loading and feature extraction
│   ├── eda.py                      ← 5 EDA plots with statistical summary
│   ├── preprocessing.py            ← StandardScaler + train/test split + SMOTE
│   ├── model.py                    ← Random Forest and XGBoost training
│   └── evaluate.py                 ← Metrics, ROC, PR curves, threshold analysis
│
├── templates/
│   └── index.html                  ← Flask web app frontend (dark fintech UI)
│
├── outputs/
│   ├── plots/                      ← 10 PNG plots auto-generated
│   ├── models/                     ← Trained model .pkl files
│   └── reports/                    ← model_comparison.csv
│
├── main.py                         ← ML pipeline entry point
├── app.py                          ← Flask web server
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Features | 30 (Time, V1–V28 via PCA, Amount) |
| Target | `Class` — 0 = Normal, 1 = Fraud |
| Fraud rate | 0.172% — 492 fraud cases |
| Missing values | None |

> The dataset is not included due to size. Download `creditcard.csv` from Kaggle and place it in `data/`.

---

## ML Pipeline — What Happens in `main.py`

### Step 1 — EDA (5 plots)
- Class imbalance visualization (bar + pie)
- Transaction amount distribution by class
- Temporal distribution across 48-hour window
- Full feature correlation heatmap (30×30)
- V1–V14 boxplots: Normal vs Fraud comparison

### Step 2 — Preprocessing
- `StandardScaler` applied to `Time` and `Amount` (V1–V28 are already PCA-scaled)
- 80/20 stratified train/test split
- **SMOTE** applied only to training set → balances from 227,451 vs 394 to 1:1

### Step 3 — Model Training

**Random Forest**
- 200 estimators, max_depth=12, class_weight='balanced'
- Ensemble of decision trees via majority vote

**XGBoost**
- 200 estimators, learning_rate=0.05, scale_pos_weight for imbalance
- Sequential gradient boosting — each tree corrects previous errors

### Step 4 — Evaluation (per model)
- Confusion matrix (TN, FP, FN, TP)
- ROC Curve with AUC score
- Precision-Recall Curve with Average Precision
- Threshold Analysis — F1/Precision/Recall across all decision thresholds

---

## Web App — What Happens in `app.py`

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the main prediction UI |
| `/predict` | POST | Accepts feature JSON, returns verdict from both models |
| `/model_status` | GET | Returns whether RF and XGBoost are loaded |

The frontend sends transaction features as JSON. Flask loads the saved `.pkl` models, runs `predict_proba()`, and returns fraud probability and verdict for both models. The browser renders the result with confidence bars and per-model breakdown.

---

## Why These Evaluation Metrics?

| Metric | Why it matters for fraud |
|---|---|
| **Recall** | Are we catching actual frauds? Missing a fraud = financial loss |
| **Precision** | Are our fraud flags real? Too many false alarms = bad UX |
| **ROC-AUC** | Overall discrimination ability across all thresholds |
| **F1-Score** | Harmonic mean — balances Precision and Recall |
| **Threshold Analysis** | Business teams can tune the operating point |

A model that only predicts "Normal" gets 99.83% accuracy — but 0% Recall. That is why accuracy is never used here.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Syntecxhub_Project_CreditCardFraudDetection.git
cd Syntecxhub_Project_CreditCardFraudDetection
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

---

## Running the Project

### Step 1 — Train the ML models (run once)

```bash
python main.py
```

This will:
- Generate 10 EDA and evaluation plots in `outputs/plots/`
- Save `random_forest.pkl` and `xgboost.pkl` in `outputs/models/`
- Save `model_comparison.csv` in `outputs/reports/`
- Print full evaluation metrics to terminal

Estimated runtime: 3–7 minutes depending on hardware.

### Step 2 — Launch the web app

```bash
python app.py
```

Open your browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

> `main.py` must be run at least once before `app.py` — the web app needs the saved `.pkl` model files.

---

## Testing the Web App

### Normal transaction
- Amount: `150`, all V sliders at center → Expected: `LEGITIMATE`

### Fraud transaction (preset)
- Click **"Suspicious Pattern"** preset
- V1=−4.77, V3=−5.03, V14=−6.1, Amount=$1.00
- Expected: `FRAUD` with high probability

### Manual fraud pattern
- Amount: `0.01–5.00` (very small)
- V1, V3, V14 sliders: drag to far left (−10)
- Watch fraud probability increase

---

## Output Files Generated

| File | Description |
|---|---|
| `01_class_distribution.png` | Class imbalance bar and pie chart |
| `02_amount_distribution.png` | Amount histogram by class |
| `03_time_distribution.png` | Temporal distribution |
| `04_correlation_heatmap.png` | 30-feature correlation matrix |
| `05_feature_boxplots.png` | V1–V14 Normal vs Fraud |
| `06_confusion_matrix_*.png` | Per-model confusion matrix |
| `07_roc_curve_*.png` | ROC curve with AUC |
| `08_precision_recall_*.png` | Precision-Recall curve |
| `09_threshold_analysis_*.png` | Business threshold analysis |
| `10_model_comparison.png` | Side-by-side metric comparison |
| `model_comparison.csv` | Final metrics table |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Flask 3.0 |
| ML Models | scikit-learn (Random Forest), XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model Persistence | joblib |

---

## Author
**Rafiul Islam**

IoT & Robotics Engineering

University of Frontier Technology Bangladesh (UFTB) 

Syntecxhub ML Internship

---
