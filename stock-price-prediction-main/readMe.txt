# 📈 Stock Price Predictor

> A powerful Streamlit web application that predicts future stock prices using **LSTM** and **Random Forest** models with full **user authentication**, **admin controls**, and **saved model management**.

![App Preview](assets/app-preview.png)

---

## 🚀 Features

- 🔐 User login & signup system
- 👑 Admin dashboard with user and log management
- 📊 Predict stock prices using:
  - 🧠 Long Short-Term Memory (LSTM)
  - 🌲 Random Forest Regressor
- 💾 Save, rename, and delete models
- 📈 Visualize predictions vs actual prices
- 🔮 30-day future forecasting
- 🧮 Customize hyperparameters (epochs, batch size, estimators, etc.)
- ⏳ Historical data from Yahoo Finance
- 📂 User-specific saved model access

---

## 📷 Preview

> You can replace this with a real screenshot.

![Screenshot](assets/app-preview.png)

---

## 🧠 ML Models Used

### ✅ LSTM
- Deep learning model for time-series prediction
- Works on past 60 days of closing stock prices
- Forecasts next 30 days using learned trends

### ✅ Random Forest
- Ensemble regression model
- Trained on flattened 60-day windows
- Faster and interpretable vs LSTM

---

## 🔐 Authentication System

- Every user must **log in or register**
- Admin users get extra features:
  - Toggle admin status for others
  - View recent user activity logs
  - View/download all user models

---

## 💾 Model Saving & Loading

- Save trained models with name and parameters
- Reload models later to forecast again
- Rename/delete any saved model from the UI

---

## 📁 Folder Structure

```bash
📦 stock-price-prediction-main/
├── app.py
├── auth.py
├── database.py
├── model/                     # Saved models
├── assets/
│   └── app-preview.png        # Optional UI screenshot
├── requirements.txt
└── README.md
