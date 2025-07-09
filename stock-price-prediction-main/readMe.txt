# 📈 Stock Price Predictor

A Streamlit web app to forecast stock trends using **LSTM** and **Random Forest**, with secure user login, admin dashboard, real-time stock data, and saved model management. Designed with a professional UI and personalized experience for each user.

---

## 🔐 Login Interface

![Login](./demo1.jpeg)

---

## 📊 Dashboard & Prediction View

![Dashboard](./demo2.png)

---

## 🚀 Features

- 🔑 **Secure Login & Signup System**
- 👑 **Admin Dashboard**:
  - Manage all users
  - Promote/demote admin access
  - View activity logs
  - Browse all user models
- 📈 **Stock Prediction with ML**:
  - LSTM (Long Short-Term Memory)
  - Random Forest Regressor
  - Compare both models
- ⚙️ **Customizable Parameters**:
  - Epochs, batch size, LSTM units
  - RF estimators and depth
- 📉 **30-Day Forecasting**
- 📊 **Prediction Charts**:
  - Actual vs Predicted
  - Future forecast visualized
- 💾 **Model Management**:
  - Save, load, rename, delete
- 📂 **User-Specific Model Views**
- 🌐 **Real-time stock prices** via Yahoo Finance

---

## 🗂 Folder Structure

```bash
📦 stock-price-predictor/
├── app.py
├── auth.py
├── database.py
├── model/                  # Saved models
├── demo1.jpeg              # Login screen preview
├── demo2.png               # Dashboard preview
├── requirements.txt
└── README.md
```

⚙️ Setup & Installation
bash
Copy
Edit
# 1. Clone this repository
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
Then open your browser at: http://localhost:8501

🧠 ML Models Used
🔹 LSTM (Long Short-Term Memory)
Deep learning model ideal for sequential time-series data

Trained on 60-day historical closing prices

Makes 30-day future predictions

🔹 Random Forest Regressor
Ensemble ML model that works on 60-day feature windows

Easier to interpret and fast to train

Good for shorter-term predictions

💾 Code Snippet Preview
python
Copy
Edit
# Train/test split
train_size = int(len(x_data) * 0.9)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(lstm_units, input_shape=(x_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

📦 Requirements
txt
Copy
Edit
streamlit
tensorflow
scikit-learn
pandas
numpy
yfinance
Pillow

Install All with:
bash
Copy
Edit
pip install -r requirements.txt