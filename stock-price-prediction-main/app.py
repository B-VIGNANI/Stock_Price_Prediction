# Imports for data handling, ML models and Streamlit UI
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from database import is_admin_user, get_all_users, set_admin_status, delete_user, log_event, get_all_user_models, get_logs
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image

# App layout set to wide for better display space
st.set_page_config(layout="wide")

# Authentication and Database Imports
from auth import show_login_signup, is_authenticated, get_current_user
from database import (
    save_model_file, list_user_models,
    delete_model_file, rename_model_file,
    load_lstm_model, load_rf_model
)

# Session State Initialization
if "user" not in st.session_state:
    st.session_state["user"] = None
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# Redirect to login/register if not authenticated
if not is_authenticated():
    show_login_signup()
    st.stop()

# Default
if "page" not in st.session_state:
    st.session_state["page"] = "📈 Price Predictor"

# Current User Info
user = get_current_user()
st.success(f"Logged in as: {user}")



# Sidebar Configuration and market snapshot
with st.sidebar:
    st.title("📊 Menu")

    if st.session_state.get("is_admin"):
        st.session_state["page"] = "Admin"
        st.markdown("🛠 Admin Dashboard")
    else:
        page = st.radio("Navigate", ["📈 Price Predictor", "💾 Saved Models"])
        st.session_state["page"] = page

    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    st.markdown("### 📈 Market Snapshot")
    tickers = ["AAPL", "TSLA", "GOOG"]
    for symbol in tickers:
        try:
            data = yf.Ticker(symbol).history(period="1d")
            price = data["Close"].iloc[-1] if not data.empty else None
            st.metric(label=symbol, value=f"${price:.2f}" if price else "N/A")
        except:
            st.metric(label=symbol, value="Error")

    st.markdown("---")
    st.markdown(f"**👤 Logged in as:** {user}")

    # Logout button resets the session
    if st.button("🚪 Logout"):
        st.session_state["user"] = None
        st.rerun()

    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    with st.expander("ℹ️ Quick Tips"):
        st.markdown("""
        - 💡 **Enter a stock ticker** like `AAPL` or `TSLA`.
        - ⏳ **Select a data range** for training.
        - 🧠 **Train the model** to forecast prices.
        - 💾 **Save and reuse** models anytime.
        """)




# Intro
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='margin-bottom: 0;'>📈 Stock Price Predictor</h1>
    <p style='color: grey; font-size: 18px;'>Predict future stock prices using a deep learning model with ease.</p>
</div>
""", unsafe_allow_html=True)

# Image
image = Image.open("nick-chong-N__BnvQ_w18-unsplash.jpg")
resized_image = image.resize((image.width, int(image.height * 0.8)))
st.image(resized_image, use_column_width=True)

st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)




# LSTM Model Training Function
def train_lstm_model(df, lstm_units, epochs, batch_size):
    closing_price = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_price.values)

    # Create sequences of past 60 days
    x_data, y_data = [], []
    for i in range(60, len(scaled_data)):
        x_data.append(scaled_data[i - 60:i])
        y_data.append(scaled_data[i])
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))

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

    # Early stopping to prevent overfitting by monitoring loss
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    with st.spinner("Training..."):
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)


    # Make predictions and reverse scaling
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # LSTM Future forecast for next 30 days
    last_100 = closing_price.tail(100)
    last_100_scaled = scaler.transform(last_100.values).reshape(1, -1, 1)
    future_preds = []
    for _ in range(30):
        pred = model.predict(last_100_scaled)[0]
        future_preds.append(pred)
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], [[pred]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds))
    mse = mean_squared_error(actual, predictions)

    return model, scaler, closing_price, predictions, actual, df.index[train_size+60:], future_preds, mse




# Random Forest Model Training Function
def train_rf_model(df, n_estimators, max_depth):
    closing_price = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_price)

    # Create feature windows of 60 days
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i].flatten())
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    # Train/test split
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    with st.spinner("Training..."):
        model.fit(X_train, y_train)

    # Make predictions and reverse scaling
    predictions = model.predict(X_test).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(actual, predictions)

    # Random Forest Future forecast for next 30 days
    last_60 = scaled_data[-60:].flatten().tolist()
    future_preds = []
    for _ in range(30):
        X_future = np.array(last_60[-60:]).reshape(1, -1)
        next_scaled = model.predict(X_future)[0]
        future_preds.append(next_scaled)
        last_60.append(next_scaled)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    return model, predictions, actual, mse, df.index[train_size+60:], future_preds




# Admin Panel Page
if st.session_state.get("page") == "Admin":
    st.title("🛠 Admin Dashboard")
    st.success(f"You are logged in as **{user}** (admin)")

    st.markdown("## 👥 User Management")
    users = get_all_users()

    for username, is_admin in users:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            role = "👑 Admin" if is_admin else "👤 User"
            st.write(f"**{username}** — {role}")
        with col2:
            if username != user:
                if st.button("🔁 Toggle Admin", key=f"toggle_{username}"):
                    set_admin_status(username, not is_admin)
                    st.rerun()
        with col3:
            if username != user:
                if st.button("🗑 Delete", key=f"delete_{username}"):
                    delete_user(username)
                    st.rerun()

    # User Models Section
    st.markdown("## 🗂 User Models")
    user_models = get_all_user_models()
    for u, models in user_models.items():
        with st.expander(f"{u} — {len(models)} model(s)"):
            for m in models:
                st.markdown(f"- {m}")

    # Activity Log Section
    st.markdown("## 📜 Recent Activity Log")
    logs = get_logs(10)
    for ts, uname, action in logs:
        st.markdown(f"🔹 `{ts}` — **{uname}**: {action}")

        st.markdown("---")
        st.markdown("⬅️ Use the sidebar to logout and return to user view.")




# Price predictor page
elif page == "📈 Price Predictor":

    # Set default hyperparameters
    default_params = {
        "lstm_units": 50,
        "epochs": 10,
        "batch_size": 32,
        "rf_estimators": 100,
        "rf_max_depth": 10
    }

    for key, val in default_params.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # User input for stock symbol
    stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, BTC-USD)", value="AAPL")
    st.markdown("🔍 [Search Ticker on Yahoo Finance](https://finance.yahoo.com/lookup)", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)


    # Model selection
    model_choice = st.radio("Choose model to train:", ["LSTM", "Random Forest", "Compare Both"])

    # Model parameter controls
    with st.expander("⚙️ Model Parameters"):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state["lstm_units"] = st.number_input("LSTM Units", min_value=10, max_value=200, value=st.session_state["lstm_units"])
            st.session_state["epochs"] = st.number_input("Epochs", min_value=1, max_value=100, value=st.session_state["epochs"])
            st.session_state["batch_size"] = st.number_input("Batch Size", min_value=8, max_value=256, value=st.session_state["batch_size"])

        with col2:
            st.session_state["rf_estimators"] = st.number_input("RF Estimators", min_value=10, max_value=500, value=st.session_state["rf_estimators"])
            st.session_state["rf_max_depth"] = st.number_input("RF Max Depth", min_value=1, max_value=50, value=st.session_state["rf_max_depth"])

        if st.button("🔄 Reset Parameters"):
            for key, value in default_params.items():
                st.session_state[key] = value
            st.rerun()

    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    # Choose how much data to use
    data_range = st.slider("Select how many years of data to use:", 2, 10, 2)
    train_model = st.button("Train Model & Predict")

    # Set date range for fetching stock data
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=365 * data_range)

    st.caption(f"Fetching data from **{start.date()}** to **{end.date()}**")
    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    if train_model:
        df = yf.download(stock, start=start, end=end, auto_adjust=True)
        if df.empty:
            st.error("Failed to load stock data.")
        else:
            st.success("Stock data loaded.")
            st.subheader(f"📉 Historical Closing Price for {stock.upper()}")
            st.line_chart(df['Close'])

        # Clear the session state of the model not selected
        if model_choice == "LSTM":
            st.session_state.pop("rf_result", None)
        if model_choice == "Random Forest":
            st.session_state.pop("lstm_result", None)

        # Train selected models
        if model_choice in ["LSTM", "Compare Both"]:
            model_lstm, scaler, closing_price, pred_lstm, actual_lstm, test_idx, future_preds, mse_lstm = train_lstm_model(
                df,
                st.session_state["lstm_units"],
                st.session_state["epochs"],
                st.session_state["batch_size"]
            )
            
            st.session_state["lstm_result"] = {
                "model": model_lstm, "scaler": scaler, "pred": pred_lstm,
                "actual": actual_lstm, "test_idx": test_idx, "forecast": future_preds,
                "mse": mse_lstm, "stock": stock
            }

        if model_choice in ["Random Forest", "Compare Both"]:
            model_rf, pred_rf, actual_rf, mse_rf, test_idx_rf, future_rf = train_rf_model(
                df,
                st.session_state["rf_estimators"],
                st.session_state["rf_max_depth"]
            )

            st.session_state["rf_result"] = {
                "model": model_rf, "pred": pred_rf, "actual": actual_rf,
                "test_idx": test_idx_rf, "mse": mse_rf, "forecast": future_rf,
                "stock": stock
            }

    # Save model section
    if "lstm_result" in st.session_state or "rf_result" in st.session_state:
            st.subheader("💾 Save Model")
            model_name = st.text_input("Name this model:")

            if st.button("Save"):
                if not model_name:
                    st.warning("Model name cannot be empty.")
                else:
                    params = {
                        "lstm_units": st.session_state["lstm_units"],
                        "epochs": st.session_state["epochs"],
                        "batch_size": st.session_state["batch_size"],
                        "rf_estimators": st.session_state["rf_estimators"],
                        "rf_max_depth": st.session_state["rf_max_depth"]
                    }
                    
                    # Save LSTM if available
                    saved_any = False    
                    if "lstm_result" in st.session_state:
                        save_model_file(
                            user,
                            st.session_state["lstm_result"]["model"],
                            st.session_state["lstm_result"]["scaler"],
                            st.session_state["lstm_result"]["stock"],
                            model_name,
                            actual=st.session_state["lstm_result"]["actual"],
                            pred=st.session_state["lstm_result"]["pred"],
                            test_idx=st.session_state["lstm_result"]["test_idx"],
                            forecast=st.session_state["lstm_result"]["forecast"],
                            params=params
                        )
                        st.success(f"LSTM model '{model_name}' saved.")
                        saved_any = True

                    # Save Random Forest if available
                    if "rf_result" in st.session_state:
                        save_model_file(
                            user,
                            st.session_state["rf_result"]["model"],
                            None,  # No scaler saving for RF in your setup
                            st.session_state["rf_result"]["stock"],
                            model_name,
                            actual=st.session_state["rf_result"]["actual"],
                            pred=st.session_state["rf_result"]["pred"],
                            test_idx=st.session_state["rf_result"]["test_idx"],
                            forecast=st.session_state["rf_result"]["forecast"],
                            params=params
                        )
                        st.success(f"Random Forest model '{model_name}' saved.")
                        saved_any = True

                    if saved_any:
                        log_event(user, f"Saved model: {model_name}")


    # Display Results section
    if "lstm_result" in st.session_state or "rf_result" in st.session_state:
        st.subheader("📉 Model Comparison")

        # Show Mean Squared Error (MSE) for each model in use
        st.markdown("### 📏 Mean Squared Error")
        col1, col2 = st.columns(2)
        with col1:
            if "lstm_result" in st.session_state:
                st.metric("LSTM MSE", f"{st.session_state['lstm_result']['mse']:.2f}")
        with col2:
            if "rf_result" in st.session_state:
                st.metric("Random Forest MSE", f"{st.session_state['rf_result']['mse']:.2f}")

        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

        # Plot the actual vs predicted test data
        st.markdown("### 🔍 Combined Predictions vs Actual (Test Data)")
        if "lstm_result" in st.session_state and "rf_result" in st.session_state:
            combined_df = pd.DataFrame({
                "Actual": st.session_state["lstm_result"]["actual"].flatten(),
                "LSTM Predicted": st.session_state["lstm_result"]["pred"].flatten(),
                "RF Predicted": st.session_state["rf_result"]["pred"].flatten()
            }, index=st.session_state["lstm_result"]["test_idx"])  # Assume same index
            st.line_chart(combined_df)
        elif "lstm_result" in st.session_state:
            df_lstm = pd.DataFrame({
                "Actual": st.session_state["lstm_result"]["actual"].flatten(),
                "LSTM Predicted": st.session_state["lstm_result"]["pred"].flatten()
            }, index=st.session_state["lstm_result"]["test_idx"])
            st.line_chart(df_lstm)
        elif "rf_result" in st.session_state:
            df_rf = pd.DataFrame({
                "Actual": st.session_state["rf_result"]["actual"].flatten(),
                "RF Predicted": st.session_state["rf_result"]["pred"].flatten()
            }, index=st.session_state["rf_result"]["test_idx"])
            st.line_chart(df_rf)


        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

        # Show 7 day future predictions from each model
        st.markdown("### 🔮 30-Day Future Forecast")
        col1, col2 = st.columns(2)
        future_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=30)

        if "lstm_result" in st.session_state:
            with col1:
                st.subheader("📈 LSTM Forecast")
                forecast_lstm = pd.DataFrame(
                    st.session_state["lstm_result"]["forecast"],
                    columns=["Predicted Close"],
                    index=future_dates
                )
                st.line_chart(forecast_lstm)
                st.dataframe(forecast_lstm.style.format({"Predicted Close": "{:.2f}"}))

        if "rf_result" in st.session_state:
            with col2:
                st.subheader("🌲 Random Forest Forecast")
                forecast_rf = pd.DataFrame(
                    st.session_state["rf_result"]["forecast"],
                    columns=["Predicted Close"],
                    index=future_dates
                )
                st.line_chart(forecast_rf)
                st.dataframe(forecast_rf.style.format({"Predicted Close": "{:.2f}"}))



# Saved Models Page
elif page == "💾 Saved Models":
    st.subheader("📂 Your Saved Models")
    models = list_user_models(user)
    st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    # List models with delete/rename UI
    for m in models:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(m)
        with col2:
            if st.button("❌ Delete", key=f"del_{m}"):
                delete_model_file(user, m)
                st.rerun()
        with col3:
            new_name = st.text_input(f"Rename '{m}'", key=f"rename_{m}")
            if st.button("✅ Rename", key=f"rn_btn_{m}"):
                rename_model_file(user, m, new_name)
                st.rerun()
        st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

    # Model selection dropdown
    selected_model = st.selectbox("🔁 Load a model to reuse", options=models)

    if selected_model and st.button("Load Model"):
        future_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=30)

        # Flags for conditional loading
        has_lstm = False
        has_rf = False

        # Try load LSTM
        try:
            model_lstm, meta_lstm = load_lstm_model(user, selected_model)
            has_lstm = True
            if "params" in meta_lstm:
                for key, val in meta_lstm["params"].items():
                    st.session_state[key] = val
        except Exception:
            pass

        # Try load RF
        try:
            model_rf, meta_rf = load_rf_model(user, selected_model)
            has_rf = True
            if "params" in meta_rf:
                for key, val in meta_rf["params"].items():
                    st.session_state[key] = val
        except Exception:
            pass


        # Dynamically assign column layout based on which models are loaded
        if has_lstm and has_rf:
            col1, col2 = st.columns(2)
        elif has_lstm:
            col1 = st.container()
            col2 = None
        elif has_rf:
            col1 = None
            col2 = st.container()
        else:
            st.warning("No saved models found for this entry.")
            col1 = col2 = None

        if has_lstm or has_rf:
            st.subheader("📊 Combined Predictions vs Actual (Loaded Model)")
            
            combined_data = {}
            index = None

            if has_lstm:
                combined_data["Actual"] = meta_lstm["actual"].flatten()
                combined_data["LSTM Predicted"] = meta_lstm["pred"].flatten()
                index = meta_lstm["test_idx"]  # Base index from LSTM

            if has_rf:
                if not has_lstm:
                    combined_data["Actual"] = meta_rf["actual"].flatten()
                    index = meta_rf["test_idx"]
                combined_data["RF Predicted"] = meta_rf["pred"].flatten()

            df_combined = pd.DataFrame(combined_data, index=index)
            st.line_chart(df_combined)

            # Display 30-Day Forecast
            st.markdown("### 🔮 30-Day Future Forecast")
            col1, col2 = st.columns(2)
            future_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=30)

            if has_lstm:
                with col1:
                    st.subheader("📈 LSTM Forecast")
                    forecast_lstm = pd.DataFrame(
                        meta_lstm["forecast"],
                        columns=["Predicted Close"],
                        index=future_dates
                    )
                    st.line_chart(forecast_lstm)
                    st.dataframe(forecast_lstm.style.format({"Predicted Close": "{:.2f}"}))

            if has_rf:
                with col2:
                    st.subheader("🌲 Random Forest Forecast")
                    forecast_rf = pd.DataFrame(
                        meta_rf["forecast"],
                        columns=["Predicted Close"],
                        index=future_dates
                    )
                    st.line_chart(forecast_rf)
                    st.dataframe(forecast_rf.style.format({"Predicted Close": "{:.2f}"}))


        
# Footer
st.markdown("""
<hr style="margin-top: 50px;">

<div style='text-align: center; color: grey; font-size: 14px; padding: 10px 0;'>
    Made by Alvyn Shibu 🪅 · © 2025
</div>
""", unsafe_allow_html=True)