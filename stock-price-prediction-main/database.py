# Imports
import sqlite3
import os
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import save_model as keras_save_model


# SQLite Database Setup
# Connect to SQLite database file 'users.db', allow access from different threads
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()


# Create a table for storing user credentials if it doesn't already exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )
""")
conn.commit()

# Create a table for activity logging
cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        timestamp TEXT,
        username TEXT,
        action TEXT
    )
""")
conn.commit()



# User Authentication Functions

# Create a new user with hashed password
def create_user(username, hashed_pw, is_admin=False):
    cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, hashed_pw, int(is_admin)))
    conn.commit()


def is_admin_user(username):
    cursor.execute("SELECT is_admin FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    return row is not None and row[0] == 1


# Verify login credentials (username & hashed password match)
def verify_user(username, hashed_pw):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    return cursor.fetchone() is not None


# Check if the username already exists in the database
def user_exists(username):
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    return cursor.fetchone() is not None


# Update the stored password for a given username
def update_password(username, new_hashed_pw):
    cursor.execute("UPDATE users SET password=? WHERE username=?", (new_hashed_pw, username))
    conn.commit()


# Model Storage

# Get or create the user's model directory
def get_user_dir(user):
    path = os.path.join("models", user)
    os.makedirs(path, exist_ok=True)
    return path


# Save both the LSTM model and associated metadata (scaler, predictions, etc.)
def save_model_file(user, model, scaler, stock, model_name, actual=None, pred=None, test_idx=None, forecast=None, params=None):
    user_dir = get_user_dir(user)

    # Save model differently depending on type
    if isinstance(model, BaseEstimator):
        joblib.dump(model, os.path.join(user_dir, f"{model_name}_RF.joblib"))
    else:
        keras_save_model(model, os.path.join(user_dir, f"{model_name}_LSTM.h5"))

    # Save metadata (same for both)
    suffix = "_RF" if isinstance(model, BaseEstimator) else "_LSTM"
    joblib.dump({
        "scaler": scaler,
        "stock": stock,
        "actual": actual,
        "pred": pred,
        "test_idx": test_idx,
        "forecast": forecast,
        "params": params or {}
    }, os.path.join(user_dir, f"{model_name}{suffix}_meta.pkl"))



# Load the LSTM and associated metadata
def load_lstm_model(user, model_name):
    user_dir = get_user_dir(user)
    keras_path = os.path.join(user_dir, f"{model_name}_LSTM.h5")
    keras_meta = os.path.join(user_dir, f"{model_name}_LSTM_meta.pkl")
    if os.path.exists(keras_path) and os.path.exists(keras_meta):
        model = keras_load_model(keras_path)
        meta = joblib.load(keras_meta)
        return model, meta
    else:
        raise FileNotFoundError("LSTM model files not found.")


# Load the RF model and associated metadata
def load_rf_model(user, model_name):
    user_dir = get_user_dir(user)
    rf_path = os.path.join(user_dir, f"{model_name}_RF.joblib")
    rf_meta = os.path.join(user_dir, f"{model_name}_RF_meta.pkl")
    if os.path.exists(rf_path) and os.path.exists(rf_meta):
        model = joblib.load(rf_path)
        meta = joblib.load(rf_meta)
        return model, meta
    else:
        raise FileNotFoundError("Random Forest model files not found.")


# List all saved model names for a given user
def list_user_models(user):
    user_dir = get_user_dir(user)
    model_names = set()
    for f in os.listdir(user_dir):
        if f.endswith("_LSTM.h5"):
            model_names.add(f.replace("_LSTM.h5", ""))
        elif f.endswith("_RF.joblib"):
            model_names.add(f.replace("_RF.joblib", ""))
    return sorted(model_names)


#Delete both the model file and its metadata
def delete_model_file(user, model_name):
    user_dir = get_user_dir(user)
    files_to_delete = [
        f"{model_name}_LSTM.h5",
        f"{model_name}_LSTM_meta.pkl",
        f"{model_name}_RF.joblib",
        f"{model_name}_RF_meta.pkl",
    ]
    for filename in files_to_delete:
        path = os.path.join(user_dir, filename)
        if os.path.exists(path):
            os.remove(path)


# Rename both the model file and its associated metadata
def rename_model_file(user, old_name, new_name):
    user_dir = get_user_dir(user)
    suffixes = ["_LSTM.h5", "_LSTM_meta.pkl", "_RF.joblib", "_RF_meta.pkl"]
    for suffix in suffixes:
        old_path = os.path.join(user_dir, f"{old_name}{suffix}")
        new_path = os.path.join(user_dir, f"{new_name}{suffix}")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)



# Admin Features

# Get all users (for admin view)
def get_all_users():
    cursor.execute("SELECT username, is_admin FROM users ORDER BY username ASC")
    return cursor.fetchall()

# Update a user's admin status
def set_admin_status(username, is_admin):
    cursor.execute("UPDATE users SET is_admin=? WHERE username=?", (int(is_admin), username))
    conn.commit()

# Delete a user
def delete_user(username):
    cursor.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()

# Get user models
def get_all_user_models():
    base_dir = "models"
    result = {}
    if not os.path.exists(base_dir):
        return result

    for user in os.listdir(base_dir):
        user_dir = os.path.join(base_dir, user)
        if os.path.isdir(user_dir):
            models = set()
            for f in os.listdir(user_dir):
                if f.endswith("_LSTM.h5"):
                    models.add(f.replace("_LSTM.h5", ""))
                elif f.endswith("_RF.joblib"):
                    models.add(f.replace("_RF.joblib", ""))
            result[user] = sorted(models)
    return result


# Activity log features

# Log Events
def log_event(username, action):
    ts = datetime.now().isoformat(timespec='seconds')
    cursor.execute("INSERT INTO logs (timestamp, username, action) VALUES (?, ?, ?)", (ts, username, action))
    conn.commit()

# Get logged events
def get_logs(limit=100):
    cursor.execute("SELECT timestamp, username, action FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    return cursor.fetchall()

