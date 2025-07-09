# Imports
import streamlit as st
import hashlib
import re
from database import create_user, verify_user, user_exists, update_password, is_admin_user, log_event

# Initialize Session State
if "user" not in st.session_state:
    st.session_state["user"] = None

# Hash the password using SHA-256 for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if user is authenticated
def is_authenticated():
    return st.session_state["user"] is not None

# Get the currently logged in user
def get_current_user():
    return st.session_state["user"]

# Display login, registration and password reset interface
def show_login_signup():
    st.title("📈 Using LSTM To Forecast Stock Trends - Secure, Interactive and Fast.")
    st.markdown("### 🔐 Please log in or register to continue.")

    # Tabbed interface for Login/Register/Reset
    tab_login, tab_register, tab_reset, tab_admin_login = st.tabs(["Login", "Register", "Reset Password", "Admin Login"])

    # Login Tab
    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if verify_user(username, hash_password(password)):
                if is_admin_user(username):
                    st.error("❌ Admins must log in using the Admin Login tab.")
                else:
                    st.session_state["user"] = username
                    st.session_state["is_admin"] = False
                    log_event(username, "User logged in")
                    st.success("✅ Logged in successfully!")
                    st.rerun()
            else:
                st.error("❌ Invalid username or password.")


    # Register Tab
    with tab_register:
        new_username = st.text_input("New Username", key="register_user")
        new_password = st.text_input("New Password", type="password", key="register_pass")
        if st.button("Register"):
            if user_exists(new_username):
                st.warning("⚠️ That username already exists.")
            elif len(new_password) < 6:
                st.warning("⚠️ Password must be at least 6 characters long.")
            elif not re.search(r"\d", new_password):
                st.warning("⚠️ Password must contain at least one number.")
            else:
                create_user(new_username, hash_password(new_password))
                st.success("✅ Registration successful! Please log in.")
                st.rerun()


    # Reset Password Tab
    with tab_reset:
        reset_user = st.text_input("Username", key="reset_user")
        reset_pass = st.text_input("New Password", type="password", key="reset_pass")
        if st.button("Reset Password"):
            if not user_exists(reset_user):
                st.error("❌ Username not found.")
            elif len(reset_pass) < 6:
                st.warning("⚠️ Password must be at least 6 characters long.")
            elif not re.search(r"\d", reset_pass):
                st.warning("⚠️ Password must contain at least one number.")
            else:
                update_password(reset_user, hash_password(reset_pass))
                st.success("✅ Password reset successfully.")
    

    # Admin Login Tab
    with tab_admin_login:
        admin_username = st.text_input("Admin Username", key="admin_user")
        admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
        if st.button("Admin Login"):
            if verify_user(admin_username, hash_password(admin_password)):
                if is_admin_user(admin_username):
                    st.session_state["user"] = admin_username
                    st.session_state["is_admin"] = True
                    log_event(admin_username, "Admin logged in")
                    st.success("✅ Admin login successful!")
                    st.rerun()
                else:
                    st.error("❌ This account is not an admin.")
            else:
                st.error("❌ Invalid admin credentials.")


