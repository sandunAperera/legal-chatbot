import streamlit as st
import yaml
import bcrypt
from yaml.loader import SafeLoader

CONFIG_FILE = "config.yaml"

# 🧠 Load existing users
def load_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            return yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error("⚠️ config.yaml not found. Please ensure it's in the same folder.")
        return None

# 💾 Save updated config
def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

# 🔐 Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# 🎨 UI
st.title("📝 User Registration")

config = load_config()
if config is not None:
    name = st.text_input("👤 Full Name")
    username = st.text_input("🆔 Username")
    email = st.text_input("📧 Email")
    password = st.text_input("🔐 Password", type="password")
    confirm = st.text_input("🔁 Confirm Password", type="password")

    if st.button("Register"):
        if not all([name, username, email, password, confirm]):
            st.warning("⚠️ Please fill in all fields.")
        elif password != confirm:
            st.error("❌ Passwords do not match.")
        elif username in config["credentials"]["usernames"]:
            st.error("🚫 Username already exists.")
        else:
            config["credentials"]["usernames"][username] = {
                "name": name,
                "email": email,
                "password": hash_password(password)
            }
            save_config(config)
            st.success(f"✅ {username} registered successfully! You can now log in.")
