# src/core/auth.py
import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import firestore
from google.cloud.firestore_v1.client import Client
import firebase_admin.credentials as credentials

# ------------------------------
# 🔥 FORCE INIT FIRESTORE WITH YOUR PROJECT
# ------------------------------
def init_firestore():
    import os
    try:
        if not firebase_admin._apps:
            # 1. Try Streamlit Secrets (CLOUD)
            if "firebase" in st.secrets:
                cred_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_dict)
            # 2. Fallback to local file (LOCAL DEV)
            else:
                json_path = os.path.join("src", "auth", "firebase-service-account.json")
                cred = credentials.Certificate(json_path)
        
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Connect error: {str(e)}")
        return None

# ------------------------------
# INIT DB & CORRECT COLLECTION NAME
# ------------------------------
db = init_firestore()
users_ref = db.collection("DATA-TALK2ME.USERS") if db else None

# ------------------------------
# SESSION STATE
# ------------------------------
def init_auth():
    defaults = {
        "logged_in": False,
        "user_data": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ------------------------------
# COUNT USERS
# ------------------------------
def get_total_users():
    try:
        if not users_ref:
            return 0
        return len(list(users_ref.stream()))
    except:
        return 0

# ------------------------------
# LOGIN / CREATE / UPDATE USER
# ------------------------------
def login_or_update_user(full_name, email, purposes):
    global users_ref
    if not users_ref:
        st.warning("Running offline mode — data not saved")
        st.session_state.user_data = {
            "full_name": full_name,
            "email": email,
            "use_times": 1,
            "first_use": "2025-01-01",
            "last_use": "2025-01-01",
            "usage_purpose": ", ".join(purposes)
        }
        return True

    today = datetime.today().strftime("%Y-%m-%d")
    purpose_str = ", ".join(purposes)

    try:
        doc = users_ref.document(email).get()

        if doc.exists:
            data = doc.to_dict()
            new_count = data["use_times"] + 1
            users_ref.document(email).update({
                "use_times": new_count,
                "last_use": today,
                "usage_purpose": purpose_str
            })
            user_data = {
                "full_name": full_name,
                "email": email,
                "use_times": new_count,
                "first_use": data["first_use"],
                "last_use": today,
                "usage_purpose": purpose_str
            }
        else:
            user_data = {
                "full_name": full_name,
                "email": email,
                "use_times": 1,
                "first_use": today,
                "last_use": today,
                "usage_purpose": purpose_str
            }
            users_ref.document(email).set(user_data)

        st.session_state.user_data = user_data
        return True
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
        return False

# ------------------------------
# LOGIN FORM
# ------------------------------
def login_form():
    total = get_total_users() + 777

    st.title("🔐 Data Analysis Suite - Login")
    st.markdown(f"""
        <div style="text-align:center; font-size:18px; margin:15px 0;">
        Welcome to join <span style="color:#1f77b4; font-size:28px; font-weight:bold;">{total}👥</span>  data mates around the world 🌍
    </div>
    """, unsafe_allow_html=True)

    full_name = st.text_input("Full Name")
    email = st.text_input("Email Address")

    st.markdown("### What will you use this tool for?")
    purposes = [
        "Automatic data preprocessing",
        "Data visualization & plotting",
        "Descriptive analysis",
        "Regression modeling",
        "Classification modeling",
        "Clustering",
        "Quick AI prototyping",
        "Education/learning",
        "Research",
        "Professional work"
    ]

    selected = []
    cols = st.columns(2)
    for i, p in enumerate(purposes):
        with cols[i % 2]:
            if st.checkbox(p):
                selected.append(p)

    if st.button("Continue", type="primary"):
        if not full_name or not email or "@" not in email:
            st.warning("Please fill name and valid email.")
        elif len(selected) == 0:
            st.warning("Select at least one purpose.")
        else:
            login_or_update_user(full_name, email, selected)
            st.session_state.logged_in = True
            st.rerun()

# ------------------------------
# SIDEBAR USER INFO
# ------------------------------
def show_user_info():
    if not st.session_state.get("user_data"):
        return
    u = st.session_state.user_data
    st.sidebar.markdown(f"### 👤 {u['full_name']}")
    st.sidebar.caption(f"Logins: {u['use_times']} | Last: {u['last_use']}")
    st.sidebar.caption(f"For: {u['usage_purpose']}")
    st.sidebar.markdown("---")

# ------------------------------
# LOGOUT
# ------------------------------
def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_data = None
        st.rerun()

        
def check_auto_login():
    if st.session_state.get("user_data"):
        st.session_state.logged_in = True