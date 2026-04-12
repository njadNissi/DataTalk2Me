# src/core/auth.py
import re
import socket
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
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets: # 1. Try Streamlit Secrets (LOCAL/CLOUD)
                cred_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_dict)
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
print(f"=======================\nDatabase {db} initialized. Users collection: {users_ref}\n===============")


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
# ------------------------------
# 1. LOGIN + CREATE USER (TOGETHER)
# ------------------------------
def login_user(user_name, email, purposes):
    global users_ref
    today = datetime.today().strftime("%Y-%m-%d")

    # Offline mode
    if not users_ref:
        user_data = {
            "user_name": user_name,
            "email": email,
            "use_times": 1,
            "first_use": today,
            "last_use": today,
            "usage_purpose": purposes
        }
        st.session_state.user_data = user_data
        return True

    try:
        doc = users_ref.document(email).get()

        # USER EXISTS → LOGIN
        if doc.exists:
            user_data = doc.to_dict()
            user_data["use_times"] += 1
            user_data["last_use"] = today
            users_ref.document(email).update({
                "use_times": user_data["use_times"],
                "last_use": today,
            })
            st.session_state.user_data = user_data
            return True

        # USER DOES NOT EXIST → CREATE NEW USER
        else:
            user_data = {
                "user_name": user_name,
                "email": email,
                "use_times": 1,
                "first_use": today,
                "last_use": today,
                "usage_purpose": purposes
            }
            users_ref.document(email).set(user_data)
            st.session_state.user_data = user_data
            return True

    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False


# ------------------------------
# 2. SEPARATE: UPDATE USER PURPOSES + LOGIN COUNT
# ------------------------------
def update_user_purposes(email, purposes):
    global users_ref
    today = datetime.today().strftime("%Y-%m-%d")

    if not users_ref:
        return True

    try:
        doc = users_ref.document(email).get()
        if not doc.exists:
            return False

        data = doc.to_dict()
        new_count = data["use_times"] + 1

        # UPDATE ONLY THESE FIELDS
        users_ref.document(email).update({
            "use_times": new_count,
            "last_use": today,
            "usage_purpose": purposes
        })

        # UPDATE SESSION STATE
        user_data = data.copy()
        user_data["use_times"] = new_count
        user_data["last_use"] = today
        user_data["usage_purpose"] = purposes
        st.session_state.user_data = user_data

        return True

    except Exception as e:
        st.error(f"Update failed: {str(e)}")
        return False



def is_email_real(email: str) -> bool:
    email = email.strip()
    # 1. Format check
    if not re.fullmatch(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False

    # 2. Extract domain
    domain = email.split('@')[1]

    # 3. Check if domain has valid MX (mail server) records
    try:
        socket.gethostbyname_ex(domain)
        return True
    except socket.gaierror:
        return False