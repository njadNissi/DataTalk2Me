# src/core/auth.py
import re
import socket
import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import firestore
import firebase_admin.credentials as credentials
import bcrypt

# ========================================
def hash_password(password: str) -> str:
    """Convert plain password to secure hash (store this in DB)"""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if plain password matches the stored hash"""
    try:
        pwd_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)
    except:
        return False
# ========================================

# ------------------------------
# Global State
# ------------------------------
_FIREBASE_INITIALIZED = False
_DB = None
_USERS_REF = None

# ------------------------------
# Firestore Initialization
# ------------------------------
def init_firestore():
    global _FIREBASE_INITIALIZED, _DB, _USERS_REF
    if _FIREBASE_INITIALIZED:
        return
    try:
        if not firebase_admin._apps:
            firebase_creds = dict(st.secrets["firebase"])
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
        _DB = firestore.client()
        _USERS_REF = _DB.collection("DATA-TALK2ME.USERS")
        _FIREBASE_INITIALIZED = True
        print("> Firestore initialized successfully.")
    except Exception as e:
        print(f"Firestore init error: {e}")
        _USERS_REF = None

# ------------------------------
# Email Validation
# ------------------------------
def is_email_real(email):
    email = email.strip()
    # Format check
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.fullmatch(pattern, email):
        return False
    # Domain check
    try:
        domain = email.split('@')[1]
        socket.gethostbyname_ex(domain)
        return True
    except:
        return False

# ------------------------------
# Get total users
# ------------------------------
def get_total_users():
    init_firestore()
    if not _USERS_REF:
        return 0
    try:
        return len(list(_USERS_REF.stream()))
    except:
        return 0

# ------------------------------
# LOGIN FUNCTION (MODULAR - ONLY LOGIN)
# ------------------------------
def login_user(email, password):
    """
    Only for LOGIN: email + password
    Returns True if success, False if failed
    """
    init_firestore()
    today = datetime.today().strftime("%Y-%m-%d")
    email = email.strip()

    # Offline fallback
    if not _USERS_REF:
        st.session_state.user_data = {
            "user_name": "Offline User",
            "email": email,
            "password": password,
            "use_times": 1,
            "first_use": today,
            "last_use": today,
            "usage_purpose": []
        }
        print("> Offline login: No database connection")
        return True

    try:
        print(f"> Attempting login for: {email} and {'*' * len(password)}")
        
        user_doc = _USERS_REF.document(email).get()

        if not user_doc.exists:
            print("> Login failed: Email not found")
            return False

        user_data = user_doc.to_dict()

        if not verify_password(password, user_data.get("password")):
            print("> Login failed: Wrong password")
            return False

        # Update login stats
        use_times = user_data["use_times"] + 1
        _USERS_REF.document(email).update({
            "use_times": use_times,
            "last_use": today
        })

        user_data["use_times"] = use_times
        user_data["last_use"] = today
        st.session_state.user_data = user_data

        return True

    except Exception as e:
        st.error(f"Login error: {str(e)}")
        print(f"> Login error: {str(e)}")
        return False

# ------------------------------
# REGISTER FUNCTION (MODULAR - ONLY REGISTER)
# ------------------------------
def register_user(username, email, password):
    """
    Only for REGISTER: username + email + password
    Pre-check: user must NOT exist already
    Returns True if success, False if failed
    """
    init_firestore()
    today = datetime.today().strftime("%Y-%m-%d")
    email = email.strip()
    password = hash_password(password)  # Hash the password before storing

    # Offline fallback
    if not _USERS_REF:
        st.session_state.user_data = {
            "user_name": username,
            "email": email,
            "password": password,
            "use_times": 1,
            "first_use": today,
            "last_use": today,
            "usage_purpose": []
        }
        print("> Offline register: No database connection")
        return True

    try:
        user_doc = _USERS_REF.document(email).get()
        if user_doc.exists:
            st.error("❌ Email already registered")
            print("> Register failed: Email already registered")
            return False

        # Create new user
        new_user = {
            "user_name": username,
            "email": email,
            "password": password,
            "use_times": 1,
            "first_use": today,
            "last_use": today,
            "usage_purpose": []
        }
        _USERS_REF.document(email).set(new_user)
        st.session_state.user_data = new_user
        return True

    except Exception as e:
        st.error(f"Register error: {str(e)}")
        print(f"Register error: {str(e)}")
        return False

# ------------------------------
# Update user purposes (original logic)
# ------------------------------
def update_user_purposes(email, purposes):
    init_firestore()
    today = datetime.today().strftime("%Y-%m-%d")
    email = email.strip()
    if not _USERS_REF:
        return True

    try:
        user_doc = _USERS_REF.document(email).get()
        if not user_doc.exists:
            print("> Update purposes failed: Email not found")
            return False

        use_times = user_doc.to_dict()["use_times"] + 1
        _USERS_REF.document(email).update({
            "use_times": use_times,
            "last_use": today,
            "usage_purpose": purposes
        })

        # Update session
        user_data = user_doc.to_dict()
        user_data["use_times"] = use_times
        user_data["last_use"] = today
        user_data["usage_purpose"] = purposes
        print(f"> Updated purposes for {email}: {purposes}")
        st.session_state.user_data = user_data
        return True
    except Exception as e:
        print(f"> Update purposes error: {str(e)}")
        return False