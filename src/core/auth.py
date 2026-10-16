# src/core/auth.py
from datetime import datetime
import streamlit as st
import firebase_admin
from firebase_admin import firestore
from src.core.database import init_firestore

# Initialize Firestore reference (using your existing init logic)
def handle_google_user_sync(user_data):
    """
    Mirrors the Firestore logic in LoginGoogle.tsx:
    Checks if user exists by UID/Email in DATA-TALK2ME.USERS.
    If new -> creates document.
    If exists -> updates lastLogin / use_times.
    """
    init_firestore()
    uid = user_data.get("uid")
    email = user_data.get("email", "no-email@datatalk2me.com")
    name = user_data.get("name", "DataTalk2Me Member")
    
    # Target document by UID or Email
    doc_id = uid if uid else email
    user_ref = _USERS_REF.document(doc_id)
    user_doc = user_ref.get()

    now_iso = datetime.now().isoformat()
    today_str = datetime.today().strftime("%Y-%m-%d")

    if not user_doc.exists:
        # Create new member record (mirroring CCFCMember structure)
        member_data = {
            "uid": uid,
            "email": email,
            "user_name": name,
            "role": "Member",
            "first_use": today_str,
            "last_use": today_str,
            "use_times": 1,
            "is_visible": True,
            "usage_purpose": []
        }
        user_ref.set(member_data)
        st.session_state.user_data = member_data
    else:
        # Existing member: Update last login metrics
        existing_data = user_doc.to_dict()
        print(f"> Existing user found: {existing_data}")
        use_times = existing_data.get("use_times", 0) + 1
        
        update_data = {
            "is_visible": True,
            "last_use": today_str,
            "use_times": use_times
        }
        user_ref.update(update_data)
        
        existing_data.update(update_data)
        st.session_state.user_data = existing_data

    st.session_state.logged_in = True
    return True