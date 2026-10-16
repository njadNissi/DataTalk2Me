import re
import socket
import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import storage, firestore
import firebase_admin.credentials as credentials
import bcrypt
import base64
import gzip
from datetime import datetime, timedelta, timezone


# ------------------------------
# Global State
# ------------------------------
_FIREBASE_INITIALIZED = False
_DB = None
_USERS_REF = None
_MESSAGES_REF = None
_CHALLENGES_REF = None

# ------------------------------
# Firestore Initialization
# ------------------------------
def init_firestore():
    global _FIREBASE_INITIALIZED, _DB, _USERS_REF, _MESSAGES_REF, _CHALLENGES_REF
    if _FIREBASE_INITIALIZED:
        return
    try:
        if not firebase_admin._apps:
            firebase_creds = dict(st.secrets["firebase"])
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
        _DB = firestore.client()
        _USERS_REF = _DB.collection("DATA-TALK2ME.USERS")
        print(_USERS_REF)
        _MESSAGES_REF = _DB.collection("DATA-TALK2ME.MESSAGES")
        _CHALLENGES_REF = _DB.collection("DATA-TALK2ME.CHALLENGES")
        _FIREBASE_INITIALIZED = True
        print("> Firestore initialized successfully.")
    except Exception as e:
        print(f"Firestore init error: {e}")
        _USERS_REF = None
        _MESSAGES_REF = None


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
# Update user purposes (original logic)
# ------------------------------
@st.cache_data(ttl=60)
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

        
@st.cache_data(ttl=60)
def set_user_visibility(email, is_visible):
    init_firestore()
    email = email.strip()
    if not _USERS_REF:
        return True

    try:
        user_doc = _USERS_REF.document(email).get()
        if not user_doc.exists:
            print("> Set visibility failed: Email not found")
            return False

        _USERS_REF.document(email).update({
            "is_visible": is_visible
        })

        # Update session
        user_data = user_doc.to_dict()
        user_data["is_visible"] = is_visible
        st.session_state.user_data = user_data
        print(f"> Updated visibility for {email}: {is_visible}")
        return True
    except Exception as e:
        print(f"> Set visibility error: {str(e)}")
        return False


# Get all active users (visible to others)
@st.cache_data(ttl=60)
def get_active_users():
    init_firestore()
    if not _USERS_REF:
        return []

    try:
        active_users = []
        for user_doc in _USERS_REF.stream():
            user_data = user_doc.to_dict()
            # Get only users who are visible to others
            if user_data.get("is_visible", True):
                active_users.append({
                    "user_name": user_data.get("user_name", "N/A"),
                    "email": user_data.get("email", "N/A"),
                    "last_use": user_data.get("last_use", "N/A"),
                    "usage_purpose": user_data.get("usage_purpose", [])
                })
        return active_users
    except Exception as e:
        print(f"> Get active users error: {str(e)}")
        return []

# Get all users (cached for optimization)
@st.cache_data(ttl=60)
def get_all_users():
    init_firestore()
    if not _USERS_REF:
        return []

    try:
        all_users = []
        for user_doc in _USERS_REF.stream():
            user_data = user_doc.to_dict()
            all_users.append({
                "user_name": user_data.get("user_name", "N/A"),
                "email": user_data.get("email", "N/A"),
                "last_use": user_data.get("last_use", "N/A"),
                "usage_purpose": user_data.get("usage_purpose", []),
                "is_visible": user_data.get("is_visible", True)
            })
        return all_users
    except Exception as e:
        print(f"> Get all users error: {str(e)}")
        return []
          
@st.cache_data(ttl=60)
def get_recent_contacts(user_email, limit=5):
    """Retrieves a list of unique users recently contacted by the current user."""
    init_firestore()
    if not _MESSAGES_REF:
        return []

    try:
        # Query messages sent by or received by the user
        sent_query = _MESSAGES_REF.where("sender_email", "==", user_email).stream()
        recv_query = _MESSAGES_REF.where("recipient_email", "==", user_email).stream()
        
        interaction_map = {} # email -> latest timestamp

        def process_stream(stream, is_sender):
            for doc in stream:
                data = doc.to_dict()
                timestamp = data.get("timestamp")
                
                # Determine the *other* person's details
                other_email = data.get("recipient_email") if is_sender else data.get("sender_email")
                other_name = data.get("recipient_name") if is_sender else data.get("sender_name")
                
                if other_email == user_email:
                    continue
                
                if other_email not in interaction_map or (timestamp and timestamp > interaction_map[other_email]["timestamp"]):
                    interaction_map[other_email] = {
                        "email": other_email,
                        "user_name": other_name or other_email,
                        "timestamp": timestamp
                    }

        process_stream(sent_query, True)
        process_stream(recv_query, False)

        # Sort by most recent timestamp
        sorted_contacts = sorted(
            interaction_map.values(), 
            key=lambda x: x.get("timestamp") or 0, 
            reverse=True
        )

        # Return just the user objects up to the limit
        return [{
            "email": c["email"],
            "user_name": c["user_name"],
            "usage_purpose": [] # Can be populated if needed
        } for c in sorted_contacts[:limit]]

    except Exception as e:
        print(f"> Get recent contacts error: {str(e)}")
        return []
    
# ===================== MESSAGES =====================

def init_messages_ref():
    global _MESSAGES_REF
    init_firestore() # Uses your existing initialization function


def send_message(recipient_email, recipient_name, sender_email, sender_name, message_text, file_data=None, file_name=None):
    """Sends a message with an optional compressed file attachment to another user."""
    if not _MESSAGES_REF:
        init_messages_ref()
    
    if not _MESSAGES_REF:
        print("> Send message error: _MESSAGES_REF is not initialized.")
        return False

    try:
        message_payload = {
            "sender_email": sender_email,
            "sender_name": sender_name,
            "recipient_email": recipient_email,
            "recipient_name": recipient_name,
            "message_text": message_text,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "read": False
        }

        # Compress file data with gzip to shrink its size under the 1MB Firestore limit
        if file_data and file_name:
            compressed_bytes = gzip.compress(file_data)
            message_payload["file_name"] = file_name
            message_payload["file_bytes"] = base64.b64encode(compressed_bytes).decode('utf-8')

        _MESSAGES_REF.add(message_payload)
        return True
    except Exception as e:
        print(f"> Send message error: {str(e)}")
        return False


@st.cache_data(ttl=60)
def get_messages_for_user(user_email):
    """Retrieves unread or recent messages for the given user."""
    if not _MESSAGES_REF:
        init_messages_ref()
        
    if not _MESSAGES_REF:
        init_messages_ref()

    try:
        user_messages = []
        # Query messages where recipient_email matches the current user
        query = _MESSAGES_REF.where("recipient_email", "==", user_email).stream()
        
        for msg_doc in query:
            msg_data = msg_doc.to_dict()
            user_messages.append({
                "id": msg_doc.id,
                "sender_email": msg_data.get("sender_email", "N/A"),
                "message_text": msg_data.get("message_text", ""),
                "timestamp": msg_data.get("timestamp", "N/A"),
                "read": msg_data.get("read", False)
            })
        return user_messages
    except Exception as e:
        print(f"> Get messages error: {str(e)}")
        return []


def get_conversation_history(user_a, user_b):
    """Retrieves conversation history and strips file attachments older than 7 weeks."""
    init_firestore()
    if not _MESSAGES_REF:
        return []

    try:
        messages = []
        seven_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=7)
        
        # Helper to process and filter docs
        def process_docs(query_stream):
            for doc in query_stream:
                data = doc.to_dict()
                data["id"] = doc.id
                
                # Check expiration for file attachments
                timestamp = data.get("timestamp")
                if timestamp and data.get("file_bytes"):
                    # Convert Firestore timestamp to python datetime if needed
                    if hasattr(timestamp, "timestamp"):
                        msg_time = datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc)
                        
                        # If older than 7 weeks, remove the heavy file bytes
                        if msg_time < seven_weeks_ago:
                            data["file_bytes"] = None
                            data["file_expired"] = True
                            
                messages.append(data)

        # Query both directions
        process_docs(_MESSAGES_REF.where("sender_email", "==", user_a).where("recipient_email", "==", user_b).stream())
        process_docs(_MESSAGES_REF.where("sender_email", "==", user_b).where("recipient_email", "==", user_a).stream())

        # Sort chronologically
        messages.sort(key=lambda x: x.get("timestamp") or 0)
        return messages
    except Exception as e:
        print(f"> Get conversation history error: {str(e)}")
        return []

        
def mark_message_as_read(message_id):
    """Updates the message document in Firestore to mark it as read."""
    init_firestore()
    if not _MESSAGES_REF:
        return False
    try:
        _MESSAGES_REF.document(message_id).update({"read": True})
        return True
    except Exception as e:
        print(f"> Mark message read error: {str(e)}")
        return False

        

# ====================== C H A L L E N G E S ====================
def db_create_challenge(challenge_data):
    if not _CHALLENGES_REF:
        # Offline fallback to session state
        if "challenges" not in st.session_state:
            st.session_state["challenges"] = []
        st.session_state["challenges"].append(challenge_data)
        return True
    try:
        _CHALLENGES_REF.document(challenge_data["id"]).set(challenge_data)
        return True
    except Exception as e:
        st.error(f"Error creating challenge in Firestore: {e}")
        return False


def db_get_challenges():
    if not _CHALLENGES_REF:
        return st.session_state.get("challenges", [])
    try:
        challenges = []
        for doc in _CHALLENGES_REF.stream():
            data = doc.to_dict()
            # Convert timestamp back if needed
            challenges.append(data)
        return challenges
    except Exception as e:
        print(f"Error fetching challenges: {e}")
        return []

def db_update_challenge(challenge_id, update_dict):
    if not _CHALLENGES_REF:
        challenges = st.session_state.get("challenges", [])
        for c in challenges:
            if c["id"] == challenge_id:
                c.update(update_dict)
        return True
    try:
        _CHALLENGES_REF.document(challenge_id).update(update_dict)
        return True
    except Exception as e:
        st.error(f"Error updating challenge: {e}")
        return False
