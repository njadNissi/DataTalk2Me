import json
import requests
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------
# Auth Helper Functions (Called by app.py)
# -----------------------------------------------------------------
def init_auth():
    defaults = {
        "logged_in": False,
        "user_data": None,
        "show_usage_step": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def check_auto_login():
    if st.session_state.get("user_data"):
        st.session_state.logged_in = True

def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_data = None
        st.session_state.show_usage_step = False
        st.query_params.clear()
        st.rerun()


def show_user_info():
    user = st.session_state.get("user_data", {})
    if user:
        picture_url = user.get("picture")
        if picture_url:
            st.sidebar.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 8px;">
                    <img src="{picture_url}" style="width: 72px; height: 72px; border-radius: 50%; object-fit: cover; border: 2px solid #dadce0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                </div>
                """,
                unsafe_allow_html=True
            )
        st.sidebar.markdown(f"<div style='text-align: center;'><b>👤 {user.get('user_name', 'User')}</b></div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='text-align: center; font-size: 13px; color: #64748B;'>`{user.get('email', '')}`</div>", unsafe_allow_html=True)
        st.sidebar.markdown("---")

# -----------------------------------------------------------------
# Main Login Page Render & Code Exchange
# -----------------------------------------------------------------
def render():
    print(f">>> ALL URL QUERY PARAMS RECEIVED: {dict(st.query_params)}")

    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #1E3A8A; font-family: sans-serif;">Data-Talk2Me Platform</h1>
            <p style="color: #64748B; font-size: 18px;">Sign in with your Google account to access your messages and challenges.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query_params = st.query_params

    # 1. Handle incoming Authorization Code from Google
    if "code" in query_params:
        code = query_params["code"]
        print(f">>> RECEIVED CODE: {code}")
        try:
            google_client_id = st.secrets["google"]["client_id"]
            google_client_secret = st.secrets["google"]["client_secret"]
            
            # Debug prints to verify secrets and environment match
            print(f">>> CLIENT ID LOADED: {google_client_id[:10]}...")
            print(f">>> CLIENT SECRET LENGTH: {len(google_client_secret) if google_client_secret else 0}")

            host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
            redirect_uri = "http://localhost:8501/" if ("localhost" in host or "127.0.0.1" in host) else "https://congomani.streamlit.app/"
            
            print(f">>> DETECTED HOST: {host}")
            print(f">>> USING REDIRECT URI: {redirect_uri}")

            token_url = "https://oauth2.googleapis.com/token"
            payload = {
                "code": code,
                "client_id": google_client_id,
                "client_secret": google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code"
            }
            
            token_res = requests.post(token_url, data=payload)
            # print(f">>> TOKEN RES STATUS: {token_res.status_code}")
            # print(f">>> TOKEN RES BODY: {token_res.text}")
            
            if token_res.status_code == 200:
                token_data = token_res.json()
                access_token = token_data.get("access_token")
                
                headers = {"Authorization": f"Bearer {access_token}"}
                user_res = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
                
                if user_res.status_code == 200:
                    data = user_res.json()
                    email = data.get("email")
                    user_name = data.get("name")
                    picture = data.get("picture")

                    # Sync user directly with Firestore DATA-TALK2ME.USERS collection
                    import src.core.database as db
                    
                    db.init_firestore()
                    today = datetime.today().strftime("%Y-%m-%d")
                    
                    user_payload = {
                        "email": email,
                        "user_name": user_name,
                        "picture": picture,
                        "last_use": today,
                        "is_visible": True
                    }

                    if db._USERS_REF:
                        print(f">>> SYNCING USER TO FIRESTORE: {email}")
                        try:
                            doc_ref = db._USERS_REF.document(email)
                            doc = doc_ref.get()
                            
                            if doc.exists:
                                existing_data = doc.to_dict()
                                use_times = existing_data.get("use_times", 1) + 1
                                user_payload["use_times"] = use_times
                                user_payload["usage_purpose"] = existing_data.get("usage_purpose", [])
                                doc_ref.update(user_payload)
                            else:
                                user_payload["use_times"] = 1
                                user_payload["first_use"] = today
                                user_payload["usage_purpose"] = []
                                doc_ref.set(user_payload)
                            
                            final_user_data = doc_ref.get().to_dict()
                            print(f">>> SUCCESSFULLY WROTE USER TO FIRESTORE: {email}")
                        except Exception as fs_err:
                            print(f">>> FIRESTORE WRITE ERROR: {fs_err}")
                            final_user_data = user_payload
                    else:
                        print(">>> ERROR: _USERS_REF is None!")
                        final_user_data = user_payload

                    # Store inside session state
                    st.session_state["user_data"] = final_user_data
                    st.session_state["logged_in"] = True

                    # Clear query params and refresh cleanly
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("Failed to fetch user profile details from Google.")
                    st.query_params.clear()
            else:
                st.error(f"Failed to exchange authorization code with Google. ({token_res.text})")
                st.query_params.clear()
        except Exception as e:
            st.error(f"Error during authentication: {e}")
            st.query_params.clear()

    # If already logged in, show success state
    if st.session_state.get("logged_in", False):
        user = st.session_state.get("user_data", {})
        st.success(f"Welcome back, {user.get('user_name', 'User')}!")
        if st.button("Go to Dashboard", use_container_width=True):
            st.rerun()
        return

    # 2. Grab Google Client ID from secrets
    try:
        google_client_id = st.secrets["google"]["client_id"]
    except Exception:
        st.error("Missing `[google]` client_id in your `secrets.toml` file.")
        return

    # 3. Dynamic Redirect URI based on environment
    host = st.context.headers.get("host", "") if hasattr(st, "context") else ""
    if "localhost" in host or "127.0.0.1" in host:
        redirect_uri = "http://localhost:8501/"
    else:
        redirect_uri = "https://congomani.streamlit.app/"

    # 4. Construct Google OAuth URL using response_type=code
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={google_client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope=email%20profile&"
        f"include_granted_scopes=true&"
        f"access_type=offline"
    )

    # 5. Render Beautiful Google Sign-In Button
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-top: 30px;">
            <a href="{google_auth_url}" target="_self" style="
                background-color: #ffffff;
                color: #3c4043;
                padding: 12px 24px;
                text-decoration: none;
                font-size: 16px;
                font-family: 'Google Sans', Roboto, Arial, sans-serif;
                font-weight: 500;
                border-radius: 4px;
                border: 1px solid #dadce0;
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            ">
                <svg width="18" height="18" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.7 1.22 9.19 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.46-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
                Sign in with Google
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )