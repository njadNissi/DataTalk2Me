import streamlit as st
from src.core.auth import get_total_users, login_user, update_user_purposes, db, is_email_real

# ------------------------------
# SESSION STATE
# ------------------------------
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

# ------------------------------
# LOGIN FORM
# ------------------------------
def render():
    # 🔥 SKIP LOGIN IN DEV MODE
    if st.session_state.logged_in:
        st.success("✅ Auto-logged in")
        return
    
    st.title("🔐 Let data talk to Me. - Login")
    st.markdown(f"""
        <div style="text-align:center; font-size:18px; margin:15px 0;">
        Welcome to join <span style="color:#1f77b4; font-size:28px; font-weight:bold;">👥</span> data mates around the world 🌍
    </div>
    """, unsafe_allow_html=True)

    email = st.text_input(
        "Email Address",
        autocomplete="email"
    )

    if st.button("Continue", type="primary"):
        if not is_email_real(email):
            st.warning("Please fill valid email.")
        else:
            # --------------------------
            # SEXY DANCING CAT LOADER
            # --------------------------
            loader = st.empty()
            with loader:
                st.markdown("""
                    <div style="text-align:center; padding: 30px 0;">
                        <h3 style="color:#666;">🔐 Logging you in...</h3>
                        <img src="https://i.giphy.com/media/3o7qE1YN7aBOFPRw8E/giphy.gif" 
                             width="200" style="border-radius: 14px;">
                        <p style="font-size:17px; color:#ff4b4b; margin-top:10px;">
                           ✨ Just a moment ✨
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # --------------------------
            # LOGIN (ALREADY SETS user_data)
            # --------------------------
            login_user(email.split("@")[0], email, [])

            # HIDE LOADER
            loader.empty()

            # FINISH
            st.session_state.show_usage_step = True
            st.rerun()

# ------------------------------
# SIDEBAR USER INFO
# ------------------------------
def show_user_info():
    u = st.session_state.get("user_data")
    if not u or not isinstance(u, dict):
        return

    st.sidebar.markdown(f"### 👤 {u.get('user_name', 'User')}")
    st.sidebar.caption(f"Logins: {u.get('use_times', 0)}")
    usage = u.get("usage_purpose", [])
    st.sidebar.caption(f"For: {', '.join(usage[:3])}{'...' if len(usage) > 3 else ''}")
    st.sidebar.markdown("---")

# ------------------------------
# LOGOUT
# ------------------------------
def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_data = None
        st.session_state.show_usage_step = False
        st.rerun()