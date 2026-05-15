import streamlit as st
from src.core.auth import get_total_users, login_user, register_user, update_user_purposes, _DB, is_email_real

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
# LOGIN & REGISTER FORMS
# ------------------------------
# ------------------------------
# LOGIN & REGISTER FORMS
# ------------------------------
def render():
    # 🔥 SKIP LOGIN IN DEV MODE
    if st.session_state.logged_in:
        st.success("✅ Auto-logged in")
        return
   
    logo_bar = st.columns(7)
    with logo_bar[3]:
        st.image("data/logos/logo_nobg.png", width=200)  # Adjust width as needed
    
    # Centered Welcome Text
    st.markdown(f"""
        <div style="text-align:center; font-size:18px; margin:10px 0;">
        Welcome to join 33k data mates around the 🌍
    </div>
        <div style="text-align:center; font-size:24px; margin:15px 0;">
        Let's Personalize Your Experience
    </div>
    """, unsafe_allow_html=True)
    
    # 🔥 PERFECT CLEAN CENTERED CARD (TABS INSIDE)
    st.markdown("""
        <style>
        /* This holds everything: login + register tabs INSIDE */
        div[data-testid="stTabs"] {
            max-width: 440px !important;
            margin: 0 auto !important;
            # background: black;
            padding: 25px 28px !important;
            border-radius: 16px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.08);
        }

        /* Fix tab button layout */
        div[data-testid="stTab"] {
            font-weight: 500;
        }

        /* Fix inner spacing */
        div[data-testid="stVerticalBlock"] {
            gap: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # TABS ARE NOW AUTOMATICALLY INSIDE THE WHITE CARD
    login_tab, register_tab = st.tabs(["🔑 Login", "📝 Register"])
    
    # --------------------------
    # LOGIN TAB
    # --------------------------
    with login_tab:
        with st.form(key="login_form", clear_on_submit=False):
            email = st.text_input(
                "Email Address",
                autocomplete="email",
                key="login_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
                key="login_password"
            )

            # 🔥 CHANGE TO form_submit_button
            submit_login = st.form_submit_button(
                "Login", 
                type="primary", 
                use_container_width=True
            )

        if submit_login:
            if not is_email_real(email):
                st.warning("Please fill valid email.")
            elif len(password) < 8:
                st.warning(f"len={len(password)} | Password must be at least 8 characters!")
            else:
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

                logged_in = login_user(email, password)
                loader.empty()

                if logged_in:
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Login failed. Please check your email and password if you have previously registered. If not, please register first!")
    # --------------------------
    # REGISTER TAB
    # --------------------------
    with register_tab:
    # 🔥 ADD FORM WRAPPER (FIXES STREAMLIT BUG)
        with st.form(key="register_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                autocomplete="username",
                key="register_username"
            )
            email = st.text_input(
                "Email Address",
                autocomplete="email",
                key="register_email"
            )
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="new-password",
                key="register_password"
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                autocomplete="new-password",
                key="register_confirm_password"
            )

            # 🔥 CHANGE TO form_submit_button
            submit_register = st.form_submit_button(
                "Register", 
                type="primary", 
                use_container_width=True
            )

        if submit_register:
            if not username:
                st.warning("Please enter a username!")
            elif not is_email_real(email):
                st.warning("Please fill valid email!")
            elif len(password) < 8:
                st.warning("Password must be at least 8 characters!")
            elif password != confirm_password:
                st.warning("Passwords do not match!")
            else:
                loader = st.empty()
                with loader:
                    st.markdown("""
                        <div style="text-align:center; padding: 30px 0;">
                            <h3 style="color:#666;">📝 Creating your account...</h3>
                            <img src="https://i.giphy.com/media/3ohhwHhY4bjsEZwq9K/giphy.gif" 
                                width="200" style="border-radius: 14px;">
                            <p style="font-size:17px; color:#ff4b4b; margin-top:10px;">
                            ✨ Just a moment ✨
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                register_user(username, email, password)
                loader.empty()

                st.session_state.show_usage_step = True
                st.success("✅ Account created successfully!")
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

# ------------------------------
# LOGOUT
# ------------------------------
def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_data = None
        st.session_state.show_usage_step = False
        st.rerun()