import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import streamlit as st

# --------------------------
# AUTH FIRST (ONLY ONE CONFIG CALL)
# --------------------------
st.set_page_config(
    page_title="Data-Talk2Me",
    page_icon="data/logos/logo_orig.png",
    layout="wide"
)

from src.pages import (
    upload, data_plotting, inference, login,
    preprocessing, feature_eng_analysis, settings, connect, challenge
)

# 🔥 DEV MODE AUTO-DETECTION
DEV_MODE = "--dev" in sys.argv 

if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

if DEV_MODE:
    st.session_state.logged_in = True
    st.session_state.dev_mode = True
    st.session_state.user_data = {
        "user_name": "dev_user",
        "email": "dev@test.com",
        "use_times": 999,
        "first_use": "2025-01-01",
        "last_use": "2025-01-01",
        "usage_purpose": ["Development"]
    }

# Initialize authentication session state variables
login.init_auth()

# If not logged in, render the login page and stop execution here
if not st.session_state.get("logged_in", False):
    login.render()
    st.stop()  

# --------------------------
# APP STARTS HERE (AFTER LOGIN)
# --------------------------
with st.sidebar:
    st.image("data/logos/logo_nobg.png", width=200)
    st.title("Data-Talk2Me")

st.sidebar.markdown("---")
# Show user info & logout button in sidebar
login.show_user_info()

pages = [
    "Upload Data",
    "Feature Analysis",
    "Data Plotting",
    "Preprocessing",
    "Inference",
    "Connect",
    "Challenge",
    "Settings"
]

if "page" not in st.session_state:
    st.session_state["page"] = "Upload Data"

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state["page"]) if st.session_state["page"] in pages else 0
)
st.session_state["page"] = page

if page == "Upload Data":
    upload.render()
elif page == "Feature Analysis":
    feature_eng_analysis.render()
elif page == "Data Plotting":
    data_plotting.render()
elif page == "Preprocessing":
    preprocessing.render()
elif page == "Inference":
    inference.render()
elif page == "Connect":
    connect.render()
elif page == "Challenge":
    challenge.render()
elif page == "Settings":
    settings.render()


st.sidebar.markdown("---")
st.sidebar.markdown("**Author**")
st.sidebar.markdown('<span style="font-size: 12px;">Joao Andre Ndombasi *Diakusala*</span>', unsafe_allow_html=True)
login.logout()