import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import pickle
import streamlit as st
st.set_page_config(page_title="Talk2Me", layout="wide")
from src.pages import (
    upload, data_plotting, inference, login,
    preprocessing, feature_eng_analysis,usage_review
)
import sys

# --------------------------
# AUTH FIRST
# --------------------------
st.set_page_config(layout="wide", page_title="Let data talk to Me.")

# 🔥 DEV MODE AUTO-DETECTION
DEV_MODE = "--dev" in sys.argv # RUN as: reamlit run app.py -- --dev (-- → tells Streamlit: “all args after this are for MY app, not for Streamlit”)

# DEV USER (auto-login)
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
# ===============================================================================

login.init_auth()
login.check_auto_login()

if not st.session_state.logged_in:
    login.render()
    st.stop()  # STOP APP HERE UNTIL LOGIN

# --------------------------
# APP STARTS HERE (AFTER LOGIN)
# --------------------------
st.sidebar.title("� Data Talk to Me")

# --------------------------
# SHOW USER INFO + USERNAME IN SIDEBAR
# --------------------------
login.show_user_info()
# --------------------------
# ✅ LOGOUT BUTTON
# --------------------------
login.logout()
st.sidebar.markdown("---")

pages = [
    "Upload Data",
    "Feature Analysis",
    "Data Plotting",
    "Preprocessing",
    "Inference",
    "Usage Review"
]

if "page" not in st.session_state:
    st.session_state["page"] = "Upload Data"
page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state["page"])
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
elif page == "Usage Review":
    usage_review.render()

st.sidebar.markdown("---")
st.sidebar.markdown('<span style="font-size: 11px;">Author:\nJoao Andre Ndombasi *Diakusala*</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">GitHub:\nhttps://github.com/njadNissi</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">🔗:\nhttps://bwania-solution.netlify.app/</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">📧:\nnjadnissi@gmail.com</span>', unsafe_allow_html=True)