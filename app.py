import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import streamlit as st
from src.pages import upload, visualize_2d, visualize_3d, inference, scaling, feature_eng_analysis
from src.core.auth import init_auth, login_form, show_user_info, logout, check_auto_login

# --------------------------
# AUTH FIRST
# --------------------------
st.set_page_config(layout="wide", page_title="Data Analysis Suite")
init_auth()
check_auto_login()

if not st.session_state.logged_in:
    login_form()
    st.stop()  # STOP APP HERE UNTIL LOGIN

# --------------------------
# YOUR ORIGINAL APP STARTS HERE (AFTER LOGIN)
# --------------------------
st.sidebar.title("📊 Data Talk to Me")

# --------------------------
# SHOW USER INFO + USERNAME IN SIDEBAR
# --------------------------
show_user_info()

# --------------------------
# ✅ SHOW USER FULL NAME IN SIDEBAR
# --------------------------
user = st.session_state.get("user_data", {})
full_name = user.get("full_name", "User")
st.sidebar.markdown(f"### 👤 Welcome, **{full_name}**")

pages = [
    "Upload Data",
    "Feature Analysis",
    "2D Visualization",
    "3D Visualization",
    "Scaling",
    "Inference"
]

if "page" not in st.session_state:
    st.session_state["page"] = "Upload Data"

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state["page"])
)

st.session_state["page"] = page

if "data" not in st.session_state:
    st.session_state.data = None

st.sidebar.subheader("📂 Load Previous Analysis")

uploaded_file = st.sidebar.file_uploader(
    "Choose a .pkl file",
    type=["pkl"]
)
if uploaded_file is not None:
    try:
        loaded_results = pickle.load(uploaded_file)
        st.session_state["analysis_results"] = loaded_results
        st.success("✅ Analysis loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")

if page == "Upload Data":
    upload.render()
elif page == "Feature Analysis":
    feature_eng_analysis.render()
elif page == "2D Visualization":
    visualize_2d.render()
elif page == "3D Visualization":
    visualize_3d.render()
elif page == "Scaling":
    scaling.render()
elif page == "Inference":
    inference.render()

# --------------------------
# ✅ LOGOUT BUTTON
# --------------------------
st.sidebar.markdown("---")
logout()

st.sidebar.markdown("---")
st.sidebar.markdown('<span style="font-size: 11px;">Author:\nJoao Andre Ndombasi *Diakusala*</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">GitHub:\nhttps://github.com/njadNissi</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">🔗:\nhttps://bwania-solution.netlify.app/</span>', unsafe_allow_html=True)
st.sidebar.markdown('<span style="font-size: 12px;">📧:\nnjadnissi@gmail.com</span>', unsafe_allow_html=True)