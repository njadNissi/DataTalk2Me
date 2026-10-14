# src/pages/settings.py
import streamlit as st
import pickle
import pandas as pd
from src.core.auth import update_user_purposes, set_user_visibility, get_active_users
import src.core.utils as utils
import gzip

def render():

    purposes = [
        # Original
        "✅ Automatic data preprocessing",
        "📊 Data visualization & plotting",
        "🔍 Descriptive analysis",
        "📈 Regression modeling",
        "🏷️ Classification modeling",
        "🧩 Clustering",
        "🧪 Quick AI prototyping",
        "📚 Education/learning",
        "🔬 Research",
        "💼 Professional work",

        # Data & image & general analysis
        "🧹 Data cleaning & filtering",
        "🖼️ Image conversion & processing",
        "📄 PDF extraction & analysis",
        "📑 Excel/CSV data manipulation",
        "🔎 Exploratory data analysis (EDA)",
        "📐 Statistical analysis & reporting",
        "⏱️ Time series analysis",
        "😊 Sentiment analysis",
        "⚠️ Anomaly & outlier detection",
        "🔗 Correlation & pattern analysis",
        "📋 Automated reporting",
        "🔄 Data transformation & formatting",
        "⚙️ Feature engineering",
        "🤖 Model training & evaluation",
        "🔮 Prediction & forecasting",
        "📝 Document analysis & summarization",
        "✨ Photo enhancement & editing",
        "⚡ Batch file processing",
        "🕸️ Data extraction from files",
        "📏 Data normalization & scaling",
        "📊 Survey & form data analysis",
        "📈 Business intelligence",
        "✅ Data quality checking",
        "📊 Visual report generation",
        "🧪 Machine learning testing",
        "🏷️ Data annotation & labeling",
        "🔗 Data merging & joining",
        "🧪 Data sampling & subset creation",
        "📈 Automated chart generation",
        "💡 Insight & conclusion generation",

        # Sound / audio manipulation
        "🎵 Audio signal processing & filtering",
        "🔄 Sound file format conversion",
        "📶 Audio feature extraction (MFCC, spectral)",
        "🗣️ Speech-to-text transcription & analysis",
        "🔇 Noise reduction & audio cleaning",
        "✂️ Audio segmentation & trimming",
        "📈 Sound visualization (waveforms, spectrograms)",
        "🗣️ Voice analysis & emotion detection",
        "🎚️ Audio mixing & amplitude normalization",
        "📡 Acoustic data analysis & pattern detection",

        # ✅ NEW 20: Programming & Data Structures Learning
        "☕ Java fundamentals & practice",
        "🐍 Python programming learning",
        "🌐 JavaScript & web coding",
        "🧱 C/C++ algorithm development",
        "📱 Kotlin mobile development learning",
        "💻 C# & .NET application learning",
        "🧱 Data structures implementation",
        "📊 Algorithm design & analysis",
        "🔢 Sorting & searching algorithms practice",
        "🌳 Tree & graph algorithms learning",
        "🧩 Object-oriented programming (OOP)",
        "🔗 Linked lists, stacks & queues learning",
        "🗄️ Database & SQL query learning",
        "🔁 Recursion & dynamic programming practice",
        "⚡ Time & space complexity analysis",
        "🧪 Coding exercises & challenges",
        "📝 Pseudocode & logic design",
        "🧠 Memory & performance optimization",
        "🔍 Debugging & code analysis",
        "🤝 Collaborative coding projects"
    ]

    # SAFETY CHECK: IF NO USER DATA → GO BACK
    user_data = st.session_state.get("user_data")
    if not user_data:
        st.rerun()
        return

    use_times = user_data.get("use_times", 1)
    prev_usage = user_data.get("usage_purpose", [])

    # Show My personal infos
    st.markdown("### 🧑‍💼 My Personal Information")
    cols = st.columns(6)
    is_visible = user_data.get("is_visible", True)
    with cols[0]:
        st.write(f"**Name:** {user_data.get('user_name', 'N/A')}")
    with cols[1]:
        st.write(f"**Email:** {user_data.get('email', 'N/A')}")
    with cols[2]:
        st.write(f"**First Use:** {user_data.get('first_use', 'N/A')}")
    with cols[3]:
        st.write(f"**Last Use:** {user_data.get('last_use', 'N/A')}")
    with cols[4]:
        st.write(f"**Login Times:** {use_times}")
    with cols[5]:
        # show a toggle button to set my status visible or invisible to other users
        is_visible = st.toggle("Make me visible", value=is_visible)
        set_user_visibility(user_data["email"], is_visible)

    # --- BUILD CADEAU SECTION ---
    st.markdown("### 🎁 Build Workspace Cadeau")
    st.caption("Package your current dataset, feature analysis, preprocessing, and inference settings into a single downloadable workspace file.")
    
    with st.expander("🛠️ Cadeau Builder", expanded=True):
        # Quick summary check of what is currently loaded in session state
        has_data = st.session_state.get("data") is not None
        has_fa = st.session_state.get("feature_analysis") is not None
        has_prep = st.session_state.get("preprocessing") is not None and has_data
        has_inf = st.session_state.get("inference") is not None

        cols = st.columns(4)
        with cols[0]:
            st.write(f"- 📊 Dataset Loaded: {'✅ Yes' if has_data else '❌ No'}")
        with cols[1]:
            st.write(f"- 🔬 Feature Analysis Results: {'✅ Yes' if has_fa else '❌ No'}")
        with cols[2]:
            st.write(f"- ⚙️ Preprocessing Settings: {'✅ Yes' if has_prep else '❌ No'}")
        with cols[3]:
            st.write(f"- 🔮 Inference Settings: {'✅ Yes' if has_inf else '❌ No'}")

        if st.button("🎁 Generate Cadeau Package", type="primary"):
            cadeau_payload = {
                "type": "environment_cadeau",
                "data": st.session_state.get("data"),
                "feature_analysis": st.session_state.get("feature_analysis"),
                "preprocessing": st.session_state.get("preprocessing"),
                "inference": st.session_state.get("inference"),
                "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            serialized_bytes = pickle.dumps(cadeau_payload) 
            cadeau_bytes = gzip.compress(serialized_bytes) 
            cadeau_filename = f"cadeau_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            st.success("✅ Cadeau package generated successfully!")
            st.download_button(
                label="📥 Download Generated Cadeau (.pkl)",
                data=cadeau_bytes,
                file_name=cadeau_filename,
                mime="application/octet-stream"
            )

    # SHOW REVIEW ON FIRST LOGIN OR EVERY 10 LOGINS
    show_review = (use_times == 1 or use_times % 10 == 0)
    selected = []

    with st.expander("📂 Select Skills & Interests", expanded=True):
        COLS = 3
        cols = st.columns(COLS)
        for i, p in enumerate(purposes):
            with cols[i % COLS]:
                checked = p in prev_usage
                if st.checkbox(p, value=checked):
                    selected.append(p)

        if st.button("💾 Confirm & Save", type="primary"):
            if not st.session_state.dev_mode:
                update_user_purposes(user_data["email"], selected)
                utils.temp_show("✅ Your choices updated successfully...", 'success', 1.0)