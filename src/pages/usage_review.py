# src/pages/usage_review.py

def render():
    import streamlit as st
    from src.core.auth import update_user_purposes
    import src.core.utils as utils

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
    print("Previous usage purposes:", prev_usage)

    # SHOW REVIEW ON FIRST LOGIN OR EVERY 10 LOGINS
    show_review = (use_times == 1 or use_times % 10 == 0)
    selected = []

    btns = st.columns([5, 2, 3])
    with btns[0]:
        st.markdown("### 📝 Please select your usage purposes to help us suggest you better!")
    with btns[1]:
        if st.button("💾 Confirm and save your choices", type="primary"):
            if not st.session_state.dev_mode:
                update_user_purposes(user_data["email"], selected)
                utils.temp_show("✅ Your choices updated successfully...", 'success', 1.0)

    st.divider()

    COLS = 3
    cols = st.columns(COLS)
    for i, p in enumerate(purposes):
        with cols[i % COLS]:
            checked = p in prev_usage
            if st.checkbox(p, value=checked):
                selected.append(p)
    print("Selected usage purposes:", selected)
