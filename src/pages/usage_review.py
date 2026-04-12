import streamlit as st
from src.core.auth import update_user_purposes
# --------------------------
# STEP 2: USAGE REVIEW
# --------------------------
purposes = [
        "Automatic data preprocessing",
        "Data visualization & plotting",
        "Descriptive analysis",
        "Regression modeling",
        "Classification modeling",
        "Clustering",
        "Quick AI prototyping",
        "Education/learning",
        "Research",
        "Professional work"
    ]

def render():
    global purposes

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

    st.markdown("### 📝 Please select your usage purposes to help us suggest you better!")
    cols = st.columns(2)
    for i, p in enumerate(purposes):
        with cols[i % 2]:
            checked = p in prev_usage
            if st.checkbox(p, value=checked):
                selected.append(p)

    if st.button("✅ Save", type="primary"):
        update_user_purposes(user_data["email"], selected)