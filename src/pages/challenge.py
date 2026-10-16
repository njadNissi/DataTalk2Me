import streamlit as st
from datetime import datetime, timedelta
import base64
import gzip
import pickle
import pandas as pd
import numpy as np
from src.core.database import db_get_challenges, db_create_challenge, db_update_challenge
import src.core.utils as utils


def render():
    # 1. Retrieve current user from session state
    user_data = st.session_state.get("user_data", None)
    if not user_data:
        st.warning("⚠️ Please log in to access Developer Challenges.")
        return

    st.markdown("## 🏆 Developer Challenges & Bounties")
    st.caption("Create technical challenges, offer workspace Cadeaux as prizes, and test your peers' code solutions!")

    # Tabs for navigation
    tab_active, tab_create, tab_manage = st.tabs(["🔥 Active Challenges", "➕ Create Challenge", "📋 My Challenges & Judging"])

    # Fetch challenges from Firestore (with session state fallback)
    challenges = db_get_challenges()

    # Available purposes/interests list (synchronized with settings.py)
    purposes = utils.SKILLS_INTERESTS

    # ==========================================
    # TAB 1: ACTIVE CHALLENGES (View & Submit)
    # ==========================================
    with tab_active:
        st.markdown("### Ongoing Challenges")
        
        # Filter active & valid challenges
        now = datetime.now()
        active_challenges = []
        for c in challenges:
            valid_until = c.get("valid_until")
            if isinstance(valid_until, str):
                try:
                    valid_until = datetime.fromisoformat(valid_until)
                except:
                    valid_until = now + timedelta(days=1)
            
            if valid_until > now and c.get("status") == "Active":
                active_challenges.append(c)

        if not active_challenges:
            st.info("No active challenges right now. Be the first to create one!")
        else:
            # Pagination for Active Challenges
            items_per_page = 5
            total_items = len(active_challenges)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            
            page_col1, page_col2 = st.columns([2, 4])
            with page_col1:
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="active_page_num")
            
            start_idx = (current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            paginated_active = active_challenges[start_idx:end_idx]

            st.caption(f"Showing challenges {start_idx + 1}–{min(end_idx, total_items)} of {total_items}")

            for challenge in paginated_active:
                with st.expander(f"📌 {challenge['title']} (By: {challenge['creator_name']}) — [{challenge.get('interest', 'General')}]"):
                    st.write(f"**Description:** {challenge['description']}")
                    st.write(f"🎯 **Answer Type:** {challenge['answer_type']}")
                    interests_str = ", ".join(challenge.get("interests", [challenge.get("interest", "General")]))
                    st.write(f"🏷️ **Interests:** {interests_str}")
                    
                    # Display Prize Cadeau info if attached
                    prize_cadeau = challenge.get("prize_cadeau_bytes")
                    if prize_cadeau:
                        st.success("🎁 **Prize:** Includes an attached Workspace Cadeau bundle!")
                        try:
                            p_bytes = gzip.decompress(base64.b64decode(prize_cadeau))
                            st.download_button(
                                label="📥 Download Prize Cadeau Preview",
                                data=p_bytes,
                                file_name=f"prize_cadeau_{challenge['id']}.pkl",
                                mime="application/octet-stream",
                                key=f"dl_prize_{challenge['id']}"
                            )
                        except:
                            pass
                    else:
                        st.write(f"🎁 **Prize:** {challenge.get('prize_desc', 'Bragging Rights')}")
                    
                    st.write(f"⏱️ **Expires:** {str(challenge.get('valid_until'))[:16]}")

                    # Submission Form per challenge
                    with st.form(key=f"submit_form_{challenge['id']}_{user_data['email']}"):
                        st.markdown("#### Submit Your Solution")
                        
                        sub_text = st.text_area("Python Code Solution (Define a function named `solution`)", placeholder="def solution(a, b):\n    return b, a")
                        
                        attach_my_cadeau = st.checkbox("🎁 Attach my current workspace as Cadeau solution")
                        
                        submitted = st.form_submit_button("🚀 Submit Entry")
                        
                        if submitted:
                            sub_cadeau_bytes = None
                            if attach_my_cadeau:
                                cadeau_payload = {
                                    "type": "environment_cadeau",
                                    "data": st.session_state.get("data"),
                                    "feature_analysis": st.session_state.get("feature_analysis"),
                                    "preprocessing": st.session_state.get("preprocessing"),
                                    "inference": st.session_state.get("inference"),
                                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                serialized = pickle.dumps(cadeau_payload)
                                sub_cadeau_bytes = base64.b64encode(gzip.compress(serialized)).decode('utf-8')

                            submission_entry = {
                                "user_email": user_data['email'],
                                "user_name": user_data['user_name'],
                                "text": sub_text,
                                "cadeau_bytes": sub_cadeau_bytes,
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "status": "Pending Review"
                            }

                            if challenge['answer_type'] == "Definite":
                                try:
                                    expected_assertion = pickle.loads(base64.b64decode(challenge['expected_answer_pickled']))
                                    namespace = {"pd": pd, "np": np}
                                    
                                    exec(sub_text, namespace)
                                    
                                    if "solution" not in namespace:
                                        raise Exception("Your code must define a function named `solution`.")
                                    
                                    test_passed = eval(expected_assertion, namespace)
                                    
                                    if test_passed is True:
                                        submission_entry["status"] = "Accepted (Correct Algorithm!)"
                                        challenge["winner"] = user_data['user_name']
                                        challenge["status"] = "Closed"
                                        st.success("🎉 Brilliant! Your algorithm passed all test assertions and won the bounty!")
                                    else:
                                        submission_entry["status"] = "Rejected (Failed Test)"
                                        st.error("❌ Your function executed, but failed the test assertion check.")
                                except Exception as err:
                                    submission_entry["status"] = "Error in execution"
                                    st.error(f"⚠️ Code execution error: {err}")
                            else:
                                st.success("📦 Submission sent successfully! Awaiting creator review.")

                            if "submissions" not in challenge:
                                challenge["submissions"] = []
                            challenge["submissions"].append(submission_entry)
                            
                            db_update_challenge(challenge['id'], {
                                "submissions": challenge["submissions"],
                                "status": challenge["status"],
                                "winner": challenge.get("winner")
                            })
                            st.rerun()

    # ==========================================
    # TAB 2: CREATE A CHALLENGE
    # ==========================================
    with tab_create:
        st.markdown("### Post a New Challenge")
        
        answer_type = st.selectbox(
            "Answer Verification Type", 
            ["Indefinite (Manual Cadeau Review)", "Definite (Python Object Comparison)"],
            key="create_challenge_answer_type"
        )

        with st.form("create_challenge_form"):
            title = st.text_input("Challenge Title", placeholder="e.g., Write a function to swap two variables")
            description = st.text_area("Problem Description & Instructions", placeholder="Describe the problem and requirements for `solution()`...")
            
            # Allow selecting multiple interests/domains for the challenge
            challenge_interests = st.multiselect(
                "Challenge Interests / Domains (Select one or more)", 
                purposes,
                default=[purposes[0]] if purposes else []
            )
            
            validity_days = st.slider("Validity Period (Days)", min_value=1, max_value=30, value=7)

            expected_obj_str = ""
            if "Definite" in answer_type:
                expected_obj_str = st.text_area("Python Test Assertion Expression", placeholder="solution(5, 10) == (10, 5) and solution('a', 'b') == ('b', 'a')")

            st.markdown("#### 🎁 Prize Configuration")
            prize_desc = st.text_input("Prize Description", placeholder="e.g., Advanced ML feature pipeline Cadeau")
            attach_prize_cadeau = st.checkbox("🎁 Attach current workspace as prize Cadeau bundle")

            create_btn = st.form_submit_button("📢 Publish Challenge")

            if create_btn:
                if not title or not description:
                    st.error("Please fill in the title and description.")
                else:
                    prize_cadeau_bytes = None
                    if attach_prize_cadeau:
                        cadeau_payload = {
                            "type": "environment_cadeau",
                            "data": st.session_state.get("data"),
                            "feature_analysis": st.session_state.get("feature_analysis"),
                            "preprocessing": st.session_state.get("preprocessing"),
                            "inference": st.session_state.get("inference"),
                            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        serialized = pickle.dumps(cadeau_payload)
                        prize_cadeau_bytes = base64.b64encode(gzip.compress(serialized)).decode('utf-8')

                    expected_pickled = None
                    if "Definite" in answer_type:
                        if not expected_obj_str.strip():
                            st.error("❌ Please provide a test assertion expression.")
                            return
                        expected_pickled = base64.b64encode(pickle.dumps(expected_obj_str)).decode('utf-8')

                    challenge_id = str(datetime.now().timestamp())
                    new_challenge = {
                        "id": challenge_id,
                        "creator_email": user_data['email'],
                        "creator_name": user_data['user_name'],
                        "title": title,
                        "description": description,
                        "interests": challenge_interests,
                        "answer_type": "Definite" if "Definite" in answer_type else "Indefinite",
                        "expected_answer_pickled": expected_pickled,
                        "prize_desc": prize_desc if prize_desc else "Bragging Rights",
                        "prize_cadeau_bytes": prize_cadeau_bytes,
                        "valid_until": (datetime.now() + timedelta(days=validity_days)).isoformat(),
                        "status": "Active",
                        "winner": None,
                        "submissions": []
                    }
                    
                    if db_create_challenge(new_challenge):
                        st.success("🎉 Challenge published successfully to Firestore!")
                        st.rerun()

    # ==========================================
    # TAB 3: MY CHALLENGES & JUDGING
    # ==========================================
    with tab_manage:
        st.markdown("### 📋 Challenge Management & Judging")
        
        mp_col1, mp_col2 = st.columns([2, 4])
        with mp_col1:
            # Toggle: Only mine vs Matched by my purposes
            only_mine = st.toggle("📌 Only mine", value=True, help="When ON, show only challenges created by you. When OFF, show challenges matching your interests.")

            user_purposes = user_data.get("usage_purpose", [])

            if only_mine:
                my_win = user_data["user_name"] + f" ({user_data['email']})"
                filtered_challenges = [c for c in challenges if c.get("creator_email") == user_data['email'] or c.get("winner") == my_win]
            else:
                filtered_challenges = [
                    c for c in challenges 
                    if any(interest in user_purposes for interest in c.get("interests", []))
                ]
        # Pagination for Managed Challenges
        manage_items_per_page = 5
        total_m_items = len(filtered_challenges)
        total_m_pages = max(1, (total_m_items + manage_items_per_page - 1) // manage_items_per_page)
        
        with mp_col2:
            current_m_page = st.number_input("Manage Page", min_value=1, max_value=total_m_pages, value=1, step=1, key="manage_page_num")
        
        m_start_idx = (current_m_page - 1) * manage_items_per_page
        m_end_idx = m_start_idx + manage_items_per_page
        paginated_manage = filtered_challenges[m_start_idx:m_end_idx]

        st.caption(f"Showing challenges {m_start_idx + 1}–{min(m_end_idx, total_m_items)} of {total_m_items}")

        for challenge in paginated_manage:
            with st.expander(f"🛠️ {challenge['title']} (Status: {challenge['status']}) — [{challenge.get('interest', 'General')}]"):
                st.write(f"**Description:** {challenge['description']}")
                st.write(f"🎯 **Type:** {challenge['answer_type']}")
                interests_str = ", ".join(challenge.get("interests", [challenge.get("interest", "General")]))
                st.write(f"🏷️ **Interests:** {interests_str}")

                if challenge.get("winner"):
                    st.success(f"🏆 Winner declared: {challenge['winner']}")

                st.markdown("#### Submissions Received:")
                submissions = challenge.get("submissions", [])
                
                if not submissions:
                    st.caption("No submissions yet.")
                else:
                    for idx, sub in enumerate(submissions):
                        st.markdown(f"**From:** `{sub['user_name']}` ({sub['user_email']}) — *{sub['status']}*")
                        st.caption(f"Submitted at: {sub['timestamp']}")
                        
                        # Beautiful Python code formatted syntax highlighting
                        st.code(sub['text'], language="python")

                        # If submission includes an attached Cadeau bundle
                        if sub.get("cadeau_bytes"):
                            try:
                                sub_bytes = gzip.decompress(base64.b64decode(sub['cadeau_bytes']))
                                st.download_button(
                                    label=f"📥 Download {sub['user_name']}'s Workspace Cadeau",
                                    data=sub_bytes,
                                    file_name=f"cadeau_sub_{sub['user_email']}.pkl",
                                    mime="application/octet-stream",
                                    key=f"dl_sub_cadeau_{challenge['id']}_{idx}"
                                )
                            except:
                                pass

                        # Manual review actions for indefinite answers
                        if challenge['answer_type'] == "Indefinite" and challenge['status'] == "Active":
                            col_acc, col_rej = st.columns(2)
                            with col_acc:
                                if st.button("✅ Accept & Declare Winner", key=f"acc_{challenge['id']}_{idx}"):
                                    challenge['submissions'][idx]['status'] = "Accepted (Winner!)"
                                    challenge['winner'] = sub['user_name']
                                    challenge['status'] = "Closed"
                                    
                                    db_update_challenge(challenge['id'], {
                                        "submissions": challenge['submissions'],
                                        "winner": sub['user_name'] + f" ({sub['user_email']})",
                                        "status": "Closed"
                                    })
                                    st.success(f"Winner declared: {sub['user_name']}!")
                                    st.rerun()
                            with col_rej:
                                if st.button("❌ Reject", key=f"rej_{challenge['id']}_{idx}"):
                                    challenge['submissions'][idx]['status'] = "Rejected"
                                    db_update_challenge(challenge['id'], {"submissions": challenge['submissions']})
                                    st.rerun()
                        st.divider()