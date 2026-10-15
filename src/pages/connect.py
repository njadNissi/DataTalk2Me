import streamlit as st
import base64
import pickle
import io
import pandas as pd
import gzip
from src.core.auth import (
    get_all_users, get_active_users, get_conversation_history, get_recent_contacts,
    send_message, mark_message_as_read
)

def clear_input():
    # Modifying state inside a callback runs BEFORE widgets instantiate in the next rerun
    st.session_state["chat_page_input"] = ""
    
def set_emoji_bar(input_state_key: str = "chat_page_input", emojis: list[str] = None):
    """
    Renders a horizontally sliding emoji toolbar that appends selected 
    symbols directly to a specified Streamlit text input session state key.
    """
    if emojis is None:
        emojis = ["🚀", "💻", "🐛", "🔥", "✅", "📦", "🎁", "🧠", "⚡", "📉", "💡", "🛠️", "✨", "🎯", "📊", "🤖"]

    # 1. Ensure the target session state string is initialized
    if st.session_state.get(input_state_key) is None:
        st.session_state[input_state_key] = ""

    # 2. Inject custom CSS for horizontal scrolling container
    st.markdown(
        """
        <style>
        .emoji-slider-container {
            display: flex;
            overflow-x: auto;
            gap: 6px;
            padding: 4px 2px 8px 2px;
            white-space: nowrap;
            scrollbar-width: thin;
        }
        .emoji-slider-container::-webkit-scrollbar {
            height: 5px;
        }
        .emoji-slider-container::-webkit-scrollbar-thumb {
            background-color: rgba(150, 150, 150, 0.3);
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 3. Render container wrapper
    st.markdown('<div class="emoji-slider-container">', unsafe_allow_html=True)

    # Render buttons horizontally
    cols = st.columns(len(emojis))
    for i, emoji in enumerate(emojis):
        with cols[i]:
            if st.button(emoji, key=f"slide_emoji_{input_state_key}_{i}", help=f"Insert {emoji}"):
                st.session_state[input_state_key] += emoji
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def active_users_bar(active_users, current_user_email, recipient_email):
    for user in active_users:
        u_name = user['user_name']
        u_email = user['email']
        purposes = user.get('usage_purpose', ['General exploration'])
        if isinstance(purposes, str):
            purposes = [purposes]

        # Row layout: Main chat select button on the left, toggle icon button on the right
        col_btn, col_icon = st.columns([5, 1])
        
        with col_btn:
            # Style the button differently if they are currently selected
            is_selected = recipient_email == u_email
            btn_label = f"🟢 {u_name}" if not is_selected else f"👉 {u_name} (Active)"
            
            if st.button(btn_label, key=f"user_select_btn_{u_email}", use_container_width=True):
                st.session_state["active_chat_user"] = {'user_name': u_name, 'email': u_email}
                st.rerun()
                
        with col_icon:
            # Manage toggle state for this specific user's usage purposes
            toggle_key = f"show_purposes_{u_email}"
            is_open = st.session_state.get(toggle_key, False)
            icon_label = "🔼" if is_open else "🔽"
            
            if st.button(icon_label, key=f"toggle_btn_{u_email}", help="Show Skills & Interests"):
                st.session_state[toggle_key] = not is_open
                st.rerun()

        # Foldable single-column table/list showing all usage purposes when toggled open
        if st.session_state.get(f"show_purposes_{u_email}", False):
            st.markdown(
                "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; font-weight: bold;'>Skills & Interests:</div>", 
                unsafe_allow_html=True
            )
            for purpose in purposes:
                st.markdown(
                    f"<div style='background-color: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 4px 8px; border-radius: 4px; margin-bottom: 3px; font-size: 0.85rem;'>{purpose}</div>", 
                    unsafe_allow_html=True
                )
        st.divider()

def unpack_and_apply_cadeau(file_bytes):
    """Unpacks a Cadeau workspace bundle and configures the entire Streamlit session state."""
    try:
        # Automatically decompress gzip bytes if compressed, otherwise fallback to raw bytes
        try:
            decompressed_bytes = gzip.decompress(file_bytes)
        except Exception:
            decompressed_bytes = file_bytes
            
        cadeau_data = pickle.load(io.BytesIO(decompressed_bytes))
        
        if isinstance(cadeau_data, dict) and cadeau_data.get("type") == "environment_cadeau":
            # 1. Restore Dataset
            if cadeau_data.get("data") is not None:
                st.session_state["data"] = cadeau_data["data"]
                
            # 2. Restore Feature Engineering
            if cadeau_data.get("feature_analysis") is not None:
                st.session_state['feature_analysis'] = cadeau_data["feature_analysis"]
                
            # 3. Restore Preprocessing Settings
            if cadeau_data.get("preprocessing") is not None:
                st.session_state['preprocessing'] = cadeau_data["preprocessing"]
                
            # 4. Restore Inference Settings
            if cadeau_data.get("inference") is not None:
                st.session_state["inference"] = cadeau_data["inference"]
                
            return True, "🎁 Cadeau unpacked successfully! Entire environment workspace is now set up."
        else:
            return False, "⚠️ File is a standard report, not an environment Cadeau."
    except Exception as e:
        return False, f"❌ Failed to unpack Cadeau: {str(e)}"

def handle_file_attachment(msg, is_me=False):
    if msg.get("file_name"):
        with st.container(border=True):
            if msg.get("file_expired") or not msg.get("file_bytes"):
                align_style = "text-align: right;" if is_me else "text-align: left;"
                st.markdown(f"<div style='{align_style}'><span style='font-size: 0.85em; color: gray;'>⌛ Attachment ({msg.get('file_name')}) has expired (> 7 weeks).</span></div>", unsafe_allow_html=True)
            else:
                f_name = msg.get("file_name")
                try:
                    raw_bytes = base64.b64decode(msg.get("file_bytes"))
                    
                    # Automatically decompress (supports both compressed and legacy files)
                    try:
                        f_bytes = gzip.decompress(raw_bytes)
                    except Exception:
                        f_bytes = raw_bytes  
                    
                    # Inject CSS to remove vertical padding/margins for buttons inside attachments
                    st.markdown(
                        """
                        <style>
                        div[data-testid="column"] {
                            padding: 0px !important;
                            margin: 0px !important;
                        }
                        .stButton button, .stDownloadButton button {
                            min-height: 32px !important;
                            height: 32px !important;
                            padding-top: 0px !important;
                            padding-bottom: 0px !important;
                            margin-top: 2px !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    col_action, col_open, col_dl = st.columns([7, 1.5, 1.5], gap="small")
                    with col_action:
                        msg_id = msg.get('id')
                        is_read = msg.get('read', False)

                        # Show a noticeable visual tag if it's already open/read
                        if is_read:
                            st.markdown("<div style='padding-top: 6px;'>🎁 **Cadeau** — <span style='color: #28a745; font-weight: bold;'>✅ Opened</span></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='padding-top: 6px;'>🎁 **Cadeau** — <span style='color: #ffc107; font-weight: bold;'>📦 Closed</span></div>", unsafe_allow_html=True)
                    with col_open:
                        if st.button("👁️", key=f"open_cadeau_{msg_id}", help="Unpack and setup entire environment workspace"):
                            success, message = unpack_and_apply_cadeau(f_bytes)
                            if success:
                                # 1. Update Firestore backend
                                mark_message_as_read(msg_id)
                                
                                # 2. Update local session state immediately so it reflects without refetching
                                msg['read'] = True
                                
                                # st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    with col_dl:
                        st.download_button(
                            label="📥",
                            data=f_bytes,
                            file_name=f_name,
                            mime="application/octet-stream",
                            help=f"Download {f_name}", # This shows the full filename on mouse hover!
                            key=f"dl_chat_file_{msg.get('id', f_name)}"
                        )
                except Exception as ex:
                    st.error(f"⚠️ Could not load attachment: {ex}")
                          
@st.fragment(run_every=60)
def render_live_messages(current_user, recipient, session_messages_key):
    # Fetch latest messages inside the fragment loop
    messages = get_conversation_history(current_user['email'], recipient['email'])
    st.session_state[session_messages_key] = messages

    # Scrollable message container
    chat_container = st.container(height=600)
    with chat_container:
        if not messages:
            st.info(f"No messages yet with {recipient['user_name']}. Start the conversation below!")
        else:
            for msg in messages:
                is_me = msg.get("sender_email") == current_user['email']
                if is_me:
                    # Use flexbox alignment to push the inline-block bubble to the right
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #DCF8C6; padding: 10px 14px; border-radius: 12px; max-width: 75%; text-align: left; color: #000; word-wrap: break-word;">
                                <b style="font-size: 0.85em; color: #555;">You</b>  🕒{msg.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if msg.get('timestamp') else ''}
                                <br>{msg.get('message_text', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    _, right_col = st.columns([3, 1])
                    with right_col:
                        handle_file_attachment(msg)
                else:
                    # Push the inline-block bubble to the left
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #F1F0F0; padding: 10px 14px; border-radius: 12px; max-width: 75%; text-align: left; color: #000; word-wrap: break-word;">
                                <b style="font-size: 0.85em; color: #555;">{msg.get('sender_name')}</b>  🕒{msg.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if msg.get('timestamp') else ''}
                                <br>{msg.get('message_text', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    left_col, _ = st.columns([1, 3])
                    with left_col:
                        handle_file_attachment(msg)

def render():
    # Create a two-column layout: Left for the chat, Right for the clickable user list
    chat_col, users_col = st.columns([3, 1], gap="medium")

    current_user = st.session_state.get("user_data")
    recipient = st.session_state.get("active_chat_user")

    # --- LEFT COLUMN: THE ACTIVE CHAT WINDOW ---
    with chat_col:
        if recipient:
            # KEEP HISTORY in LOCAL SESSION STATE for the current chat partner
            chat_partner_email = recipient['email']
            session_messages_key = f"chat_history_{chat_partner_email}"

            # Header with a close/back button for the current chat
            head_col1, head_col2, head_col3 = st.columns([4, 1, 1])
            with head_col1:
                st.markdown(f"### 💬 Connect with {recipient['user_name']}")

            with head_col2:
                if st.button("🔄 Refresh", key=f"refresh_chat_{chat_partner_email}", help="Fetch latest messages from Firestore"):
                    # Clear or force-update the session state history
                    st.session_state[session_messages_key] = get_conversation_history(
                        current_user['email'], 
                        chat_partner_email
                    )
                    st.rerun()  

            with head_col3:
                if st.button("❌ Close", key="close_active_chat"):
                    st.session_state.pop("active_chat_user", None)
                    st.rerun()
            
            # 3. Load from session state (or fetch if it doesn't exist yet)
            if session_messages_key not in st.session_state:
                st.session_state[session_messages_key] = get_conversation_history(
                    current_user['email'], 
                    chat_partner_email
                )

            st.divider()
            render_live_messages(current_user, recipient, session_messages_key) 

            # --------------- A T T A C H M E N T S ---------------
            cols = st.columns([2, 3]) 
            # --- Optional File Attachment Expander ---
            with cols[0]:
                # with st.expander("📎 Attach File or Cadeau", expanded=False):
                uploaded_pkl = st.file_uploader(
                    "📎 Select a report or workspace Cadeau to attach",
                    type=["pkl"],
                    key="chat_pkl_uploader"
                )

            # Horizontally Scrolling Emoji Bar
            with cols[1]:
                # set_emoji_bar()
                pass

            # ------------------------------
            # Chat Input Box at the bottom
            # ------------------------------
            if prompt := st.chat_input(f"Type a message to {recipient['user_name']}...", key="chat_page_input"):
                file_bytes_data = None
                file_name_val = None
                
                # Check if a file was uploaded in the expander
                if uploaded_pkl is not None:
                    file_bytes_data = uploaded_pkl.getvalue()
                    file_name_val = uploaded_pkl.name
                
                # Fallback message text if only a file is sent with empty prompt text
                message_text = prompt if prompt else f"Shared file: {file_name_val}"

                success = send_message(
                    recipient_email=recipient['email'],
                    recipient_name=recipient['user_name'],
                    sender_email=current_user['email'],
                    sender_name=current_user['user_name'],
                    message_text=message_text,
                    file_data=file_bytes_data,
                    file_name=file_name_val
                )
                if success:
                    # Immediately refresh the local session state history so it updates instantly
                    st.session_state[session_messages_key] = get_conversation_history(
                        current_user['email'], 
                        chat_partner_email
                    )
                    # Clear the input state after successful send
                    st.rerun()
                else:
                    st.error("Failed to send message and file.")
            
        else:
            # Empty state when no chat is selected
            st.markdown("### 💬 Connecting with Gigs")
            st.info("👈 Select an active developer from the right-hand panel to open a chat and start messaging.")
    # --- RIGHT COLUMN: CLICKABLE ACTIVE USERS LIST ---
    with users_col:
        st.markdown("### 👥 Developers")
        
        if "dev_search_input" not in st.session_state:
            st.session_state["dev_search_input"] = ""

        search_col, toggle_col = st.columns([3, 1], gap="small")
        
        with search_col:
            search_query = st.text_input(
                "Search developers", 
                placeholder="Search by name, email, skill...", 
                label_visibility="collapsed", 
                key="dev_search_input"
            )
            
        with toggle_col:
            active_only_toggle = st.toggle("🟢", value=True, key="dev_active_only_toggle", help="Active users only")

        st.divider()

        developers_pool = []

        if st.session_state.get("dev_mode", False):
            if active_only_toggle:
                developers_pool = st.session_state.get("active_users", [])
            else:
                developers_pool = st.session_state.get("all_users", st.session_state.get("active_users", []))
        else:
            if active_only_toggle:
                @st.cache_data(ttl=300)
                def cached_get_active_users():
                    return get_active_users()
                developers_pool = cached_get_active_users()
            else:
                @st.cache_data(ttl=300)
                def cached_get_all_users():
                    return get_all_users()
                developers_pool = cached_get_all_users()

        current_query = st.session_state.get("dev_search_input", "").strip()

        current_query = st.session_state.get("dev_search_input", "").strip()

        # Determine which users to display
        if not current_query or len(current_query) < 2:
            # NO SEARCH QUERY: Show your true recently contacted developers!
            display_users = get_recent_contacts(current_user['email'], limit=5)
            
            # Fallback to active pool if you haven't chatted with anyone yet
            if not display_users:
                display_users = developers_pool[:5]
                list_title_caption = "🟢 Active Developers:"
            else:
                list_title_caption = "💬 Recently Contacted:"
        else:
            # WITH SEARCH QUERY: Filter by name, email, or skill tags
            query_lower = current_query.lower()
            filtered_users = []
            
            for u in developers_pool:
                u_name = u.get('user_name', '').lower()
                u_email = u.get('email', '').lower()
                
                purposes = u.get('usage_purpose', [])
                if isinstance(purposes, str):
                    purposes = [p.strip() for p in purposes.split(',')]
                
                skills = u.get('skills', [])
                if isinstance(skills, str):
                    skills = [s.strip() for s in skills.split(',')]
                    
                combined_tags = [str(p).lower() for p in (purposes + skills)]
                
                if query_lower in u_name or query_lower in u_email or any(query_lower in tag for tag in combined_tags):
                    filtered_users.append(u)
                    
            display_users = filtered_users
            list_title_caption = f"🔍 Search Results ({len(display_users)}):"

        with st.container(height=600):
            if not display_users:
                st.caption("No developers found.")
            else:
                st.markdown(f"<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;'>{list_title_caption}</div>", unsafe_allow_html=True)
                
                active_users_bar(
                    display_users, 
                    current_user['email'],
                    st.session_state.get("active_chat_user", {}).get("email")
                )