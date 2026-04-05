import streamlit as st
import pandas as pd
import numpy as np
import os
from src.core import feature_analysis as fan

# =========================================================
# 🔐 STATE INITIALIZATION
# =========================================================
def init_state():
    if "data" not in st.session_state:
        st.session_state["data"] = None

    if "history" not in st.session_state:
        st.session_state["history"] = [] # list of tuples ('change name', df_copy)

    if "data_version" not in st.session_state:
        st.session_state["data_version"] = 0

    if "column_version" not in st.session_state:
        st.session_state["column_version"] = 0

# =========================================================
# 🔄 CENTRAL UPDATE FUNCTION
# =========================================================
def update_data(df):
    st.session_state["data"] = df.copy()
    st.session_state["data_version"] += 1
    st.session_state["column_version"] += 1

# =========================================================
# 🎯 MAIN APP
# =========================================================
def render():
    init_state()

    st.title("🧾 Data Lab")

    tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload", "✏️ Edit", "📊 Explore", "🧾 History"])

    # =========================================================
    # 📂 UPLOAD
    # =========================================================
    with tab1:
        uploaded_file = st.file_uploader("Upload your csv file", type=["csv"])

        st.subheader("📌 Example Datasets")

        # Create data folder if it doesn't exist (safe)
        if not os.path.exists("data"):
            os.makedirs("data")

        # Get all CSV files in data/ folder
        csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

        if csv_files:
            cols = st.columns(len(csv_files))
            for i, file in enumerate(csv_files):
                with cols[i]:
                    if st.button(f"📄 {file}"):
                        file_path = os.path.join("data", file)
                        df = pd.read_csv(file_path)
                        st.session_state["history"] = [("Example Dataset: " + file, df.copy())]
                        update_data(df)
                        st.rerun()  # refresh to show data
        else:
            st.info("No CSV files found in the /data folder.")

        # --------------------------
        # UPLOADED / EXAMPLE DATA LOGIC
        # --------------------------
        df = st.session_state.get("data")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state["history"] = [("Uploaded File", df.copy())]
            update_data(df)

        if df is not None:
            st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            st.dataframe(df)
            if st.button("🗑️ Release file"):
                init_state()
                st.success("✅ File released")
                st.rerun()
        else:
            if not uploaded_file and len(csv_files) > 0:
                st.info("👆 Upload a CSV or select an example dataset")
            
    # =========================================================
    # ✏️ EDIT
    # =========================================================
    with tab2:
        df = st.session_state.get("data")

        # -----------------------------
        # CREATE DATASET
        # -----------------------------
        if df is None:
            st.warning("No dataset loaded")

            n_rows = st.number_input("Number of rows", min_value=1, value=10)

            if st.button("➕ Create Dataset"):
                df = pd.DataFrame({
                    "x0": ["0"]*n_rows,
                    "x1": ["0"]*n_rows,
                    "y": ["0"]*n_rows
                })

                st.session_state["history"] = [("Created Dataset", df.copy())]
                update_data(df)
                st.rerun()

        else:
            st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

            # -----------------------------
            # DATA EDITOR (FIXED INSTANT REFRESH)
            # -----------------------------
            editor_key = f"editor_{st.session_state.column_version}_{len(st.session_state.data.columns)}"
            edited_df = st.data_editor(
                st.session_state["data"],
                num_rows="dynamic",
                width="stretch",
                key=editor_key
            )

            if not edited_df.equals(st.session_state["data"]):
                st.session_state["history"].append(("Edited Data", st.session_state["data"].copy()))
                update_data(edited_df)

            st.markdown("---")

            # -----------------------------
            # COLUMN OPERATIONS
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                new_col = st.text_input("New column name")
                if st.button("➕ Add Column") and new_col:
                    df_new = st.session_state["data"].copy()
                    df_new[new_col] = np.nan  # default value

                    st.session_state["history"].append((f"Added {new_col} column", df_new.copy()))
                    update_data(df_new)
                    st.rerun()

            with col2:
                del_col = st.selectbox(
                    "Delete column",
                    [""] + list(st.session_state["data"].columns)
                )

                if st.button("🗑️ Delete Column") and del_col:
                    df_new = st.session_state["data"].copy()
                    df_new = df_new.drop(columns=[del_col])

                    st.session_state["history"].append((f"Deleted {del_col} column", df_new.copy()))
                    update_data(df_new)
                    st.rerun()

            # =====================================================
            # ⚡ PYTHON COLUMN GENERATOR
            # =====================================================
            st.markdown("---")
            st.subheader("⚡ Python Column Generator")

            pycol1, pycol2 = st.columns([9, 1])
            user_code = None
            st.info("Available columns: " + ", ".join(df.columns))
            with pycol1:
                user_code = st.text_area("Examples (can use multiline code | use prefix 'col_' in column name to create new columns): n = len(x0) ........|.......... col_x0 = np.arange(-5,5) .........|.......... col_x1 = np.random.randint(-10,15) ...........|........... col_x2 = x0 - x1**2 ",
                                        height=120)

            with pycol2:
                if st.button("▶️ Run Code"):
                    try:
                        df_temp = df.copy()

                        local_env = {col: df_temp[col].values for col in df_temp.columns}

                        global_env = {
                            "__builtins__": {},
                            "np": np,
                            "len": len,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            'list': list,
                            'dict': dict
                        }
                        exec(user_code, global_env, local_env)

                        for key, value in local_env.items():
                            if not key.startswith("col_"):
                                continue  # Only process variables starting with col_
                            
                            col_name = key.replace("col_", "") # Only process variables named col_...
                     
                            # Convert to numpy array (safe handling)
                            if isinstance(value, (list, np.ndarray)):
                                arr = np.array(value)

                                try:
                                    arr = arr.astype(np.int64)
                                except:
                                    arr, le = fan.encode_labels(arr.astype(str))
                                    st.session_state[f"{col_name}_enc"] = le  # Store encoder for potential inverse transform

                                    # show the changes
                                    clean_classes = [str(cls) for cls in le.classes_]
                                    mapping = dict(zip(clean_classes, range(len(le.classes_))))
                                    st.session_state["history"].append((f"Encoded column '{col_name}' with mapping: {mapping}", df_temp.copy()))
                                    st.caption(f"Encoded '{col_name}' with mapping: {mapping}.")
                                
                                df_temp[col_name] = arr.astype(np.int64)

                            # 🔴 FIX 2: Length match check
                            if len(value) != len(df_temp):
                                st.warning(f"⚠️ '{col_name}' length mismatch — skipped")
                                continue

                        st.session_state["history"].append((f"Generated columns [{', '.join(key[4:] for key in local_env if key.startswith('col_'))}] via code", df.copy()))
                        update_data(df_temp)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

                # -----------------------------
                # UNDO
                # -----------------------------
                if st.button("↩️ Undo"):
                    if len(st.session_state["history"]) > 1:
                        st.session_state["history"].pop()
                        prev_change, prev_df = st.session_state["history"][-1]
                        update_data(prev_df)
                        st.rerun()
                    else:
                        st.warning("No more undo steps")

    # =========================================================
    # 📊 EXPLORE
    # =========================================================
    with tab3:
        df = st.session_state.get("data")

        if df is None:
            st.warning("No dataset available")
            return

        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape}")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.write("🔢 Numeric:", numeric_cols)
        st.write("🔤 Categorical:", categorical_cols)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0])

        st.subheader("Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Matrix")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            st.dataframe(corr)
        else:
            st.info("Need at least 2 numeric columns for correlation matrix")

        st.subheader("Preprocessing")

        if st.button("Fill NA with Mean"):
            df_new = df.copy()
            df_new[numeric_cols] = df_new[numeric_cols].fillna(df_new[numeric_cols].mean())
            update_data(df_new)
            st.rerun()

        st.info("Changes apply globally.")

    # =========================================================
    # 🧾 HISTORY
    # =========================================================
    with tab4:
        for i, (change, df_state) in enumerate(st.session_state["history"][::-1]):
            st.write(f"{i+1}. {change}")

if __name__ == "__main__":
    render()
