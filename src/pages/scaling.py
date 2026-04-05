import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def render():
    st.title("Data Scaling")

    # =====================================================
    # 🔐 STATE INIT
    # =====================================================
    if "raw_data" not in st.session_state:
        if "data" in st.session_state:
            st.session_state["raw_data"] = st.session_state["data"].copy()
        else:
            st.warning("Upload data first")
            return

    raw_df = st.session_state["raw_data"]
    df = raw_df.copy()

    # =====================================================
    # 🎯 TARGET
    # =====================================================
    st.subheader("Target Selection")

    target = st.selectbox(
        "Select target column",
        df.columns,
        index=df.columns.get_loc(st.session_state.get("target", df.columns[-1]))
        if "target" in st.session_state else len(df.columns) - 1
    )

    st.session_state["target"] = target

    # =====================================================
    # 📊 NUMERIC FEATURES
    # =====================================================
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    feature_cols = [col for col in numeric_cols if col != target]

    # =====================================================
    # 🎯 FEATURE SCALING
    # =====================================================
    st.subheader("Feature Scaling")

    selected_cols = st.multiselect(
        "Select features to SCALE",
        feature_cols,
        default=feature_cols
    )

    scaler_type = st.selectbox(
        "Feature Scaler",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    # =====================================================
    # 🎯 LABEL SCALING (NEW)
    # =====================================================
    st.subheader("Label Scaling")

    scale_label = st.checkbox("Scale label (for regression models)")

    label_scaler_type = st.selectbox(
        "Label Scaler",
        ["StandardScaler", "MinMaxScaler"],
        disabled=not scale_label
    )

    # =====================================================
    # ⚙️ PROCESSING
    # =====================================================
    df_processed = df.copy()

    scaler_X = None
    scaler_y = None

    # -------- Feature scaling --------
    if scaler_type != "None" and len(selected_cols) > 0:

        scaler_X = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()

        df_processed[selected_cols] = scaler_X.fit_transform(df[selected_cols])

    # -------- Label scaling --------
    if scale_label:

        scaler_y = StandardScaler() if label_scaler_type == "StandardScaler" else MinMaxScaler()

        y_scaled = scaler_y.fit_transform(df[[target]])
        df_processed[target] = y_scaled

    # =====================================================
    # 📊 PREVIEW
    # =====================================================
    st.write("Processed Data Preview")
    st.dataframe(df_processed.head())

    # =====================================================
    # 💾 APPLY
    # =====================================================
    if st.button("💾 Apply Preprocessing"):

        st.session_state["data"] = df_processed

        # ---- Feature scaling info ----
        if scaler_X is not None:
            st.session_state["scaler"] = scaler_X
            st.session_state["scaled_columns"] = selected_cols
        else:
            st.session_state.pop("scaler", None)
            st.session_state.pop("scaled_columns", None)

        # ---- Label scaling info ----
        if scale_label:
            st.session_state["scaler_y"] = scaler_y
            st.session_state["target_scaled"] = True
        else:
            st.session_state.pop("scaler_y", None)
            st.session_state["target_scaled"] = False

        st.success("✅ Changes applied")

    # =====================================================
    # 🔄 RESET
    # =====================================================
    if st.button("♻️ Reset to Original"):

        st.session_state["data"] = st.session_state["raw_data"].copy()

        # Clear all pipeline info
        for key in ["scaler", "scaled_columns", "scaler_y", "target_scaled"]:
            st.session_state.pop(key, None)

        st.success("🔄 Reset successful")
        st.rerun()

        
if __name__ == "__main__":
    render()